# The core omega image (DL-069). Three stages, and the order is the point:
#
#   build — compile the PyO3 extension into a wheel.
#   test  — run the whole suite, Rust and Python, on Linux. CI otherwise only
#           ever tests macOS, and this image is the first thing that runs omega
#           anywhere else.
#   final — the wheel on a slim Python. It copies a marker out of `test`, so
#           BuildKit cannot produce this stage without the suite going green.
#
# The repo is public, so the image is public: no secret is ever copied in.
# OPENAI_API_KEY (and the adapter's SARVAM_API_KEY) arrive at run time from the
# host (see deploy/compose.yaml).

FROM python:3.11-slim-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

RUN pip install --no-cache-dir "maturin>=1.7,<2.0"

# `.cargo/config.toml` pins PYO3_PYTHON to the checkout's `.venv`, which does
# not exist here; an exported value wins over it (`force = false`).
ENV PYO3_PYTHON=/usr/local/bin/python3.11

WORKDIR /src
COPY Cargo.toml Cargo.lock pyproject.toml ./
COPY .cargo .cargo
COPY src src
COPY python python
# The wheel is built for, and only ever installed into, this same base image,
# so a plain `linux` tag is honest; a manylinux claim would need auditing.
RUN maturin build --release --compatibility linux --out /wheels


FROM build AS test

RUN pip install --no-cache-dir pytest /wheels/*.whl
COPY tests tests

# Not root: the lock and permission cases mean nothing to a user that
# bypasses file modes.
RUN useradd --create-home tester && chown -R tester /src
USER tester

# `cargo test` links libpython, so the extension-module feature stays off
# here, as in ci.yml. `omega` imports from the installed wheel, not python/.
RUN cargo test \
    && pytest -q \
    && touch /src/.linux-suite-green


FROM python:3.11-slim-bookworm

COPY --from=test /src/.linux-suite-green /etc/omega.linux-suite-green
# ffmpeg for the Discord adapter (DL-079): Sarvam takes 30 s a request, so
# audio sent over Discord is cut into 25 s pieces here first. Not in `test`:
# the suite fakes it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
# With the `discord` extra: the Discord adapter (DL-073) runs from this same
# image as a second service, `python -m omega.discord`. The loop is there
# because `[discord]` after a glob would be read as a character class.
RUN for wheel in /wheels/*.whl; do pip install --no-cache-dir "${wheel}[discord]"; done \
    && rm -rf /wheels

# The store is a host mount (DL-069 #4); the uid is fixed so the host
# directory can be chowned to it once. The Discord adapter runs as the same
# user, so its cursor directory is chowned to the same uid.
RUN useradd --system --uid 10001 --home-dir /data omega \
    && mkdir /data && chown omega /data
USER omega
VOLUME /data

ENV PYTHONUNBUFFERED=1

# Loopback only, as always: the container runs with host networking and
# `tailscale serve` on the VM is the only way in (DL-069 #2).
ENTRYPOINT ["python", "-m", "omega", "--serve", "--store", "/data"]
