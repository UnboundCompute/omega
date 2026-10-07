#!/bin/bash
# Pull-based auto-deploy for the core (DL-074). Run by omega-autodeploy.timer.
#
# The VM asks GHCR what `main` is; GitHub holds no way into this box. A new
# digest is deployed as: pull → stop (a turn gets its grace) → back up the
# store → start pinned by digest → health check. A failed check moves the new
# store aside, restores the backup and starts the previous digest again, and
# that digest is never tried twice.
#
# State, all under /var/lib/omega-deploy:
#   failed     digests that failed a health check (skipped forever)
#   backups/   one store copy per deploy, newest $KEEP kept
# The deployed digest itself is the pin in $COMPOSE_DIR/.env, which is also
# what a hand-run `docker compose up -d` uses.

set -euo pipefail

REPO=unboundcompute/omega
IMAGE=ghcr.io/$REPO
COMPOSE_DIR=${COMPOSE_DIR:?set by the unit}
STORE=/var/lib/omega
STATE=/var/lib/omega-deploy
KEEP=5
# Overridable only so a rollback can be rehearsed: point the check at a port
# nothing listens on and every deploy fails it.
PORT=${OMEGA_HEALTH_PORT:-7717}

log() { echo "autodeploy: $*"; }

mkdir -p "$STATE/backups"
touch "$STATE/failed"
exec 9>"$STATE/lock"
flock -n 9 || { log "another run holds the lock"; exit 0; }

cd "$COMPOSE_DIR"

# What `main` points at, without pulling a layer: one token, one HEAD.
token=$(curl -fsS "https://ghcr.io/token?scope=repository:$REPO:pull" | jq -r .token)
remote=$(curl -fsSI \
    -H "Authorization: Bearer $token" \
    -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/$REPO/manifests/main" \
    | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:" {print $2}')
case "$remote" in
    sha256:*) ;;
    *) log "could not read the digest of main (got '$remote')"; exit 1 ;;
esac

current=$(sed -n 's/^OMEGA_IMAGE=.*@//p' .env 2>/dev/null || true)
if [ -z "$current" ]; then
    # First run: pin whatever is running, so a rollback has somewhere to go.
    id=$(docker compose ps -q omega)
    current=$(docker image inspect "$(docker inspect "$id" --format '{{.Image}}')" \
        --format '{{range .RepoDigests}}{{.}} {{end}}' | tr ' ' '\n' | sed -n "s|^$IMAGE@||p" | head -1)
    [ -n "$current" ] || { log "cannot tell which digest is running; not deploying"; exit 1; }
    echo "OMEGA_IMAGE=$IMAGE@$current" > .env
    log "pinned the running image $current"
fi

[ "$remote" = "$current" ] && exit 0
if grep -qx "$remote" "$STATE/failed"; then
    exit 0
fi

# The head the new core must reach. Unreadable means the old core is down,
# and then the bar is only that the new one answers.
head_of() {
    python3 - "$PORT" <<'EOF'
import json, socket, sys
try:
    f = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=5).makefile("rw")
    f.readline()
    f.write(json.dumps({"op": "ping"}) + "\n"); f.flush()
    print(json.loads(f.readline())["head"])
except Exception:
    print("")
EOF
}

before=$(head_of)
log "deploying $remote over $current (head ${before:-unknown})"

docker pull -q "$IMAGE@$remote" >/dev/null

docker compose stop
stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup="$STATE/backups/$stamp-${current#sha256:}"
cp -a "$STORE" "$backup"

echo "OMEGA_IMAGE=$IMAGE@$remote" > .env
# Not fatal: a start that fails is a failed health check, and that rolls back.
docker compose up -d || true

healthy=""
for _ in $(seq 1 30); do
    sleep 2
    now=$(head_of)
    if [ -n "$now" ] && [ "$now" -ge "${before:-0}" ]; then
        healthy=$now
        break
    fi
done

if [ -n "$healthy" ]; then
    log "healthy at head $healthy"
    ls -1d "$STATE"/backups/*/ | head -n -"$KEEP" | xargs -r rm -rf
    exit 0
fi

log "health check failed; rolling back to $current"
echo "$remote" >> "$STATE/failed"
docker compose stop
# Kept, never deleted: whatever the bad build wrote is evidence.
mv "$STORE" "$STATE/backups/$stamp-failed-${remote#sha256:}"
cp -a "$backup" "$STORE"
echo "OMEGA_IMAGE=$IMAGE@$current" > .env
docker compose up -d
log "rolled back; head now $(head_of)"
exit 1
