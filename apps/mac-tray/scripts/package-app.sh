#!/bin/zsh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
repo_dir="$(cd -- "$project_dir/../.." && pwd)"
app_dir="$project_dir/.build/app/omega.app"
contents_dir="$app_dir/Contents"
agent_python="$repo_dir/.venv/bin/python"

if [[ ! -x "$agent_python" ]]; then
    echo "Missing omega Python environment at $agent_python" >&2
    echo "Create .venv and install omega before packaging the tray." >&2
    exit 1
fi

cd "$project_dir"
swift build -c release
bin_dir="$(swift build -c release --show-bin-path)"

rm -rf "$app_dir"
mkdir -p "$contents_dir/MacOS" "$contents_dir/Resources"
cp "$bin_dir/OmegaTray" "$contents_dir/MacOS/OmegaTray"
cp "$project_dir/Resources/Info.plist" "$contents_dir/Info.plist"

agent_config="$contents_dir/Resources/AgentLaunch.plist"
plutil -create xml1 "$agent_config"
plutil -insert executable -string "$agent_python" "$agent_config"
plutil -insert workingDirectory -string "$repo_dir" "$agent_config"
plutil -insert arguments -json '["-m","omega","--serve"]' "$agent_config"

signing_identity="${OMEGA_CODESIGN_IDENTITY:-}"
if [[ -z "$signing_identity" ]]; then
    signing_identity="$(security find-identity -v -p codesigning 2>/dev/null | awk '/[0-9]+\) [0-9A-F]+ "/ { print $2; exit }')"
fi

if [[ -z "$signing_identity" ]]; then
    signing_identity="-"
    echo "Warning: no code-signing identity was found; Screen Recording permission will not survive rebuilds." >&2
fi

codesign --force --deep --options runtime --timestamp=none --sign "$signing_identity" "$app_dir"
plutil -lint "$contents_dir/Info.plist"
codesign --verify --deep --strict "$app_dir"

echo "$app_dir"
