#!/bin/zsh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
app_dir="$project_dir/.build/app/omega.app"
contents_dir="$app_dir/Contents"

cd "$project_dir"
swift build -c release
bin_dir="$(swift build -c release --show-bin-path)"

rm -rf "$app_dir"
mkdir -p "$contents_dir/MacOS" "$contents_dir/Resources"
cp "$bin_dir/OmegaTray" "$contents_dir/MacOS/OmegaTray"
cp "$project_dir/Resources/Info.plist" "$contents_dir/Info.plist"

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
