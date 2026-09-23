#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
project_dir="${script_dir:h}"
app_dir="$project_dir/.build/app/omega.app"
contents_dir="$app_dir/Contents"

cd "$project_dir"
swift build -c release
bin_dir="$(swift build -c release --show-bin-path)"

rm -rf "$app_dir"
mkdir -p "$contents_dir/MacOS" "$contents_dir/Resources"
cp "$bin_dir/OmegaTray" "$contents_dir/MacOS/OmegaTray"
cp "$project_dir/Resources/Info.plist" "$contents_dir/Info.plist"

codesign --force --deep --sign - "$app_dir"
plutil -lint "$contents_dir/Info.plist"
codesign --verify --deep --strict "$app_dir"

echo "$app_dir"
