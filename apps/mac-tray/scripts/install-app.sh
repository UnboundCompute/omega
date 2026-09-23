#!/bin/zsh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
built_app="$project_dir/.build/app/omega.app"
installed_app="/Applications/omega.app"
bundle_id="com.unboundcompute.omega.tray"
old_requirement=""

if [[ "$installed_app" != "/Applications/omega.app" ]]; then
    echo "Refusing to install to an unexpected destination: $installed_app" >&2
    exit 1
fi

if [[ -e "$installed_app" ]]; then
    old_requirement="$(codesign -d -r- "$installed_app" 2>&1 | sed -n 's/^# designated => //p')"
fi

"$script_dir/package-app.sh"
new_requirement="$(codesign -d -r- "$built_app" 2>&1 | sed -n 's/^# designated => //p')"
permission_identity_changed=false
if [[ -n "$old_requirement" && "$old_requirement" != "$new_requirement" ]]; then
    permission_identity_changed=true
fi

if pgrep -x OmegaTray >/dev/null 2>&1; then
    osascript -e 'tell application id "com.unboundcompute.omega.tray" to quit' >/dev/null 2>&1 || true

    for _ in {1..30}; do
        if ! pgrep -x OmegaTray >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done

    if pgrep -x OmegaTray >/dev/null 2>&1; then
        echo "omega is still running. Quit it from the menu bar, then run this script again." >&2
        exit 1
    fi
fi

if [[ -e "$installed_app" ]]; then
    rm -rf -- "$installed_app"
fi

ditto "$built_app" "$installed_app"
codesign --verify --deep --strict "$installed_app"

if [[ "$permission_identity_changed" == true ]]; then
    tccutil reset ScreenCapture "$bundle_id" >/dev/null
    echo "Reset a stale Screen Recording grant because omega's signing identity changed."
    echo "Choose Capture once and allow omega again; future rebuilds will keep this permission."
fi

open "$installed_app"

echo "Installed and opened $installed_app"
