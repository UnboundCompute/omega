#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
project_dir="${script_dir:h}"
built_app="$project_dir/.build/app/omega.app"
installed_app="/Applications/omega.app"

if [[ "$installed_app" != "/Applications/omega.app" ]]; then
    echo "Refusing to install to an unexpected destination: $installed_app" >&2
    exit 1
fi

"$script_dir/package-app.sh"

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
open "$installed_app"

echo "Installed and opened $installed_app"
