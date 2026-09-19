#!/usr/bin/env bash
# Export the current XQuartz cookie for a Docker Desktop X11 client.
# No global xhost access-control changes; the cookie stays outside the repo.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "Run this helper on the Mac hosting XQuartz." >&2
    exit 1
fi
if [[ ! -x /opt/X11/bin/xauth ]]; then
    echo "Install and start XQuartz first (/opt/X11/bin/xauth is missing)." >&2
    exit 1
fi
xquartz_display="${DISPLAY:-$(launchctl getenv DISPLAY)}"
if [[ -z "$xquartz_display" ]]; then
    echo "Start XQuartz, then run this helper from its Applications > Terminal." >&2
    exit 1
fi

auth_dir="$HOME/.cache/fusionpoly"
mkdir -p "$auth_dir"
umask 077
cookie_file="$(mktemp "$auth_dir/cookie.XXXXXX")"
auth_file="$(mktemp "$auth_dir/auth.XXXXXX")"
trap 'rm -f "$cookie_file" "$auth_file"' EXIT
/opt/X11/bin/xauth nlist "$xquartz_display" > "$cookie_file"
if [[ ! -s "$cookie_file" ]]; then
    echo "No cookie for DISPLAY=$xquartz_display. Run in the XQuartz terminal with authentication enabled." >&2
    exit 1
fi
# FamilyWild allows the same cookie with host.docker.internal as the hostname.
sed 's/^..../ffff/' "$cookie_file" | /opt/X11/bin/xauth -f "$auth_file" nmerge -
chmod 600 "$auth_file"
mv "$auth_file" "$auth_dir/Xauthority"
echo "XQuartz cookie exported to $auth_dir/Xauthority (display $xquartz_display)."
