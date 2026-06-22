#!/usr/bin/env bash
# Render logo.svg into AppIcon.icns (all required macOS sizes).
set -euo pipefail
cd "$(dirname "$0")"

SVG="logo.svg"
SET="AppIcon.iconset"
rm -rf "$SET" && mkdir -p "$SET"

render() { rsvg-convert -w "$1" -h "$1" "$SVG" -o "$SET/$2"; }

render 16   icon_16x16.png
render 32   icon_16x16@2x.png
render 32   icon_32x32.png
render 64   icon_32x32@2x.png
render 128  icon_128x128.png
render 256  icon_128x128@2x.png
render 256  icon_256x256.png
render 512  icon_256x256@2x.png
render 512  icon_512x512.png
render 1024 icon_512x512@2x.png

iconutil -c icns "$SET" -o AppIcon.icns
rm -rf "$SET"
echo "Built $(pwd)/AppIcon.icns"
