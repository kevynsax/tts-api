#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

APP="TTSServer.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

swiftc -O -swift-version 5 \
  -target arm64-apple-macosx13.0 \
  -o "$APP/Contents/MacOS/TTSServer" \
  TTSServer.swift \
  -framework Cocoa -framework ServiceManagement

[ -f AppIcon.icns ] || ./make-icon.sh
cp AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cp Info.plist "$APP/Contents/Info.plist"
codesign --force --sign - "$APP" >/dev/null 2>&1 || true

echo "Built $(pwd)/$APP"
