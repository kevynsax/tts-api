#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

APP="ChatterboxTTS.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

swiftc -O -swift-version 5 \
  -target arm64-apple-macosx13.0 \
  -o "$APP/Contents/MacOS/ChatterboxTTS" \
  ChatterboxTTS.swift \
  -framework Cocoa -framework ServiceManagement

cp Info.plist "$APP/Contents/Info.plist"
codesign --force --sign - "$APP" >/dev/null 2>&1 || true

echo "Built $(pwd)/$APP"
