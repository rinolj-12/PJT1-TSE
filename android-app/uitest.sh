#!/usr/bin/env bash
# Drive the app on a connected device/emulator: tap buttons by their text, read logcat.
export MSYS_NO_PATHCONV=1   # keep /sdcard paths as they are (Git Bash)
ADB="${ADB:-adb}"   # adb from the Android SDK platform-tools (set ADB=/path/to/adb to override)
tap() {  # tap the first view whose text contains $1 (scrolls down until found)
  for attempt in 1 2 3 4 5 6; do
    "$ADB" shell uiautomator dump /sdcard/ui.xml > /dev/null 2>&1
    b=$("$ADB" shell cat /sdcard/ui.xml | tr '>' '\n' | grep -F "text=\"$1" | head -1 | grep -o 'bounds="[^"]*"' | grep -o '[0-9]\+' | tr '\n' ' ')
    if [ -n "$b" ]; then set -- $b; x=$(( ($1 + $3) / 2 )); y=$(( ($2 + $4) / 2 )); "$ADB" shell input tap $x $y; return 0; fi
    "$ADB" shell input swipe 540 1700 540 900 300
  done
  echo "NOT FOUND: $1"; return 1
}
wait_status() {  # wait until logcat shows $1
  for i in $(seq 1 ${2:-60}); do "$ADB" logcat -d -s TSEEdge:I | grep -qF "$1" && return 0; sleep 2; done
  echo "TIMEOUT waiting for: $1"; return 1
}
tapx() {  # tap the n-th ($2, default 1) view whose text is exactly $1
  local k=${2:-1}
  "$ADB" shell uiautomator dump /sdcard/ui.xml > /dev/null 2>&1
  b=$("$ADB" shell cat /sdcard/ui.xml | tr '>' '\n' | grep -F "text=\"$1\"" | sed -n "${k}p" | grep -o 'bounds="[^"]*"' | grep -o '[0-9]\+' | tr '\n' ' ')
  [ -z "$b" ] && { echo "NOT FOUND: $1 #$k"; return 1; }
  set -- $b; "$ADB" shell input tap $(( ($1 + $3) / 2 )) $(( ($2 + $4) / 2 ))
}
