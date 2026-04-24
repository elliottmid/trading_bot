#!/bin/zsh
# Wrapper for launchd: runs the daily Coppock scan, logs every run, and sends
# a macOS notification on a new bar or a scan error. Silent when no new bar.
set -u

REPO="/Users/elliottmiddleton/trading-bot"
PY="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
LOG="$HOME/trading-bot-daily.log"

cd "$REPO" || exit 1

OUT=$("$PY" scripts/coppock_daily_scan.py 2>&1)
STATUS=$?
TS=$(date -Iseconds)

printf "%s status=%d %s\n" "$TS" "$STATUS" "${OUT:-<silent>}" >> "$LOG"

if [ "$STATUS" -ne 0 ]; then
  BODY=$(printf '%s' "$OUT" | head -c 200 | tr -d '"')
  /usr/bin/osascript -e "display notification \"$BODY\" with title \"Coppock Scan ERROR\""
elif [ -n "$OUT" ]; then
  BODY=$(printf '%s' "$OUT" | tr -d '"')
  /usr/bin/osascript -e "display notification \"$BODY\" with title \"SPY Coppock\""
fi

exit "$STATUS"
