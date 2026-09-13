#!/usr/bin/env sh
# Start the studio. Works on any machine; it detects what is there.
set -e
cd "$(dirname "$0")/backend"
python3 -m pip install -q -r requirements.txt
echo "Studio on http://127.0.0.1:8000"
exec python3 -m uvicorn uls.api.app:app --host 127.0.0.1 --port 8000 "$@"
