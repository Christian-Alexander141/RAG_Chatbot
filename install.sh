#!/usr/bin/env sh
# Starter für Linux, WSL und macOS. Die eigentliche Arbeit macht install.py.
# Aufruf: ./install.sh [--dev] [--no-image] [--force]
set -e
cd "$(dirname "$0")"

for candidate in "${PYTHON:-}" python3 python; do
    if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" install.py "$@"
    fi
done

echo "Python wurde nicht gefunden. Bitte Python 3.10 oder neuer installieren," >&2
echo "unter Debian/Ubuntu (auch WSL): sudo apt install python3 python3-venv" >&2
exit 1
