#!/usr/bin/env bash
# Double-click me on macOS (or run me on Linux). Serves the review pages and opens a browser.
cd "$(dirname "$0")" || exit 1

for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" serve.py
    fi
done

echo
echo "  Python 3 was not found."
echo "  macOS:  xcode-select --install   (or install from python.org)"
echo "  Linux:  sudo apt install python3"
echo "  No packages are needed, just the interpreter."
echo
read -r -p "Press Enter to close..."
