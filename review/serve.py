"""
Serves the contact review pages and opens a browser at them. Stdlib only.

Launched by start-labelling.cmd (Windows) or start-labelling.command (macOS/Linux) so a
teammate can clone the repo and double-click one file. Serving over http rather than opening
the HTML directly matters: a file:// page cannot reach the shared label store, so verdicts
would silently stay on that laptop.

    python review/serve.py            # or just double-click the launcher
"""

import os
import sys
import socket
import threading
import webbrowser
import http.server
import socketserver

HERE = os.path.dirname(os.path.abspath(__file__))
PREFERRED = 8771


def free_port(start: int) -> int:
    for port in range(start, start + 40):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0  # let the OS choose


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):  # a teammate does not need a request log
        pass

    def end_headers(self):
        # The pages are rebuilt often; a cached copy would hide new contacts.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    if not os.path.exists(os.path.join(HERE, "index.html")):
        print("No index.html here. Run: python scripts/build_contact_sheet.py -s data/outputs/us_scorecard -o review --split")
        input("\nPress Enter to close...")
        return 1

    endpoint = ""
    config = os.path.join(HERE, "config.js")
    if os.path.exists(config):
        with open(config, "r", encoding="utf-8") as f:
            text = f.read()
        endpoint = text.split('LABEL_ENDPOINT = "', 1)[-1].split('"', 1)[0] if 'LABEL_ENDPOINT = "' in text else ""

    port = free_port(PREFERRED)
    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://127.0.0.1:{port}"

    print("=" * 62)
    print("  IRIS - contact review")
    print("=" * 62)
    print(f"  {url}")
    print(f"  shared label store: {'ON' if endpoint else 'OFF - verdicts stay in this browser'}")
    if not endpoint:
        print("    (ask Jack for the endpoint URL, then put it in review/config.js)")
    print("\n  Type your name at the top of the page before you start labelling.")
    print("  Keep this window open while you label. Ctrl+C or close it when done.")
    print("=" * 62)

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
