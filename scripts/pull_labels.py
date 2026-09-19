"""
Pulls the team's verdicts out of the shared Apps Script sheet into one labels.json.

Also reports inter-rater agreement wherever two people judged the same contact, because a
precision figure built on labels nobody checked is just a different kind of assumption. Any
contact labelled by more than one person is scored for agreement; the overlap scenes in
docs/LABELLING.md exist to make sure there are some.

    python scripts/pull_labels.py                              # endpoint from review/config.js
    python scripts/pull_labels.py --url https://script.google.com/.../exec
    python scripts/pull_labels.py --merge data/outputs/review/labels.json
"""

import os
import re
import sys
import json
import argparse
import collections
import urllib.request
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECISIVE = ("vessel", "not_vessel", "structure")


def endpoint_from_config(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        m = re.search(r'LABEL_ENDPOINT\s*=\s*["\']([^"\']*)["\']', f.read())
    return m.group(1).strip() if m else ""


def fetch(url: str) -> List[Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "spacehack-pull-labels"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("rows", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--review", default="review", help="directory holding config.js")
    ap.add_argument("--output", "-o", default="data/outputs/review/labels.json")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="extra labels.json files to fold in (exports made before the endpoint existed)")
    args = ap.parse_args()

    url = args.url or endpoint_from_config(os.path.join(args.review, "config.js"))
    rows: List[Dict[str, Any]] = []
    if url:
        print(f"[pull] {url}")
        rows = fetch(url)
        print(f"[pull] {len(rows)} rows")
    else:
        print("[pull] no endpoint configured; merging local files only")

    # Append-only log: the last row for an id is that id's current verdict.
    final: Dict[str, str] = {}
    who: Dict[str, str] = {}
    seen: Dict[str, Dict[str, str]] = collections.defaultdict(dict)  # id -> labeller -> verdict
    for row in sorted(rows, key=lambda r: r.get("ts", "")):
        cid, verdict, labeller = row.get("id"), row.get("verdict"), row.get("labeller") or "anonymous"
        if not cid or not verdict:
            continue
        final[cid] = verdict
        who[cid] = labeller
        seen[cid][labeller] = verdict

    for path in args.merge:
        if not os.path.exists(path):
            print(f"[merge] missing, skipped: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f).get("labels", {})
        # Local exports pre-date the shared store, so they lose to anything synced since.
        for cid, verdict in extra.items():
            final.setdefault(cid, verdict)
        print(f"[merge] {len(extra)} from {path}")

    counts = collections.Counter(final.values())
    per_labeller = collections.Counter(who.values())

    # Agreement, over contacts two or more people judged decisively
    pairs = 0
    agreed = 0
    disputes = []
    for cid, verdicts in seen.items():
        decisive = {p: v for p, v in verdicts.items() if v in DECISIVE}
        if len(decisive) < 2:
            continue
        pairs += 1
        if len(set(decisive.values())) == 1:
            agreed += 1
        else:
            disputes.append({"id": cid, "verdicts": decisive})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "labelled": len(final),
        "labels": final,
        "labelled_by": who,
        "per_labeller": dict(per_labeller.most_common()),
        "verdict_counts": dict(counts),
        "agreement": {
            "double_labelled": pairs,
            "agreed": agreed,
            "rate": round(agreed / pairs, 3) if pairs else None,
            "disputes": disputes[:50],
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    print(f"\n{len(final)} verdicts: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    if per_labeller:
        print("by labeller: " + ", ".join(f"{k}={v}" for k, v in per_labeller.most_common()))
    if pairs:
        print(f"agreement: {agreed}/{pairs} double-labelled contacts agree ({100 * agreed / pairs:.0f}%)")
        for d in disputes[:5]:
            print(f"  disputed {d['id']}: " + ", ".join(f"{p}={v}" for p, v in d["verdicts"].items()))
    else:
        print("agreement: no contact has been labelled by two people yet")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
