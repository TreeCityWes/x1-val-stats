#!/usr/bin/env python3
"""
merge_selfstake.py

Drop into the x1-val-stats repo (root level, alongside fetch_validators.py).

Fetches self-stake values from x1.ninja's public Capy-Mon endpoint, then
augments public/data/validators.json with per-validator `selfStake` and
`selfStakeStrict` fields plus a `metadata.selfStake` block.

Run AFTER fetch_validators.py in the GitHub Action workflow:

    - name: Run validator fetch script
      run: python fetch_validators.py --fast || true

    - name: Merge self-stake data from x1.ninja
      run: python merge_selfstake.py || true

The merge is best-effort; non-zero exit codes are suppressed in the workflow
so a transient outage on x1.ninja doesn't block the validator data update.

Definition of self-stake (per upstream):
  selfStake        = strict + operator-style heuristic
  selfStakeStrict  = strict only (withdraw_authority == validator identity)
"""

import json
import sys
import os
import urllib.request
import urllib.error
from pathlib import Path

ENDPOINT = os.environ.get(
    "CAPY_SELFSTAKE_URL",
    "https://x1.ninja/api/capy-mon/selfstake",
)
VALIDATORS_PATH = Path(
    os.environ.get("VALIDATORS_JSON", "public/data/validators.json"),
)
TIMEOUT_SEC = 30


def fetch_selfstake() -> dict:
    req = urllib.request.Request(
        ENDPOINT,
        headers={"Accept": "application/json", "User-Agent": "x1-val-stats-merge/1"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        if resp.status != 200:
            raise RuntimeError(f"x1.ninja returned HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def merge() -> int:
    if not VALIDATORS_PATH.exists():
        print(f"merge_selfstake: {VALIDATORS_PATH} not found — skipping", file=sys.stderr)
        return 0

    try:
        upstream = fetch_selfstake()
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as e:
        print(f"merge_selfstake: fetch failed: {e} — skipping", file=sys.stderr)
        return 0

    self_stake = upstream.get("selfStake") or {}
    self_stake_strict = upstream.get("selfStakeStrict") or {}
    meta = upstream.get("metadata") or {}
    if not self_stake:
        print("merge_selfstake: empty selfStake map — skipping", file=sys.stderr)
        return 0

    with open(VALIDATORS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    validators = data.get("validators", [])
    matched = 0
    for v in validators:
        vote = v.get("votePubkey")
        if not vote:
            continue
        if vote in self_stake:
            try:
                v["selfStake"] = int(self_stake[vote])
                matched += 1
            except (TypeError, ValueError):
                pass
        if vote in self_stake_strict:
            try:
                v["selfStakeStrict"] = int(self_stake_strict[vote])
            except (TypeError, ValueError):
                pass

    # Top-level metadata block — informative, not load-bearing.
    if "metadata" not in data or not isinstance(data["metadata"], dict):
        data["metadata"] = {}
    data["metadata"]["selfStake"] = {
        "source": upstream.get("source"),
        "definition": upstream.get("definition"),
        "lastUpdated": meta.get("lastUpdated"),
        "validatorsWithSelfStake": meta.get("validatorsWithSelfStake"),
        "validatorsWithStrictOnly": meta.get("validatorsWithStrictOnly"),
        "averageLamports": meta.get("averageLamports"),
        "totalLamports": meta.get("totalLamports"),
        "totalLamportsStrict": meta.get("totalLamportsStrict"),
    }

    # Atomic-ish write: write to tmp then rename.
    tmp_path = VALIDATORS_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp_path.replace(VALIDATORS_PATH)

    print(
        f"merge_selfstake: enriched {matched}/{len(validators)} validators "
        f"from {upstream.get('source')} "
        f"(scanner ran {meta.get('lastUpdated')})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(merge())
