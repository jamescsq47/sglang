"""Verify observed source/destination recurrent-state hashes in smoke logs."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    pattern = re.compile(r"mamba_digest phase=(\S+) snapshot=(\S+) digest=(\S+)")
    snapshots = defaultdict(lambda: defaultdict(set))
    errors = []
    for path in sorted((args.run_dir / "logs").glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                phase, snapshot, digest = match.groups()
                rank = re.search(r"\bTP(?: Rank)?\s*(\d+)\b", line)
                rank_id = rank.group(1) if rank else "0"
                snapshots[(snapshot, rank_id)][phase].add(digest)
            if any(term in line for term in ("Traceback (most recent call last)", "Scheduler hit an exception", "memory leak")):
                errors.append({"file": str(path), "line": line})
    pairs = []
    missing_sources = []
    for (snapshot, rank), phases in snapshots.items():
        for source, destination in (
            ("p2d_source", "p2d_received"),
            ("d2p_source", "d2p_direct_received"),
            ("d2p_source", "d2p_host_received"),
        ):
            if destination not in phases:
                continue
            if source not in phases:
                missing_sources.append({"snapshot": snapshot, "destination": destination})
            pairs.append({"snapshot": snapshot, "tp_rank": rank, "source": source, "destination": destination,
                          "exact": len(phases[source]) == 1 and phases[source] == phases[destination]})
    result = {"pairs": pairs, "missing_sources": missing_sources,
              "engine_error_markers": errors,
              "all_observed_pairs_exact": bool(pairs) and all(p["exact"] for p in pairs),
              "note": "Only observed paths; absence of a path is not a pass for that path."}
    output = args.run_dir / "mamba-digest-comparison.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
