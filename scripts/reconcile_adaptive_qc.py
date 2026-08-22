"""Idempotently resume legacy automatic QC holds without rewriting history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenminvideomaker.state_store import PipelineStateStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    store = PipelineStateStore(args.db.resolve())
    resumed = store.resume_automatic_holds(args.job_id)
    print(json.dumps({"job_id": args.job_id, "resumed": len(resumed), "candidate_ids": list(resumed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
