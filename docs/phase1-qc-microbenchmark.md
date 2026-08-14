# Phase-1 production QC micro-benchmark

The hardware gate uses `scripts/run_phase1_qc_microbenchmark.py` and a local,
untracked manifest. Labels are retained by the scorer and never enter frame
preprocessing, judge requests, or the production rubric.

```json
{
  "schema_version": 1,
  "samples": [
    {"sample_id": "sample-001", "path": "X:/blind/media/a.mp4", "label": "BAD"},
    {"sample_id": "sample-002", "path": "X:/blind/media/b.mp4", "label": "BAD"},
    {"sample_id": "sample-003", "path": "X:/blind/media/c.mp4", "label": "BAD"},
    {"sample_id": "sample-004", "path": "X:/blind/media/d.mp4", "label": "GOOD"}
  ]
}
```

The manifest must contain exactly three `BAD` labels and one `GOOD` label.
Media and evidence paths under live production storage are rejected. The runner
records its deterministic shuffle seed, uses one owned llama backend for all
four clips, and evaluates each clip through the production preprocessing,
windowing, shifted-confirmation, normalization, and evidence policy.

Run without `--execute` to validate only the manifest, paths, and blind order.
The explicit `--execute` flag is required for the later authorized hardware
gate.
