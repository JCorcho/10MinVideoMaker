# 10MinVideoMaker

An independent ComfyUI custom-node project for building a guided long-form video workflow.

## Current status

The durable job contract, SQLite state machine, Gmail transport, LoRA resolver, FFmpeg assembly service, eight
interactive ComfyUI nodes, and scene-specific Anima/Pony/LTX workflow builders are implemented. The unattended
supervisor is still under construction; no production render has been run.

## Available nodes

- **Validate Job** — validates and normalizes the exact Grok JSON contract.
- **Pipeline Status** — reads the durable pipeline state.
- **Request Grok Job** — sends the exact request subject using environment-provided Gmail credentials.
- **Poll Gmail Once** — performs one attachment-first IMAP poll; scheduling belongs to the supervisor.
- **Resolve LoRAs** — checks local LoRA roots, resolves dynamic assets, and verifies mandatory I2V LoRAs.
- **Release Memory** — runs Python and CUDA cache cleanup.
- **Save Scene Frame** — atomically caches a deterministic 704×1248 PNG for the matching scene.
- **Stitch Clips** — verifies every clip is 704×1248 at 24 fps before FFmpeg concat.

Nodes that access Gmail, download assets, or stitch video have side effects and should only be queued deliberately.

## Installation

The repository is intended to live at:

`ComfyUI/custom_nodes/10MinVideoMaker`

Restart ComfyUI after adding or changing a Python node module. Once nodes are introduced, ComfyUI will discover the
pack through `NODE_CLASS_MAPPINGS` in `__init__.py`.

This installation currently discovers this package through the legacy V1 mappings. The node implementations remain
thin wrappers over framework-independent services so the automation supervisor and ComfyUI surface share behavior.

## Repository layout

- `__init__.py` — ComfyUI custom-node package entry point.
- `workflows/` — versioned ComfyUI workflow JSON files.
- `examples/` — a safe example of the exact incoming JSON contract.
- `tests/` — focused regression tests for node and routing behavior.
- `AI_DEVELOPMENT_RULES.md` — persistent implementation, validation, and documentation rules.

## Development baseline

Before implementing a node or workflow, inspect its live contract through the local ComfyUI API. Validate workflow
routing without rendering before running any expensive generation. See `AI_DEVELOPMENT_RULES.md` for the detailed
project rules and verification commands.

See `docs/user-guide.md` for workflow locations, Gmail environment variables, and the current no-render boundary.
