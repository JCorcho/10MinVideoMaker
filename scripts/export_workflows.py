"""Export versioned API and GUI workflow templates using the live ComfyUI contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.artifacts import scene_frame_path
from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.delivery import (
    DiscordDeliverySettings,
    TEMPLATE_WEBHOOK_PLACEHOLDER,
)
from tenminvideomaker.workflow_builder import build_i2v_api_workflow, build_t2i_api_workflow
from tenminvideomaker.workflow_export import api_to_gui_workflow, inspect_gui_workflow

DEFAULT_SHARED_WORKFLOW_ROOT = Path(
    r"C:\AI\ComfyUI\ComfyUI-Easy-Install\ComfyUI-Easy-Install"
    r"\ComfyUI\user\default\workflows\10minvideomaker"
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _replace_value(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _replace_value(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_value(item, old, new) for item in value]
    return new if value == old else value


def export(*, shared_root: Path | None) -> dict[str, dict]:
    with urlopen("http://127.0.0.1:8188/object_info", timeout=30) as response:
        object_info = json.load(response)
    raw = json.loads((PROJECT_ROOT / "examples" / "example_job.json").read_text(encoding="utf-8"))
    template_delivery = DiscordDeliverySettings(TEMPLATE_WEBHOOK_PLACEHOLDER)
    runtime_delivery = DiscordDeliverySettings.from_environment(
        load_project_environment(PROJECT_ROOT),
        required=False,
    )
    anima_job = parse_job_payload(raw)
    anima = build_t2i_api_workflow(
        anima_job,
        anima_job.scenes[0],
        delivery=template_delivery,
    )

    pony_raw = json.loads(json.dumps(raw))
    pony_raw["character"]["lora"]["base"] = "Pony"
    pony_raw["character"]["lora"]["name"] = "Example Character Pony"
    pony_raw["scenes"][0]["t2i"]["loras"][0]["name"] = "Example Character Pony"
    pony_job = parse_job_payload(pony_raw)
    pony = build_t2i_api_workflow(
        pony_job,
        pony_job.scenes[0],
        delivery=template_delivery,
    )

    frame_path = scene_frame_path(anima_job.job_id, anima_job.scenes[0].scene_id)
    i2v = build_i2v_api_workflow(
        anima_job,
        anima_job.scenes[0],
        frame_path,
        delivery=template_delivery,
    )
    builds = {
        "10MinVideoMaker_T2I_Anima": anima,
        "10MinVideoMaker_T2I_Pony": pony,
        "10MinVideoMaker_I2V_LTX23_TwoPass": i2v,
    }
    report: dict[str, dict] = {}
    for name, build in builds.items():
        api_path = PROJECT_ROOT / "workflows" / f"{name}.api.json"
        gui_path = PROJECT_ROOT / "workflows" / f"{name}.json"
        title = name.replace("_", " ")
        gui = api_to_gui_workflow(build.api, object_info, title=title)
        inspection = inspect_gui_workflow(gui)
        _write_json(api_path, build.api)
        _write_json(gui_path, gui)
        if shared_root is not None:
            shared_root.mkdir(parents=True, exist_ok=True)
            shared_gui = (
                _replace_value(
                    gui,
                    TEMPLATE_WEBHOOK_PLACEHOLDER,
                    runtime_delivery.webhook_url,
                )
                if runtime_delivery
                else gui
            )
            _write_json(shared_root / gui_path.name, shared_gui)
        report[name] = {
            "api": str(api_path),
            "gui": str(gui_path),
            "node_count": len(build.api),
            "layout": inspection,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=None,
        help="Copy GUI workflows into an approved shared ComfyUI workflow directory.",
    )
    parser.add_argument(
        "--install-approved-shared-copies",
        action="store_true",
        help=f"Copy GUI workflows into {DEFAULT_SHARED_WORKFLOW_ROOT}",
    )
    args = parser.parse_args()
    if args.shared_root and args.install_approved_shared_copies:
        parser.error("Choose only one shared workflow target.")
    shared_root = DEFAULT_SHARED_WORKFLOW_ROOT if args.install_approved_shared_copies else args.shared_root
    print(json.dumps(export(shared_root=shared_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
