"""Convert project API graphs into deterministic, inspectable ComfyUI GUI workflows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .constants import PRODUCTION_FPS, PRODUCTION_HEIGHT, PRODUCTION_WIDTH
from .workflow_builder import validate_against_object_info

_NODE_WIDTH = 360
_MIN_NODE_HEIGHT = 220
_COLUMN_GAP = 160
_ROW_GAP = 80
_GROUP_PADDING = 80
_SEED_CONTROL_WIDGET_NODES = frozenset({"KSampler", "KSamplerAdvanced", "FaceDetailer"})


class WorkflowExportError(ValueError):
    """Raised when a GUI workflow cannot be exported safely."""


def _input_type(name: str, spec: Any | None) -> Any:
    if spec is not None:
        return spec[0]
    if name.endswith(".image_1"):
        return "IMAGE"
    if name.endswith(".strength_1"):
        return "FLOAT"
    if name.endswith(".index_1"):
        return "INT"
    return {
        "pix_fmt": ["yuv420p", "yuv420p10le"],
        "crf": "INT",
        "save_metadata": "BOOLEAN",
        "trim_to_audio": "BOOLEAN",
    }.get(name, "STRING")


def _ordered_input_names(node: Mapping[str, Any], info: Mapping[str, Any]) -> list[str]:
    present = list(node["inputs"])
    order: list[str] = []
    input_order = info.get("input_order", {})
    for kind in ("required", "optional", "hidden"):
        for name in input_order.get(kind, []):
            if name in node["inputs"] and name not in order:
                order.append(name)
    order.extend(name for name in present if name not in order)
    return order


def _depths(api: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    depths: dict[str, int] = {}
    unresolved = set(api)
    while unresolved:
        progressed = False
        for node_id in list(unresolved):
            dependencies = {
                value[0]
                for value in api[node_id]["inputs"].values()
                if isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and value[0] in api
            }
            if dependencies.issubset(depths):
                depths[node_id] = 0 if not dependencies else max(depths[item] for item in dependencies) + 1
                unresolved.remove(node_id)
                progressed = True
        if not progressed:
            raise WorkflowExportError("Workflow contains a dependency cycle.")
    return depths


def _layout(
    nodes: list[dict[str, Any]],
    depths: Mapping[str, int],
) -> None:
    columns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        columns[depths[str(node["id"])]].append(node)
    for depth, column in columns.items():
        y = 0
        for node in column:
            node["pos"] = [depth * (_NODE_WIDTH + _COLUMN_GAP), y]
            y += node["size"][1] + _ROW_GAP


def _fit_group(nodes: list[dict[str, Any]], title: str) -> dict[str, Any]:
    left = min(node["pos"][0] for node in nodes) - _GROUP_PADDING
    top = min(node["pos"][1] for node in nodes) - _GROUP_PADDING
    right = max(node["pos"][0] + node["size"][0] for node in nodes) + _GROUP_PADDING
    bottom = max(node["pos"][1] + node["size"][1] for node in nodes) + _GROUP_PADDING
    return {
        "id": 1,
        "title": title,
        "bounding": [left, top, right - left, bottom - top],
        "color": "#2b6f8a",
        "font_size": 28,
        "flags": {},
    }


def api_to_gui_workflow(
    api: Mapping[str, Mapping[str, Any]],
    object_info: Mapping[str, Mapping[str, Any]],
    *,
    title: str,
) -> dict[str, Any]:
    """Create a GUI-format workflow after validating against the live node contracts."""
    contract_errors = validate_against_object_info(api, object_info)
    if contract_errors:
        raise WorkflowExportError("; ".join(contract_errors))

    nodes: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    link_id = 0
    links: list[list[Any]] = []

    for order, (node_id, api_node) in enumerate(api.items()):
        class_type = api_node["class_type"]
        info = object_info[class_type]
        required = info.get("input", {}).get("required", {})
        optional = info.get("input", {}).get("optional", {})
        gui_inputs: list[dict[str, Any]] = []
        widgets: list[Any] = []
        for name in _ordered_input_names(api_node, info):
            value = api_node["inputs"][name]
            spec = required.get(name) or optional.get(name)
            gui_input = {"name": name, "type": _input_type(name, spec), "link": None}
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
            ):
                link_id += 1
                gui_input["link"] = link_id
                links.append([link_id, int(value[0]), value[1], int(node_id), len(gui_inputs), None])
            else:
                widgets.append(value)
                if name in {"seed", "noise_seed"} and class_type in _SEED_CONTROL_WIDGET_NODES:
                    # ComfyUI serializes a separate control-after-generate widget
                    # immediately after sampler seed widgets. It is not an API
                    # input, so omitting it shifts every following value on the
                    # canvas (for example, CFG 6 appears as six steps).
                    widgets.append("fixed")
            gui_inputs.append(gui_input)

        output_names = info.get("output_name") or [
            f"output_{index}" for index, _ in enumerate(info.get("output", []))
        ]
        gui_outputs = [
            {
                "name": output_names[index],
                "type": output_type,
                "links": [],
                "slot_index": index,
            }
            for index, output_type in enumerate(info.get("output", []))
        ]
        height = max(_MIN_NODE_HEIGHT, 110 + 28 * len(gui_inputs))
        gui_node = {
            "id": int(node_id),
            "type": class_type,
            "pos": [0, 0],
            "size": [_NODE_WIDTH, height],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": gui_inputs,
            "outputs": gui_outputs,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets,
            "title": api_node.get("_meta", {}).get("title", class_type),
        }
        nodes.append(gui_node)
        node_by_id[node_id] = gui_node

    for link in links:
        source_id = str(link[1])
        source_slot = link[2]
        destination_id = str(link[3])
        destination_slot = link[4]
        source_info = object_info[api[source_id]["class_type"]]
        output_type = source_info["output"][source_slot]
        link[5] = output_type
        node_by_id[source_id]["outputs"][source_slot]["links"].append(link[0])
        node_by_id[destination_id]["inputs"][destination_slot]["type"] = output_type

    depths = _depths(api)
    _layout(nodes, depths)
    group = _fit_group(nodes, title)
    workflow = {
        "last_node_id": max(int(node_id) for node_id in api),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [group],
        "config": {},
        "extra": {
            "ds": {"scale": 0.65, "offset": [120, 120]},
            "10MinVideoMaker": {
                "production_width": PRODUCTION_WIDTH,
                "production_height": PRODUCTION_HEIGHT,
                "fps": PRODUCTION_FPS,
                "generated_from": "tenminvideomaker.workflow_builder",
            },
        },
        "version": 0.4,
    }
    inspection = inspect_gui_workflow(workflow)
    if inspection["overlaps"] or inspection["out_of_group"]:
        raise WorkflowExportError(f"Layout validation failed: {inspection}")
    return workflow


def _rect(node: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x, y = node["pos"]
    width, height = node["size"]
    return (x, y, x + width, y + height)


def _segments_cross(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    def orientation(px, py, qx, qy, rx, ry):
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

    return (
        orientation(ax1, ay1, ax2, ay2, bx1, by1)
        * orientation(ax1, ay1, ax2, ay2, bx2, by2)
        < 0
        and orientation(bx1, by1, bx2, by2, ax1, ay1)
        * orientation(bx1, by1, bx2, by2, ax2, ay2)
        < 0
    )


def inspect_gui_workflow(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """Report node overlaps, approximate link crossings, and group-bound violations."""
    nodes = workflow.get("nodes", [])
    by_id = {node["id"]: node for node in nodes}
    overlaps: list[tuple[int, int]] = []
    for index, left in enumerate(nodes):
        lx1, ly1, lx2, ly2 = _rect(left)
        for right in nodes[index + 1 :]:
            rx1, ry1, rx2, ry2 = _rect(right)
            if lx1 < rx2 and lx2 > rx1 and ly1 < ry2 and ly2 > ry1:
                overlaps.append((left["id"], right["id"]))

    segments: list[tuple[int, tuple[float, float, float, float]]] = []
    for link in workflow.get("links", []):
        source = by_id[link[1]]
        destination = by_id[link[3]]
        source_rect = _rect(source)
        destination_rect = _rect(destination)
        segments.append(
            (
                link[0],
                (
                    source_rect[2],
                    (source_rect[1] + source_rect[3]) / 2,
                    destination_rect[0],
                    (destination_rect[1] + destination_rect[3]) / 2,
                ),
            )
        )
    crossings = [
        (left_id, right_id)
        for index, (left_id, left_segment) in enumerate(segments)
        for right_id, right_segment in segments[index + 1 :]
        if _segments_cross(left_segment, right_segment)
    ]

    out_of_group: list[int] = []
    groups = workflow.get("groups", [])
    if groups:
        gx, gy, width, height = groups[0]["bounding"]
        gright, gbottom = gx + width, gy + height
        for node in nodes:
            x1, y1, x2, y2 = _rect(node)
            if x1 < gx or y1 < gy or x2 > gright or y2 > gbottom:
                out_of_group.append(node["id"])
    return {
        "overlaps": overlaps,
        "crossings": crossings,
        "out_of_group": out_of_group,
        "bounds": groups[0]["bounding"] if groups else None,
    }
