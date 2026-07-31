from __future__ import annotations

from pathlib import Path
import unittest

from tenminvideomaker.continuation import build_scene_frame_plan
from tenminvideomaker.continuation_workflow import (
    build_assembled_scene_delivery_workflow,
    build_continuation_decode_workflow,
    build_continuation_stage1_workflow,
    build_continuation_stage2_workflow,
)
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.workflow_builder import validate_api_graph

from test_contracts import payload


def nodes_of_type(workflow, class_type):
    return [node for node in workflow.values() if node["class_type"] == class_type]


class ContinuationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = payload()
        raw["scenes"][0]["estimated_sec"] = 10.0
        self.job = parse_job_payload(raw)
        self.scene = self.job.scenes[0]
        self.frame = Path(
            r"D:\LTX_Supervisor_Storage\jobs\job\scenes\scene_0001\frame.png"
        )
        self.plan = build_scene_frame_plan(
            job_id=self.job.job_id,
            scene_id=self.scene.scene_id,
            revision=1,
            requested_duration_seconds=self.scene.estimated_sec,
            base_seed=self.scene.i2v.seed,
            fallback_prompt=self.scene.i2v.prompt,
            fallback_negative=self.scene.i2v.negative,
        )

    def test_initial_stage1_is_plain_video_lcm_and_checkpoints_bounded_handoff(self):
        build = build_continuation_stage1_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
        )
        self.assertEqual(validate_api_graph(build.api), ())
        self.assertEqual(len(nodes_of_type(build.api, "SamplerCustom")), 1)
        self.assertFalse(nodes_of_type(build.api, "SamplerCustomAdvanced"))
        self.assertFalse(nodes_of_type(build.api, "CFGGuider"))
        self.assertFalse(nodes_of_type(build.api, "STGGuiderAdvanced"))
        self.assertFalse(nodes_of_type(build.api, "LTXVConcatAVLatent"))
        saver = build.api[build.output_node_id]
        self.assertEqual(saver["class_type"], "10MinVideoMaker_SaveChunkLatent")
        self.assertEqual(saver["inputs"]["artifact_kind"], "stage1_handoff")
        bounded = nodes_of_type(build.api, "LTXVSelectLatents")
        self.assertEqual(len(bounded), 1)
        self.assertEqual(
            (bounded[0]["inputs"]["start_index"], bounded[0]["inputs"]["end_index"]),
            (-16, -1),
        )
        self.assertEqual(
            nodes_of_type(build.api, "EmptyLTXVLatentVideo")[0]["inputs"]["length"],
            121,
        )

    def test_extension_uses_official_24_frame_overlap_and_96_new_transitions(self):
        build = build_continuation_stage1_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[1],
            revision=1,
            attempt_number=1,
            previous_attempt_number=1,
        )
        self.assertEqual(validate_api_graph(build.api), ())
        extension = nodes_of_type(build.api, "LTXVExtendSampler")[0]["inputs"]
        self.assertEqual(extension["num_new_frames"], 96)
        self.assertEqual(extension["frame_overlap"], 24)
        self.assertEqual(extension["strength"], 0.5)
        loader = nodes_of_type(
            build.api,
            "10MinVideoMaker_LoadChunkLatent",
        )[0]["inputs"]
        self.assertEqual((loader["chunk_index"], loader["attempt_number"]), (0, 1))
        self.assertEqual(loader["artifact_kind"], "stage1_handoff")
        self.assertEqual(loader["expected_temporal_tokens"], 16)
        saver = nodes_of_type(
            build.api,
            "10MinVideoMaker_SaveChunkLatent",
        )[0]["inputs"]
        self.assertNotIn("expected_temporal_tokens", saver)
        selections = nodes_of_type(build.api, "LTXVSelectLatents")
        self.assertEqual(len(selections), 1)
        self.assertEqual(
            [
                (node["inputs"]["start_index"], node["inputs"]["end_index"])
                for node in selections
            ],
            [(-17, -1)],
        )
        self.assertFalse(nodes_of_type(build.api, "LTXVConcatAVLatent"))

    def test_first_refinement_window_is_121_frames_and_raw_only(self):
        build = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
        )
        self.assertEqual(validate_api_graph(build.workflow.api), ())
        self.assertFalse(nodes_of_type(build.workflow.api, "LTXVSelectLatents"))
        handoff = nodes_of_type(
            build.workflow.api,
            "10MinVideoMaker_LoadChunkLatent",
        )[0]["inputs"]
        self.assertEqual(handoff["artifact_kind"], "stage1_handoff")
        audio = nodes_of_type(
            build.workflow.api,
            "LTXVEmptyLatentAudio",
        )[0]["inputs"]
        self.assertEqual(audio["frames_number"], 121)
        self.assertFalse(nodes_of_type(build.workflow.api, "DaSiWa_Watermark"))
        self.assertFalse(nodes_of_type(build.workflow.api, "DiscordSendSaveVideo"))
        checkpoints = nodes_of_type(
            build.workflow.api,
            "10MinVideoMaker_SaveChunkLatent",
        )
        self.assertEqual(
            {node["inputs"]["artifact_kind"] for node in checkpoints},
            {"stage2_video", "stage2_audio"},
        )
        self.assertEqual(
            build.workflow.api[build.audio_checkpoint_node_id]["inputs"][
                "artifact_kind"
            ],
            "stage2_audio",
        )
        combine = build.workflow.api[build.workflow.output_node_id]["inputs"]
        self.assertFalse(combine["save_output"])
        self.assertEqual(combine["images"][0], next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "LTXVSpatioTemporalTiledVAEDecode"
        ))
        decoder = nodes_of_type(
            build.workflow.api,
            "LTXVSpatioTemporalTiledVAEDecode",
        )[0]["inputs"]
        self.assertEqual(
            (
                decoder["spatial_tiles"],
                decoder["spatial_overlap"],
                decoder["temporal_tile_length"],
                decoder["temporal_overlap"],
            ),
            (4, 1, 16, 1),
        )

    def test_later_refinement_has_causal_preroll_and_25_frame_visible_guide(self):
        prior = Path(
            r"D:\LTX_Supervisor_Storage\jobs\job\chunks\chunk_0000\window.mkv"
        )
        build = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[1],
            revision=1,
            attempt_number=1,
            previous_attempt_number=1,
            previous_chunk_path=prior,
        )
        self.assertFalse(nodes_of_type(build.workflow.api, "LTXVSelectLatents"))
        audio = nodes_of_type(
            build.workflow.api,
            "LTXVEmptyLatentAudio",
        )[0]["inputs"]
        self.assertEqual(audio["frames_number"], 129)
        handoff = nodes_of_type(
            build.workflow.api,
            "10MinVideoMaker_LoadChunkLatent",
        )[0]["inputs"]
        self.assertEqual(handoff["expected_temporal_tokens"], 17)
        loader = nodes_of_type(build.workflow.api, "VHS_LoadVideoPath")[0]["inputs"]
        self.assertEqual(loader["video"], str(prior))
        self.assertEqual(loader["frame_load_cap"], 25)
        self.assertEqual(loader["skip_first_frames"], 96)
        guide = nodes_of_type(
            build.workflow.api,
            "LTXVAddGuide",
        )[0]["inputs"]
        self.assertEqual(guide["strength"], 1.0)
        self.assertEqual(guide["frame_idx"], 8)
        self.assertEqual(guide["image"][0], next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "VHS_LoadVideoPath"
        ))
        self.assertEqual(validate_api_graph(build.workflow.api), ())

    def test_decode_only_resume_loads_both_checkpoints_without_diffusion(self):
        build = build_continuation_decode_workflow(
            self.job,
            self.scene,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
        )
        self.assertEqual(validate_api_graph(build.api), ())
        loaders = nodes_of_type(
            build.api,
            "10MinVideoMaker_LoadChunkLatent",
        )
        self.assertEqual(
            {node["inputs"]["artifact_kind"] for node in loaders},
            {"stage2_video", "stage2_audio"},
        )
        self.assertFalse(nodes_of_type(build.api, "SamplerCustom"))
        self.assertFalse(nodes_of_type(build.api, "LTXVExtendSampler"))
        self.assertFalse(nodes_of_type(build.api, "CLIPTextEncode"))
        self.assertFalse(nodes_of_type(build.api, "LoraLoaderModelOnly"))
        combine = build.api[build.output_node_id]["inputs"]
        self.assertEqual(combine["format"], "video/ffv1-mkv")
        self.assertEqual(combine["pix_fmt"], "yuv444p")
        self.assertFalse(combine["save_output"])

    def test_short_final_extension_keeps_causal_preroll(self):
        prior = Path(
            r"D:\LTX_Supervisor_Storage\jobs\job\chunks\chunk_0001\window.mkv"
        )
        build = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[2],
            revision=1,
            attempt_number=1,
            previous_attempt_number=1,
            previous_chunk_path=prior,
        )
        self.assertFalse(nodes_of_type(build.workflow.api, "LTXVSelectLatents"))
        audio = nodes_of_type(
            build.workflow.api,
            "LTXVEmptyLatentAudio",
        )[0]["inputs"]
        self.assertEqual(audio["frames_number"], 57)
        loader = nodes_of_type(build.workflow.api, "VHS_LoadVideoPath")[0]["inputs"]
        self.assertEqual(
            loader["skip_first_frames"],
            104,
            "later guides must include the prior raw chunk's eight-frame preroll",
        )

    def test_assembled_delivery_watermarks_only_the_discord_branch(self):
        raw = Path(
            r"D:\LTX_Supervisor_Storage\jobs\job\scenes\scene_0001\video.mp4"
        )
        build = build_assembled_scene_delivery_workflow(
            self.job,
            self.scene,
            raw,
            "https://discord.com/api/webhooks/123/token",
        )
        self.assertEqual(validate_api_graph(build.api), ())
        loaded = nodes_of_type(build.api, "VHS_LoadVideoPath")[0]
        watermark = nodes_of_type(build.api, "DaSiWa_Watermark")[0]
        sender = nodes_of_type(build.api, "DiscordSendSaveVideo")[0]
        self.assertEqual(loaded["inputs"]["video"], str(raw))
        self.assertEqual(sender["inputs"]["images"][0], next(
            node_id for node_id, node in build.api.items() if node is watermark
        ))
        self.assertFalse(sender["inputs"]["save_output"])
        self.assertTrue(sender["inputs"]["send_to_discord"])


if __name__ == "__main__":
    unittest.main()
