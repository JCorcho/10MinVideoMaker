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
from tenminvideomaker.workflow_builder import LTX_CHECKPOINT, validate_api_graph

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

    def test_later_stage1_uses_exact_extracted_frame_as_a_fresh_121_frame_window(self):
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
        self.assertFalse(nodes_of_type(build.api, "LTXVExtendSampler"))
        self.assertFalse(nodes_of_type(build.api, "10MinVideoMaker_LoadChunkLatent"))
        self.assertEqual(len(nodes_of_type(build.api, "VHS_LoadImagePath")), 1)
        self.assertEqual(
            nodes_of_type(build.api, "EmptyLTXVLatentVideo")[0]["inputs"]["length"],
            121,
        )
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
            [(-16, -1)],
        )
        self.assertFalse(nodes_of_type(build.api, "LTXVConcatAVLatent"))

    def test_stage1_isolates_both_text_conditionings_before_ltx_conditioning(self):
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

        isolates = nodes_of_type(build.api, "10MinVideoMaker_IsolateConditioning")
        self.assertEqual(len(isolates), 2)
        ltx = nodes_of_type(build.api, "LTXVConditioning")[0]["inputs"]
        isolate_ids = {
            node_id
            for node_id, node in build.api.items()
            if node["class_type"] == "10MinVideoMaker_IsolateConditioning"
        }
        self.assertIn(ltx["positive"][0], isolate_ids)
        self.assertIn(ltx["negative"][0], isolate_ids)
        self.assertTrue(
            all("stage1" in node["inputs"]["scope"] for node in isolates)
        )

    def test_stage1_isolates_model_before_and_after_chunk_feed_forward(self):
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

        isolates = nodes_of_type(build.api, "10MinVideoMaker_IsolateModel")
        self.assertEqual(len(isolates), 2)
        isolate_ids = {
            node_id
            for node_id, node in build.api.items()
            if node["class_type"] == "10MinVideoMaker_IsolateModel"
        }
        chunk = nodes_of_type(build.api, "LTXVChunkFeedForward")[0]["inputs"]
        self.assertIn(chunk["model"][0], isolate_ids)
        reference = nodes_of_type(build.api, "LTXReferenceEnable")[0]["inputs"]
        self.assertIn(reference["model"][0], isolate_ids)
        self.assertTrue(
            all("stage1" in node["inputs"]["scope"] for node in isolates)
        )

    def test_continuation_generation_uses_a_forced_fresh_checkpoint_loader(self):
        stage1 = build_continuation_stage1_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[1],
            revision=1,
            attempt_number=1,
            previous_attempt_number=1,
        )
        stage2 = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
        ).workflow
        decode = build_continuation_decode_workflow(
            self.job,
            self.scene,
            self.plan,
            self.plan.chunks[1],
            revision=1,
            attempt_number=1,
        )

        for workflow in (stage1.api, stage2.api, decode.api):
            fresh = nodes_of_type(workflow, "10MinVideoMaker_FreshCheckpoint")
            self.assertEqual(len(fresh), 1)
            self.assertEqual(fresh[0]["inputs"]["ckpt_name"], LTX_CHECKPOINT)
            self.assertTrue(fresh[0]["inputs"]["scope"])
            self.assertFalse(nodes_of_type(workflow, "CheckpointLoaderSimple"))

    def test_continuation_graph_ids_are_scoped_per_stage_and_chunk(self):
        initial = build_continuation_stage1_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
        )
        later = build_continuation_stage1_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[1],
            revision=1,
            attempt_number=1,
            previous_attempt_number=1,
        )
        refinement = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
        ).workflow
        self.assertTrue(set(initial.api).isdisjoint(later.api))
        self.assertTrue(set(initial.api).isdisjoint(refinement.api))
        self.assertEqual(validate_api_graph(initial.api), ())
        self.assertEqual(validate_api_graph(later.api), ())
        self.assertEqual(validate_api_graph(refinement.api), ())

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
        split_id = next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "LTXVSeparateAVLatent"
        )
        video_saver = build.workflow.api[build.video_checkpoint_node_id]["inputs"]
        audio_saver = build.workflow.api[build.audio_checkpoint_node_id]["inputs"]
        self.assertEqual(
            video_saver["latent"],
            [split_id, 0],
            "The durable video checkpoint must retain the native 768x1344 second pass.",
        )
        self.assertEqual(audio_saver["latent"], [split_id, 1])

        combine = build.workflow.api[build.workflow.output_node_id]["inputs"]
        self.assertFalse(combine["save_output"])
        self.assertFalse(nodes_of_type(build.workflow.api, "UpscaleModelLoader"))
        self.assertFalse(nodes_of_type(build.workflow.api, "ImageUpscaleWithModel"))
        decoder_id = next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "LTXVSpatioTemporalTiledVAEDecode"
        )
        self.assertEqual(combine["images"], [decoder_id, 0])
        decoder = nodes_of_type(
            build.workflow.api,
            "LTXVSpatioTemporalTiledVAEDecode",
        )[-1]["inputs"]
        self.assertEqual(decoder["latents"], [build.video_checkpoint_node_id, 0])
        self.assertEqual(
            (
                decoder["spatial_tiles"],
                decoder["spatial_overlap"],
                decoder["temporal_tile_length"],
                decoder["temporal_overlap"],
            ),
            (4, 1, 16, 1),
        )

    def test_initial_refinement_can_use_a_decoded_17_frame_diagnostic_guide(self):
        guide_path = Path(
            r"D:\LTX_Supervisor_Storage\acceptance\base\window.mkv"
        )
        build = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            self.plan,
            self.plan.chunks[0],
            revision=1,
            attempt_number=1,
            initial_guide_path=guide_path,
            initial_guide_skip_frames=96,
        )
        loader = nodes_of_type(build.workflow.api, "VHS_LoadVideoPath")[0]["inputs"]
        self.assertEqual(loader["video"], str(guide_path))
        self.assertEqual(loader["frame_load_cap"], 17)
        self.assertEqual(loader["skip_first_frames"], 96)
        guide_id = next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "LTXVAddGuide"
        )
        guide = build.workflow.api[guide_id]["inputs"]
        self.assertEqual(guide["frame_idx"], 0)
        self.assertEqual(guide["strength"], 1.0)
        self.assertFalse(
            nodes_of_type(build.workflow.api, "LTXVImgToVideoInplaceKJ"),
            "A decoded guide must replace, not follow, the initial single-frame guide.",
        )
        self.assertFalse(nodes_of_type(build.workflow.api, "LTXReferenceConditioning"))
        upscaled = next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "LTXVLatentUpsamplerTiled"
        )
        self.assertEqual(guide["latent"], [upscaled, 0])
        self.assertEqual(
            guide["image"][0],
            next(
                node_id
                for node_id, node in build.workflow.api.items()
                if node["class_type"] == "VHS_LoadVideoPath"
            ),
        )
        crop_id = next(
            node_id
            for node_id, node in build.workflow.api.items()
            if node["class_type"] == "LTXVCropGuides"
        )
        crop = build.workflow.api[crop_id]["inputs"]
        self.assertEqual(crop["positive"], [guide_id, 0])
        self.assertEqual(crop["negative"], [guide_id, 1])
        self.assertEqual(crop["latent"], [guide_id, 2])
        av = nodes_of_type(build.workflow.api, "LTXVConcatAVLatent")[0]["inputs"]
        sampler = nodes_of_type(build.workflow.api, "SamplerCustom")[0]["inputs"]
        self.assertEqual(av["video_latent"], [crop_id, 2])
        self.assertEqual(sampler["positive"], [crop_id, 0])
        self.assertEqual(sampler["negative"], [crop_id, 1])
        self.assertEqual(validate_api_graph(build.workflow.api), ())

    def test_later_refinement_reinjects_only_the_exact_extracted_handoff_frame(self):
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
        self.assertEqual(audio["frames_number"], 121)
        handoff = nodes_of_type(
            build.workflow.api,
            "10MinVideoMaker_LoadChunkLatent",
        )[0]["inputs"]
        self.assertEqual(handoff["expected_temporal_tokens"], 16)
        self.assertFalse(nodes_of_type(build.workflow.api, "VHS_LoadVideoPath"))
        self.assertFalse(nodes_of_type(build.workflow.api, "LTXVAddGuide"))
        self.assertFalse(nodes_of_type(build.workflow.api, "LTXVCropGuides"))
        image_loader = nodes_of_type(build.workflow.api, "VHS_LoadImagePath")[0]
        self.assertEqual(image_loader["inputs"]["image"], str(self.frame))
        self.assertEqual(len(nodes_of_type(build.workflow.api, "LTXReferenceConditioning")), 1)
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
        self.assertFalse(nodes_of_type(build.api, "UpscaleModelLoader"))
        self.assertFalse(nodes_of_type(build.api, "ImageUpscaleWithModel"))
        combine = build.api[build.output_node_id]["inputs"]
        decoder_id = next(
            node_id
            for node_id, node in build.api.items()
            if node["class_type"] == "LTXVSpatioTemporalTiledVAEDecode"
        )
        self.assertEqual(combine["images"], [decoder_id, 0])
        self.assertEqual(combine["format"], "video/ffv1-mkv")
        self.assertEqual(combine["pix_fmt"], "yuv444p")
        self.assertFalse(combine["save_output"])

    def test_short_final_window_is_fresh_and_has_no_causal_preroll(self):
        short_plan = build_scene_frame_plan(
            job_id=self.job.job_id,
            scene_id=self.scene.scene_id,
            revision=1,
            requested_duration_seconds=11.0,
            base_seed=self.scene.i2v.seed,
            fallback_prompt=self.scene.i2v.prompt,
            fallback_negative=self.scene.i2v.negative,
        )
        prior = Path(
            r"D:\LTX_Supervisor_Storage\jobs\job\chunks\chunk_0001\window.mkv"
        )
        stage1 = build_continuation_stage1_workflow(
            self.job,
            self.scene,
            self.frame,
            short_plan,
            short_plan.chunks[2],
            revision=1,
            attempt_number=1,
            previous_attempt_number=1,
        )
        self.assertFalse(nodes_of_type(stage1.api, "LTXVExtendSampler"))
        empty = nodes_of_type(stage1.api, "EmptyLTXVLatentVideo")[0]["inputs"]
        self.assertEqual(
            empty["length"],
            short_plan.chunks[2].new_transition_frames + 1,
        )
        build = build_continuation_stage2_workflow(
            self.job,
            self.scene,
            self.frame,
            short_plan,
            short_plan.chunks[2],
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
        self.assertEqual(audio["frames_number"], 25)
        self.assertFalse(nodes_of_type(build.workflow.api, "VHS_LoadVideoPath"))
        self.assertEqual(len(nodes_of_type(build.workflow.api, "VHS_LoadImagePath")), 1)

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
