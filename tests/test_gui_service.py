from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.gui_service import SupervisorController
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState
from tenminvideomaker.storage import StorageLayout

from test_contracts import payload


class GuiServiceTests(unittest.TestCase):
    def test_interrupt_cancels_only_project_prompts_and_preserves_job_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.begin_scene_stage(
                job.job_id,
                1,
                PipelineState.RUNNING_I2V,
                prompt_id="project-prompt",
            )
            supervisor = Mock()
            supervisor.store = store
            supervisor.comfy.cancel_project_prompts.return_value = ("project-prompt",)
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )

            cancelled = controller.interrupt_current_job()

            self.assertEqual(cancelled, ("project-prompt",))
            supervisor.comfy.cancel_project_prompts.assert_called_once_with()
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            self.assertEqual(store.scene_records(job.job_id)[0].state, SceneState.CANCELLED)
            self.assertEqual(store.load_job(job.job_id).job_id, job.job_id)


if __name__ == "__main__":
    unittest.main()
