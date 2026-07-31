from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_supervisor import _require_auto_continuation_approval
from tenminvideomaker.storage import StorageLayout


class _Renderer:
    @staticmethod
    def implementation_sha256() -> str:
        return "1" * 64

    @staticmethod
    def runtime_contract_sha256() -> str:
        return "2" * 64


class RunSupervisorTests(unittest.TestCase):
    def test_explicit_mode_does_not_require_auto_approval(self) -> None:
        supervisor = SimpleNamespace(
            settings=SimpleNamespace(continuation_mode="explicit"),
            continuation_renderer=_Renderer(),
        )
        with patch(
            "scripts.run_supervisor.require_auto_rollout_approval"
        ) as require:
            _require_auto_continuation_approval(
                supervisor,
                StorageLayout(Path(r"D:\test")),
            )
        require.assert_not_called()

    def test_auto_mode_binds_approval_to_current_runtime_identity(self) -> None:
        storage = StorageLayout(Path(r"D:\test"))
        supervisor = SimpleNamespace(
            settings=SimpleNamespace(continuation_mode="auto"),
            continuation_renderer=_Renderer(),
        )
        with patch(
            "scripts.run_supervisor.require_auto_rollout_approval"
        ) as require:
            _require_auto_continuation_approval(supervisor, storage)
        require.assert_called_once_with(
            storage,
            implementation_sha256="1" * 64,
            node_contracts_sha256="2" * 64,
        )


if __name__ == "__main__":
    unittest.main()
