from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tenminvideomaker.ownership import OwnershipError, SupervisorInstanceLock


class SupervisorInstanceLockTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows byte-lock regression")
    def test_locked_sentinel_read_denial_becomes_ownership_error(self) -> None:
        handle = Mock()
        handle.read.side_effect = PermissionError(13, "Permission denied")
        handle.tell.return_value = 1

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(Path, "open", return_value=handle),
            patch("msvcrt.locking", side_effect=PermissionError(13, "locked")),
            self.assertRaisesRegex(
                OwnershipError,
                "Another 10MinVideoMaker controller is already running",
            ),
        ):
            SupervisorInstanceLock(Path(directory) / "supervisor.lock").acquire()

        handle.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
