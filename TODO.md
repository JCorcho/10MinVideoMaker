# 10MinVideoMaker follow-up work

- [ ] Strengthen I2V crash recovery around temporary `VHS_VideoCombine` output. Recover a completed prompt's
  history output when possible, or persist directly to project-owned durable staging, so a crash after rendering
  but before the supervisor copies the clip to its versioned directory under
  `D:\LTX_Supervisor_Storage\jobs` does not require rerendering that scene.
  Preserve the current deterministic clip paths and completed-scene resume behavior.
