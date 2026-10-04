# Working on the HumanED robot lab

Read README.md, docs/architecture.md, and the module you change. Keep the original week-1.ipynb exercise intact.

- The src/humaned_lab package owns physics/control/policies/training/hardware. Examples stay small and runnable; editable worlds belong in configs/scenes.
- Public simulation units are metres, radians, seconds, kilograms, and wxyz quaternions. Upstream PAROL6 uses degrees/mm; conversion belongs only in hardware adapters/bridge.
- Policies output six normalized joint deltas: 0.04 rad per action, 0.05 s per control. Change observations/actions/rewards with a new task version and retrain checkpoints.
- MuJoCo's OpenGL renderer belongs to one thread. The service owns a shared world; lock atomic state changes.
- Simulation HTTP never forwards policy actions to physical motors. Physical control is an explicit opt-in adapter with operator commissioning; never silently home hardware.
- Preserve license/provenance/checksums for model assets. Clearly identify approximated dynamics and contact; no fake grasping claims.
- Keep dependencies in pyproject.toml and update uv.lock. Bootstrap uses --frozen; the default PyTorch index is CPU. Optional upstream dependencies stay in .upstream-venv.
- Use .venv/Scripts/python.exe on Windows or .venv/bin/python on Linux. Run python -m pytest -q, ruff check ., and meaningful physics/control verification for changed behavior. Inspect rendered output when changing visuals/model geometry.
- Checkpoints require metadata, resolved scene, and held-out evaluation. Do not infer policy quality from training reward or a short smoke run.
- Do not commit outputs/, local environments, device recordings, credentials, invitation links, or logs. Commit portable examples and documented small reference checkpoints only.
