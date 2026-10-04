# Skills used and primary references

The repository analysis and learning workflow were prepared on 4 October 2026 using the following skills. Their instructions informed development; they are not runtime dependencies of the robot lab.

| Skill | Source and inspected revision | How it was applied |
|---|---|---|
| Local repository structure/analysis workflow | `C:/Users/Peter/.agents/skills/source-command-analysis-analyze-repo-for-claude/SKILL.md` | Inspect source/manifests/history, check existing summaries, map modules and audit mismatches; persist summaries under the user's project-summary directory. |
| `stable-baselines3` | [K-Dense Scientific Skills](https://github.com/K-Dense-AI/claude-scientific-skills), revision `154988403bb5a18e9d3c0ce4e6d5e2e4b184a298` | Gymnasium contract, `check_env`, normalized actions, Monitor, seeds, checkpointing and held-out evaluation. |
| `hf-cli` | [Official Hugging Face skills](https://github.com/huggingface/skills), revision `ca0325bb20b2d0a1b2efa893670c4c72f79e707b` | Use the current `hf` CLI, inspect model metadata and pin/download a matching checkpoint; separate Hub artifact access from robot compatibility. |

The two public source repositories had 47,541 and 11,136 GitHub stars respectively when inspected. Those are repository-level popularity signals, not evaluations of an individual skill or guarantees of correctness. The public skills were installed in `C:/Users/Peter/.codex/skills/`; the relevant instructions were read and checked against local code and official library documentation.

DeepWiki discovery was attempted as required by the local analysis workflow. PAROL6's page was unavailable; librealsense's page indexed an earlier revision. The actual repository mapping and audit therefore use the pinned local source and primary documentation, not a wiki summary treated as authority.

## Primary documentation

- [Pinned PAROL6 Python API source](https://github.com/PCrnjak/PAROL6-python-API/tree/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4): exact method names, protocol, planner, transport and units.
- [Pinned RealSense SDK source](https://github.com/realsenseai/librealsense/tree/e15c5d6bb1563e778d116f682aeefffbae2daedc): Python binding, camera pipeline, acquisition and source version.
- [MuJoCo model XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html): inertials, contact primitives, actuators and integrator parameters.
- [Stable-Baselines3 2.9.0 custom environment guide](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/custom_env.html) and [PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html): training interfaces and algorithm behavior.
- [Official Hub CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli): model inspection/download.
- [LeRobot custom hardware integration](https://huggingface.co/docs/lerobot/main/en/integrate_hardware), [dataset format](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3) and [SmolVLA guide](https://huggingface.co/docs/lerobot/main/en/smolvla): the additional adapter/data work required for a VLA experiment.

## Citation

Kassis, Timothy; Agarwal, Vinayak; He, Yuhuan; Patel, Darshil; Brueckner, Aubrey M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. [DOI: 10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).

The current version inspected for this work was v2, revised 2 September 2026.

This citation credits the scientific-skills procedural contribution to the released training/environment workflow; it does not claim the paper evaluates this robot or this package.
