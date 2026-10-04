# Third-party notices and source provenance

The new robot lab code is distributed under GPL-3.0-only; see the root [LICENSE](LICENSE). Third-party components retain their own licenses. Package dependencies are installed from the sources recorded in `uv.lock`; they are not relabeled as original HumanED work.

## PAROL6 robot assets

Source: [PCrnjak/PAROL6-python-API](https://github.com/PCrnjak/PAROL6-python-API), exact revision `b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4`.

Files copied unchanged into `src/humaned_lab/simulation/assets/parol6/`:

- `parol6/urdf_model/urdf/PAROL6.urdf` → `PAROL6.urdf`.
- Seven original visual STLs: `base_link.STL`, `L1.STL` through `L6.STL` → `meshes/`.
- The upstream GPL version 3 license → `LICENSE` in that asset directory.

The source URDF contains the original SolidWorks URDF exporter attribution, which is preserved. Copied asset hashes are recorded in [MANIFEST.sha256](src/humaned_lab/simulation/assets/parol6/MANIFEST.sha256). Detailed source/physics modifications are described in [PROVENANCE.md](src/humaned_lab/simulation/assets/parol6/PROVENANCE.md). `simulation/model.py` generates the modified MuJoCo model with additional actuators/contact geometry. Keep the source, GPL license and notices together when redistributing these assets.

The optional upstream PAROL6 controller is installed separately from a pinned source checkout. Its source/license and dependency notices remain in that checkout/environment; this package does not copy its entire implementation into the core source tree.

## RealSense SDK

Source: [realsenseai/librealsense](https://github.com/realsenseai/librealsense), audited revision `e15c5d6bb1563e778d116f682aeefffbae2daedc`, SDK source version 2.58.4. The source repository's license is Apache-2.0. The optional `pyrealsense2` package supplies SDK binaries; the SDK source or viewer is not bundled in this branch. Preserve the SDK distribution's own notices when redistributing it.

## Runtime and learning libraries

MuJoCo, Gymnasium, NumPy, FastAPI, Uvicorn, HTTPX, Pillow, Stable-Baselines3, PyTorch and Hugging Face Hub are independent upstream projects. Their exact installed versions/sources are in `uv.lock` and runtime reports. Their installed distributions carry their respective license/notice files; review those if creating a binary redistributable rather than installing dependencies through the provided bootstrap.

## Development skill attribution

The K-Dense `stable-baselines3` and official Hugging Face `hf-cli` skills informed the workflow. They are not copied into runtime source. Source revisions, their application and the requested scientific-skills citation are recorded in [docs/skills.md](docs/skills.md).
