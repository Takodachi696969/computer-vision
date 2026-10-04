"""Opt-in Hugging Face artifact transfer; importing this never authenticates."""

from __future__ import annotations

from pathlib import Path


def download_run(repo_id: str, output_dir: str | Path, revision: str | None = None) -> str:
    """Download a trusted lab-format model repo; review metadata before loading.

    A generic Hugging Face vision/action model is not automatically compatible
    with this six-action MuJoCo task. SB3 model loading uses Python serialization.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=["model.zip", "scene.json", "metadata.json", "evaluation.json", "README.md"],
    )


def upload_run(repo_id: str, run_dir: str | Path, *, private: bool = True) -> str:
    """Explicitly publish a complete run to a user-selected Hub repository.

    Requires the optional huggingface_hub package and prior `hf auth login`.
    The caller/user chooses this action; training never uploads automatically.
    """
    from huggingface_hub import HfApi

    run_dir = Path(run_dir)
    for filename in ("model.zip", "scene.json", "metadata.json", "evaluation.json"):
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(f"Missing run artifact: {run_dir / filename}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    return str(
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(run_dir),
            allow_patterns=[
                "model.zip",
                "scene.json",
                "metadata.json",
                "evaluation.json",
                "README.md",
            ],
            commit_message="Upload HumanED PAROL6 simulation reaching policy",
        )
    )
