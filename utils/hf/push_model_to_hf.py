"""Push a trained GLiNER2 adapter folder to the Hugging Face Hub.

python -m utils.hf.push_model_to_hf models/options_adapter_v1/best --repo-id user/options-recognizer-model --version v1
"""

import argparse
import os

from huggingface_hub import HfApi


def push_model_to_hub(folder_path, repo_id, version=None, private=True):
    """Upload an adapter folder and, if a version is given, tag the commit so
    consumers can pin a specific `revision`."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Adapter folder not found: {folder_path}")

    api = HfApi()
    print(f"Creating/verifying repo: https://huggingface.co/{repo_id}")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    commit_message = f"Upload adapter{f' ({version})' if version else ''}"
    print(f"Uploading {folder_path} to {repo_id}...")
    commit_info = api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    if version:
        print(f"Tagging commit as {version}...")
        api.create_tag(
            repo_id,
            tag=version,
            repo_type="model",
            revision=commit_info.oid,
            exist_ok=True,
        )

    print(f"Model published{' and tagged ' + version if version else ''}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder_path",
        help="Adapter folder, e.g. models/options_adapter_v1/best",
    )
    parser.add_argument("--repo-id", required=True, help="Target HF Hub model repo.")
    parser.add_argument(
        "--version", default=None, help="Tag to apply to the commit, e.g. v1."
    )
    parser.add_argument(
        "--public", action="store_true", help="Make the repo public (default: private)."
    )
    args = parser.parse_args()

    push_model_to_hub(
        args.folder_path, args.repo_id, version=args.version, private=not args.public
    )


if __name__ == "__main__":
    main()
