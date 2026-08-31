"""Publish only the backed-up Space entry point and batch extension."""

import argparse
from pathlib import Path


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, set_client_factory
    import httpx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="hkchengrex/OmniVoice")
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    # Retry connection establishment only, not ambiguous commit POST responses.
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(retries=3), timeout=60, follow_redirects=True
    ))
    api = HfApi()
    api.whoami()  # Require an existing local login; never embed credentials.
    current = api.space_info(args.space).sha
    if current != args.expected_revision:
        raise SystemExit("Space changed since inspection; review it before deploying.")
    directory = Path(__file__).resolve().parent
    commit = api.create_commit(
        repo_id=args.space, repo_type="space", parent_commit=current,
        commit_message="Add bounded voice-cloning batch API with timing metrics",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=directory / name)
            for name in ("app.py", "batch_api.py")
        ],
    )
    print(commit.oid)
    print(commit.commit_url)


if __name__ == "__main__":
    main()
