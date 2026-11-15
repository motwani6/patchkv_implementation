# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Package uncommitted files over HEAD without changing Git state."""

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def git(*options):
        return subprocess.check_output(["git", "-C", str(root), *options])

    if args.output.exists():
        parser.error("Output already exists; choose a new filename.")
    tracked = git("diff", "HEAD", "--name-only", "-z").decode().split("\0")
    untracked = (
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    paths = sorted(set(tracked + untracked) - {""})
    # Capture the path list before creating the archive.
    deleted = [p for p in paths if not (root / p).exists()]
    metadata = {
        "base_commit": git("rev-parse", "HEAD").decode().strip(),
        "branch": git("branch", "--show-current").decode().strip(),
        "files": paths,
        "deleted": deleted,
    }
    with tarfile.open(args.output, "w:gz") as archive:
        for path in paths:
            if path not in deleted:
                archive.add(root / path, arcname=path, recursive=False)
        data = json.dumps(metadata, indent=2).encode()
        info = tarfile.TarInfo("patchkv_overlay.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps(metadata, indent=2))
    print(f"Overlay: {args.output.resolve()}")


if __name__ == "__main__":
    main()
