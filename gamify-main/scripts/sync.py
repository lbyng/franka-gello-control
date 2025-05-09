"""
Sample run:
python scripts/sync.py \
    --path data/test/ \
    --from_remote \
    --local_prefix /scr/suvir/cil \
    --remote_prefix suvir@scdt.stanford.edu:/iliad2/u/suvir/cil

--path may be a directory or as single file

Optionally, set REPO_PATH and REMOTE_REPO_PATH in setup_shell.sh so that
local_prefix and remote_prefix get populated by default
"""

import argparse
import os
import subprocess


def run_sync(
    path,
    from_remote,
    to_remote,
    to_scratch,
    local_prefix,
    remote_prefix,
    scratch_prefix,
    dry_run,
    ignore_existing,
    return_output=False,
    copy_links=False,
):
    assert from_remote ^ to_remote ^ to_scratch, "Must choose one of --to_remote or --from_remote or --to_scratch"

    if path.endswith("/"):
        path = path[:-1]

    if to_remote or to_scratch:
        remote_prefix = remote_prefix if to_remote else scratch_prefix
        src = os.path.join(local_prefix, path)
        dest = os.path.join(remote_prefix, os.path.dirname(path))
        dest_without_server = dest.split(":")[-1]
        if to_scratch:
            # rsync-path doesn't work on local machines
            mkdir_call = ["mkdir", "-p", f"{dest_without_server}"]
            subprocess.run(mkdir_call)

        call = [
            "rsync",
            "-rP",
            "--stats",
            "--rsync-path",
            f"mkdir -p {dest_without_server} && rsync",
            f"{src}",
            f"{dest}",
        ]

    elif from_remote:
        src = os.path.join(remote_prefix, path)
        dest = os.path.join(local_prefix, os.path.dirname(path))
        os.makedirs(dest, exist_ok=True)
        call = ["rsync", "-rP", "--stats", f"{src}", f"{dest}"]

    if dry_run:
        call.append("--dry-run")
    if ignore_existing:
        call.append("--ignore-existing")
    if copy_links:
        call.append("--copy-links")

    print("[research] Source:", src)
    print("[research] Destination:", dest)

    print("[research] Executing rsync command")
    print(" ".join(call))

    if return_output:
        return subprocess.check_output(call, universal_newlines=True)
    else:
        subprocess.run(call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--from_remote", default=False, action="store_true")
    parser.add_argument("--to_remote", default=False, action="store_true")
    parser.add_argument("--to_scratch", default=False, action="store_true")
    parser.add_argument("--local_prefix", default=os.environ["RESEARCH_LIGHTNING_REPO_PATH"])
    parser.add_argument("--remote_prefix", default=os.environ["RESEARCH_LIGHTNING_REMOTE_REPO_PATH"], type=str)
    parser.add_argument("--scratch_prefix", default=os.environ["RESEARCH_LIGHTNING_SCRATCH_PATH"], type=str)
    parser.add_argument("--dry_run", default=False, action="store_true")
    parser.add_argument("--ignore_existing", default=False, action="store_true")
    parser.add_argument("--copy_links", default=False, action="store_true")

    args = parser.parse_args()

    run_sync(
        path=args.path,
        from_remote=args.from_remote,
        to_remote=args.to_remote,
        to_scratch=args.to_scratch,
        local_prefix=args.local_prefix,
        remote_prefix=args.remote_prefix,
        scratch_prefix=args.scratch_prefix,
        dry_run=args.dry_run,
        ignore_existing=args.ignore_existing,
        copy_links=args.copy_links,
    )
