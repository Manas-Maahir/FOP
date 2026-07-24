#!/usr/bin/env python
"""Download and extract the raw TBX11K archive.

The official repo (github.com/yun-liu/Tuberculosis) publishes exactly two sources -- a Google Drive
file and Baidu Yunpan -- and documents **no folder layout**, which is why
``tools/prepare_tbx11k.py`` discovers the structure instead of assuming it.

The realistic failure here is not a bug, it is Google: Drive refuses to virus-scan large public
files and serves an HTML interstitial instead of the archive. gdown usually handles the token dance,
but a heavily-downloaded public file also hits a quota and simply stops working for a while. So this
script verifies what it got rather than trusting the exit code, and when Drive says no it prints the
manual route instead of failing the notebook run.

    python scripts/download_tbx11k.py --dst data/raw
    python scripts/download_tbx11k.py --dst data/raw --archive ~/Downloads/TBX11K.zip   # local file
    python scripts/download_tbx11k.py --dst data/raw --check                            # verify only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# From the "Dataset on Google Drive" link in the official README.
TBX11K_GDRIVE_ID = "1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR"
DRIVE_VIEW = f"https://drive.google.com/file/d/{TBX11K_GDRIVE_ID}/view"
BAIDU = "https://pan.baidu.com/s/1INhqaZyPFKWPFXgynerXew"

# An archive smaller than this is certainly not the dataset (the real one is tens of GB).
MIN_ARCHIVE_BYTES = 1_000_000_000
# Rough headroom needed: the zip plus its extracted contents before the zip is deleted.
NEEDED_GB = 70


def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    say("\n" + "=" * 72)
    say(title)
    say("=" * 72)


def looks_extracted(root: Path) -> tuple[bool, dict]:
    """Is there a usable TBX11K tree under `root`? Counts rather than trusting directory names."""
    if not root.is_dir():
        return False, {}
    n_xml = n_img = 0
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if low.endswith(".xml"):
                n_xml += 1
            elif low.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                n_img += 1
        if n_img > 12000 and n_xml > 1200:
            break
    info = {"n_xml": n_xml, "n_img": n_img}
    # The real set is ~11,200 images and ~1,200 XMLs; accept a generous band so a partial-but-usable
    # copy is not rejected outright, and let prepare_tbx11k.py --inspect be the real gate.
    return (n_xml >= 500 and n_img >= 5000), info


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def is_html(path: Path) -> bool:
    """Drive's quota / virus-scan interstitial is HTML wearing a .zip extension."""
    try:
        with open(path, "rb") as f:
            head = f.read(512).lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html")
    except OSError:
        return False


def manual_instructions() -> None:
    say("")
    say("-" * 72)
    say("MANUAL DOWNLOAD REQUIRED")
    say("-" * 72)
    say("Google Drive throttles large public files, so an automated fetch often cannot")
    say("complete. Any one of these works:")
    say("")
    say(f"  1. Open {DRIVE_VIEW}")
    say("     click 'Add shortcut to Drive' / make a copy into YOUR OWN Drive, then re-run")
    say("     this script with your copy's id:")
    say("         python scripts/download_tbx11k.py --dst data/raw --gdrive-id <YOUR_ID>")
    say("     (personal copies are not throttled the way the shared public file is)")
    say("")
    say("  2. Download the archive in a browser, then point this script at the file:")
    say("         python scripts/download_tbx11k.py --dst data/raw --archive path/to/TBX11K.zip")
    say("")
    say(f"  3. Baidu Yunpan mirror: {BAIDU}")
    say("")
    say("Once it is extracted, verify the layout before preparing:")
    say("     python tools/prepare_tbx11k.py --inspect --src data/raw/TBX11K")
    say("-" * 72)


def extract(archive: Path, dst: Path, delete_after: bool) -> bool:
    say(f"extracting {archive.name} ({archive.stat().st_size / 1e9:.1f} GB) -> {dst}")
    say("(this takes a while and is disk-bound; no progress bar from zipfile)")
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            total = len(members)
            for i, m in enumerate(members, 1):
                zf.extract(m, dst)
                if i % 2000 == 0:
                    say(f"   {i}/{total} entries ...")
    except zipfile.BadZipFile:
        say(f"ERROR: {archive} is not a valid zip archive.")
        if is_html(archive):
            say("       It is an HTML page -- Google Drive served its quota/scan interstitial.")
        return False

    if delete_after:
        try:
            archive.unlink()
            say(f"removed {archive.name} to reclaim disk")
        except OSError as e:
            say(f"[warn] could not delete {archive}: {e}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dst", default="data/raw", help="where to extract the dataset")
    ap.add_argument("--archive", default=None,
                    help="use an already-downloaded zip instead of fetching")
    ap.add_argument("--gdrive-id", default=TBX11K_GDRIVE_ID,
                    help="Google Drive file id (use your own copy's id if the public one throttles)")
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument("--keep-archive", action="store_true", help="do not delete the zip after extract")
    args = ap.parse_args(argv)

    dst = Path(args.dst)
    rule("TBX11K raw dataset")
    say(f"  target : {dst.resolve()}")

    ok, info = looks_extracted(dst)
    if ok:
        say(f"  status : already extracted ({info['n_img']} images, {info['n_xml']} annotations)")
        say("\nNothing to do. Next:")
        say(f"     python tools/prepare_tbx11k.py --inspect --src {dst}")
        return 0
    if info:
        say(f"  status : incomplete ({info['n_img']} images, {info['n_xml']} annotations)")
    else:
        say("  status : not present")

    if args.check:
        return 1

    disk = free_gb(dst)
    say(f"  disk   : {disk:.0f} GB free (need ~{NEEDED_GB} GB for archive + extracted copy)")
    if disk < NEEDED_GB:
        say(f"\nERROR: not enough free space. Free up ~{NEEDED_GB - disk:.0f} GB, or pass a --dst")
        say("       on a bigger drive.")
        return 2

    # -- a local archive short-circuits the whole Drive problem ---------------------------
    if args.archive:
        archive = Path(args.archive)
        if not archive.is_file():
            say(f"\nERROR: --archive not found: {archive}")
            return 2
        if not extract(archive, dst, delete_after=False):
            return 2
    else:
        archive = dst.parent / "tbx11k.zip"
        rule("downloading (tens of GB -- this is slow)")
        try:
            import gdown  # noqa: F401
        except ImportError:
            say("gdown is not installed in this interpreter.")
            say("Run this script with the project venv:  .venv\\Scripts\\python.exe "
                "scripts/download_tbx11k.py")
            manual_instructions()
            return 2

        # Pass the BARE id positionally: every gdown version accepts `url_or_id`, whereas --fuzzy
        # (needed only to parse a full /view URL) does not exist in older builds.
        cmd = [sys.executable, "-m", "gdown", args.gdrive_id, "-O", str(archive)]
        say("$ " + " ".join(cmd))
        subprocess.run(cmd, check=False)

        if not archive.is_file():
            say("\n!! gdown produced no file.")
            manual_instructions()
            return 2
        size = archive.stat().st_size
        if size < MIN_ARCHIVE_BYTES or is_html(archive):
            say(f"\n!! downloaded file is only {size / 1e6:.1f} MB -- too small to be the dataset.")
            if is_html(archive):
                say("   It is HTML: Google Drive served its quota / virus-scan page.")
            archive.unlink(missing_ok=True)
            manual_instructions()
            return 2
        if not extract(archive, dst, delete_after=not args.keep_archive):
            manual_instructions()
            return 2

    ok, info = looks_extracted(dst)
    rule("RESULT")
    say(f"  images      : {info.get('n_img', 0)}   (expect ~11,200)")
    say(f"  annotations : {info.get('n_xml', 0)}   (expect ~1,200)")
    if not ok:
        say("\n!! The extracted tree does not look like TBX11K. Inspect it manually:")
        say(f"     python tools/prepare_tbx11k.py --inspect --src {dst}")
        return 2
    say("\nNext:")
    say(f"     python tools/prepare_tbx11k.py --inspect --src {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
