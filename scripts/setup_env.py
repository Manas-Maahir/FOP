#!/usr/bin/env python
"""Bootstrap the pinned training environment. Runs under ANY Python >= 3.8, stdlib only.

This is the script `notebooks/local_runbook.ipynb` cell 4 calls. It is deliberately importable by a
kernel we know nothing about: no third-party imports, no assumptions about the host Python version,
and no need for the caller to be (or become) the interpreter we build.

What it builds
--------------
A single `.venv` on **Python 3.11** holding *both* detection stacks:

    torch 2.1.0+cu121 / torchvision 0.16.0+cu121   <- newest torch with an mmcv Windows wheel
    mmengine 0.10.7 / mmcv 2.1.0 / mmdet 3.3.0     <- optional; see STAGES below
    numpy<2, pycocotools 2.0.7, ...

Why 3.11 and why that torch: OpenMMLab publishes prebuilt **win_amd64** mmcv wheels (so no MSVC is
needed), but only for cp38-cp311, and mmcv 2.1.0 -- the last version mmdet 3.3.0 accepts, since
`mmdet/__init__.py` asserts mmcv < 2.2.0 -- is built against torch 2.1.0. Pinning *both* stacks to
one torch also removes a version confound from the torchvision-vs-mmdet comparison.

The host PC does not need Python 3.11 installed: `uv python install 3.11` fetches a standalone
CPython. That is the whole reason `uv` is used rather than `venv`.

Staged installs
---------------
Stages run in dependency order and each is verified before the next starts. mmdet is **last and
optional**: if OpenMMLab's index is unreachable or the wheel does not match, the torchvision stack is
already complete and usable, so we record `have_mmdet: false` and carry on. Nothing in here may
hard-fail a Run All.

Idempotency
-----------
`.venv/.symformer_stamp.json` records the pins and the verification result. A re-run with a matching
stamp exits immediately, so re-running the notebook top-to-bottom costs seconds, not a re-download.

    python scripts/setup_env.py                 # build (or skip if the stamp matches)
    python scripts/setup_env.py --check         # verify only, never install
    python scripts/setup_env.py --force         # ignore the stamp and reinstall
    python scripts/setup_env.py --no-mmdet      # torchvision stack only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
STAMP = VENV / ".symformer_stamp.json"
LOCK = ROOT / "requirements.lock.txt"

PY_VERSION = "3.11"

# The exact pin set. Any change here invalidates the stamp and forces a rebuild.
PINS = {
    "python": PY_VERSION,
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "cuda": "cu121",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.3.0",
}

TORCH_INDEX = "https://download.pytorch.org/whl/{cuda}"
MMCV_INDEX = "https://download.openmmlab.com/mmcv/dist/{cuda}/torch{torch}/index.html"

# numpy<2 because mmcv 2.1.0 is compiled against the numpy 1.x ABI; pycocotools 2.0.7 is the last
# release built the same way. Everything else is a pure wheel and floats.
CORE_DEPS = [
    "numpy<2",
    "pycocotools==2.0.7",
    "pillow>=9",
    "tqdm>=4.65",
    "pyyaml>=6",
    "matplotlib>=3.6",
    "pandas>=2",
    "opencv-python>=4.6",
    "colorama>=0.4",       # ANSI on legacy Windows consoles (conhost, not Windows Terminal)
    "gdown>=4.7",
    "pytest>=7",
    "scipy>=1.10",         # sklearn-free AUC/ROC helpers for the stage-2 metrics
]


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, streaming its output. Raises CalledProcessError on failure."""
    say("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def try_run(cmd: list[str], **kw) -> bool:
    try:
        run(cmd, **kw)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        say(f"[failed] {e}")
        return False


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ------------------------------------------------------------------------------------------
# uv
# ------------------------------------------------------------------------------------------
def ensure_uv() -> str:
    """Return a path to a `uv` executable, installing it into the *host* Python if needed.

    Three lookups, cheapest first. The `uv.find_uv_bin()` route is the one the uv PyPI package
    documents, and it works even when the host's Scripts/ dir is not on PATH -- which is the norm
    for a Jupyter kernel launched from a Python that was never "activated".
    """
    exe = shutil.which("uv")
    if exe:
        say(f"uv found on PATH: {exe}")
        return exe

    try:
        from uv import find_uv_bin  # type: ignore

        exe = str(find_uv_bin())
        if Path(exe).exists():
            say(f"uv found via the uv package: {exe}")
            return exe
    except Exception:
        pass

    say("uv not found -- installing it into the current Python ...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", "uv"])

    try:
        from uv import find_uv_bin  # type: ignore

        exe = str(find_uv_bin())
        if Path(exe).exists():
            say(f"uv installed: {exe}")
            return exe
    except Exception:
        pass

    exe = shutil.which("uv")
    if exe:
        return exe

    # last resort: the scripts dir of the interpreter we are running under
    cand = Path(sysconfig.get_path("scripts")) / ("uv.exe" if os.name == "nt" else "uv")
    if cand.exists():
        say(f"uv installed: {cand}")
        return str(cand)

    raise RuntimeError(
        "Could not install or locate `uv`.\n"
        "Fix: install it manually from https://docs.astral.sh/uv/getting-started/installation/\n"
        "     (or `pip install uv`) and re-run this cell."
    )


# ------------------------------------------------------------------------------------------
# GPU detection
# ------------------------------------------------------------------------------------------
def detect_gpu() -> tuple[bool, str]:
    """(has_nvidia, description). Drives the cu121-vs-cpu wheel choice."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        name = (out.stdout or "").strip().splitlines()
        if out.returncode == 0 and name:
            return True, name[0].strip()
        return False, f"nvidia-smi returned {out.returncode}"
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"nvidia-smi failed: {e}"


# ------------------------------------------------------------------------------------------
# stamp
# ------------------------------------------------------------------------------------------
def read_stamp() -> dict | None:
    try:
        return json.loads(STAMP.read_text())
    except Exception:
        return None


def stamp_is_current(stamp: dict | None, cuda: str) -> bool:
    if not stamp or not venv_python().exists():
        return False
    want = dict(PINS, cuda=cuda)
    return stamp.get("pins") == want and stamp.get("torch_ok") is True


def write_stamp(cuda: str, torch_ok: bool, have_mmdet: bool, gpu: str, versions: dict) -> None:
    STAMP.write_text(json.dumps({
        "pins": dict(PINS, cuda=cuda),
        "torch_ok": torch_ok,
        "have_mmdet": have_mmdet,
        "gpu": gpu,
        "versions": versions,
    }, indent=2))


# ------------------------------------------------------------------------------------------
# verification (runs *inside* the venv)
# ------------------------------------------------------------------------------------------
PROBE = r"""
import json, sys
info = {"python": sys.version.split()[0]}
try:
    import torch, torchvision
    info["torch"] = torch.__version__
    info["torchvision"] = torchvision.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    info["torch_ok"] = True
except Exception as e:
    info["torch_ok"] = False
    info["torch_error"] = repr(e)
for mod in ("numpy", "pycocotools", "mmengine", "mmcv", "mmdet"):
    try:
        m = __import__(mod)
        info[mod] = getattr(m, "__version__", "?")
    except Exception:
        info[mod] = None
# The CUDA deformable-attention op is the reason mmcv is worth having: it retires the grid_sample
# approximation the SAS block falls back to. Report whether it actually imports.
try:
    from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
    info["msda_op"] = True
except Exception as e:
    info["msda_op"] = False
    info["msda_error"] = repr(e)[:200]
info["have_mmdet"] = bool(info.get("mmdet"))
print("SYMFORMER_PROBE " + json.dumps(info))
"""


def probe() -> dict:
    """Ask the venv what it actually has. Never raises."""
    py = venv_python()
    if not py.exists():
        return {"torch_ok": False, "error": "venv missing"}
    try:
        out = subprocess.run([str(py), "-c", PROBE], capture_output=True, text=True, timeout=300)
        for line in (out.stdout or "").splitlines():
            if line.startswith("SYMFORMER_PROBE "):
                return json.loads(line[len("SYMFORMER_PROBE "):])
        return {"torch_ok": False, "error": (out.stderr or out.stdout or "")[-2000:]}
    except Exception as e:
        return {"torch_ok": False, "error": repr(e)}


def report(info: dict) -> None:
    rule("ENVIRONMENT")
    say(f"  venv python      : {info.get('python', '?')}   ({venv_python()})")
    say(f"  torch            : {info.get('torch')}")
    say(f"  torchvision      : {info.get('torchvision')}")
    say(f"  cuda available   : {info.get('cuda_available')}")
    say(f"  device           : {info.get('device')}")
    say(f"  numpy            : {info.get('numpy')}")
    say(f"  pycocotools      : {info.get('pycocotools')}")
    say(f"  mmengine         : {info.get('mmengine')}")
    say(f"  mmcv             : {info.get('mmcv')}")
    say(f"  mmdet            : {info.get('mmdet')}")
    say(f"  mmcv CUDA MSDA op: {info.get('msda_op')}")
    if info.get("msda_error"):
        say(f"      -> {info['msda_error']}")


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="ignore the stamp and reinstall")
    ap.add_argument("--check", action="store_true", help="verify only; never install")
    ap.add_argument("--no-mmdet", action="store_true", help="skip the mmdetection stack")
    ap.add_argument("--cpu", action="store_true", help="force CPU wheels even if a GPU is present")
    args = ap.parse_args(argv)

    has_gpu, gpu_desc = detect_gpu()
    if args.cpu:
        has_gpu = False
        gpu_desc = "forced CPU (--cpu)"
    cuda = PINS["cuda"] if has_gpu else "cpu"

    rule("SymFormer local environment")
    say(f"  project : {ROOT}")
    say(f"  venv    : {VENV}")
    say(f"  gpu     : {gpu_desc}")
    say(f"  wheels  : {cuda}")
    if not has_gpu:
        say("\n  !! No NVIDIA GPU detected -- installing CPU wheels.")
        say("     The smoke test will still pass, but real training is impractical on CPU.")

    if args.check:
        info = probe()
        report(info)
        return 0 if info.get("torch_ok") else 1

    stamp = read_stamp()
    if stamp_is_current(stamp, cuda) and not args.force:
        say("\nStamp matches the pins and the venv is present -- nothing to do.")
        say("(Use --force to rebuild.)")
        info = probe()
        report(info)
        return 0 if info.get("torch_ok") else 1

    uv = ensure_uv()

    # ---- stage 0: interpreter + venv ----------------------------------------------------
    rule(f"STAGE 0/4  Python {PY_VERSION} + venv")
    # Downloads a standalone CPython if the host has no 3.11. This is why `uv` is required:
    # the mmcv wheel ceiling (cp38-cp311) must not become a requirement on the user's PC.
    try_run([uv, "python", "install", PY_VERSION])
    if VENV.exists() and args.force:
        say(f"removing {VENV} (--force)")
        shutil.rmtree(VENV, ignore_errors=True)
    run([uv, "venv", str(VENV), "--python", PY_VERSION])
    py = venv_python()
    if not py.exists():
        say(f"ERROR: venv python missing at {py}")
        return 1

    pip = [uv, "pip", "install", "--python", str(py)]

    # ---- stage 0.5: seed the venv --------------------------------------------------------
    # `uv venv` creates a bare environment with no setuptools, but torch 2.1.0 imports
    # `pkg_resources` at `import torch` -- so without this, torch installs fine and then fails to
    # import with "No module named 'pkg_resources'". setuptools<81 still ships pkg_resources.
    #
    # numpy<2 is pinned HERE, before torch, on purpose: torch 2.1.0 only requires `numpy` with no
    # upper bound, so left to itself pip pulls numpy 2.x, whose ABI torch 2.1.0 was not built
    # against (it aborts at import with "_ARRAY_API not found"). Installing numpy 1.x first means
    # torch's requirement is already satisfied and it is never upgraded.
    rule("STAGE 0.5  seed the venv (setuptools + numpy<2)")
    if not try_run(pip + ["setuptools<81", "wheel", "numpy<2"]):
        say("\nERROR: could not seed the venv with setuptools/numpy.")
        return 1

    # ---- stage 1: torch ------------------------------------------------------------------
    rule(f"STAGE 1/4  torch {PINS['torch']} + torchvision {PINS['torchvision']} ({cuda})")
    say("  ~2.5 GB download on a cold machine -- this is the slow part.")
    index = TORCH_INDEX.format(cuda=cuda)
    # Keep numpy<2 in this command too: --index-url makes the torch index the ONLY source for the
    # command, and pinning numpy here stops a stray resolution from reaching for 2.x off it.
    ok = try_run(pip + [
        f"torch=={PINS['torch']}", f"torchvision=={PINS['torchvision']}", "numpy<2",
        "--index-url", index,
    ])
    if not ok:
        say("\nERROR: torch install failed. Most likely causes: no network, or a proxy blocking")
        say(f"       {index}")
        return 1

    info = probe()
    if not info.get("torch_ok"):
        say(f"\nERROR: torch imported but is broken: {info.get('torch_error') or info.get('error')}")
        return 1
    say(f"  ok: torch {info['torch']}, cuda_available={info['cuda_available']}, "
        f"device={info['device']}")
    if has_gpu and not info.get("cuda_available"):
        say("  !! A GPU was detected but torch cannot see CUDA. Training will fall back to CPU.")
        say("     Usually an outdated NVIDIA driver -- cu121 wheels need driver >= 527.")

    # ---- stage 2: core deps --------------------------------------------------------------
    rule("STAGE 2/4  core dependencies")
    if not try_run(pip + CORE_DEPS):
        say("\nERROR: core dependency install failed.")
        return 1

    # ---- stage 3: mmdetection (OPTIONAL) -------------------------------------------------
    have_mmdet = False
    if args.no_mmdet:
        rule("STAGE 3/4  mmdetection -- SKIPPED (--no-mmdet)")
    else:
        rule(f"STAGE 3/4  mmengine / mmcv {PINS['mmcv']} / mmdet {PINS['mmdet']}  (optional)")
        say("  Prebuilt win_amd64 wheels -- no compiler needed.")
        say("  If this stage fails, the torchvision stack above is already complete and usable;")
        say("  only `--stack mmdet` becomes unavailable.")
        mmcv_index = MMCV_INDEX.format(cuda=cuda, torch=PINS["torch"])
        ok = try_run(pip + [f"mmengine=={PINS['mmengine']}"])
        if ok:
            ok = try_run(pip + [f"mmcv=={PINS['mmcv']}", "-f", mmcv_index])
        if ok:
            # mmdet pulls a numpy>=2 compatible pycocotools on some resolutions; pin them back
            # afterwards so mmcv's numpy-1.x ABI stays satisfied.
            ok = try_run(pip + [f"mmdet=={PINS['mmdet']}"])
            if ok:
                try_run(pip + ["numpy<2", "pycocotools==2.0.7"])
        if not ok:
            say("\n  [warn] mmdetection stack unavailable. Continuing with torchvision only.")

    # ---- stage 4: verify -----------------------------------------------------------------
    rule("STAGE 4/4  verify")
    info = probe()
    have_mmdet = bool(info.get("have_mmdet"))
    report(info)

    if not info.get("torch_ok"):
        say("\nERROR: verification failed.")
        return 1

    # freeze
    try:
        out = subprocess.run([uv, "pip", "freeze", "--python", str(py)],
                             capture_output=True, text=True, timeout=300)
        if out.returncode == 0:
            LOCK.write_text(
                "# Generated by scripts/setup_env.py -- the exact working stack.\n"
                f"# python {info.get('python')}  |  wheels {cuda}  |  gpu {gpu_desc}\n"
                + out.stdout
            )
            say(f"\nwrote {LOCK.name}")
    except Exception as e:
        say(f"[warn] could not write {LOCK.name}: {e}")

    write_stamp(cuda, True, have_mmdet, gpu_desc, {
        k: info.get(k) for k in ("python", "torch", "torchvision", "mmcv", "mmdet", "numpy")
    })

    rule("READY")
    say(f"  venv python : {py}")
    say(f"  stacks      : torchvision{' + mmdet' if have_mmdet else ''}")
    if not have_mmdet and not args.no_mmdet:
        say("  note        : mmdet unavailable -- `--stack mmdet` will be skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
