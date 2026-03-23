"""
Download RVC WebUI assets from Hugging Face (or a mirror via HF_ENDPOINT).

Prefer huggingface_hub: resumable downloads and retries help on flaky links
(ChunkedEncodingError / connection reset on hf-mirror.com).

Default — only what the myshot pipeline needs (hubert + rmvpe + f0G40k/f0D40k):
  python download_models.py

Full RVC WebUI asset bundle (optional; not required for myshot):
  python download_models.py --full

Resume one file (HTTP Range + long timeouts):
  python download_models.py --requests --only pretrained/f0G40k.pth

Or with huggingface_hub:
  python download_models.py --only pretrained/f0G40k.pth

Full bundle over requests (unstable mirrors — prefer --full with huggingface_hub):
  python download_models.py --requests --full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 64 * 1024  # smaller reads → steadier progress on flaky links
MAX_RETRIES = 16
RETRY_DELAY_BASE = 8  # exponential backoff: base * 2^(attempt-1), capped
RETRY_DELAY_CAP = 180
CONNECT_TIMEOUT = 60
READ_TIMEOUT = 900  # seconds between chunks (large files on slow mirrors)
FSYNC_EVERY_BYTES = 32 * 1024 * 1024  # flush to disk periodically for safer resume

DEFAULT_REPO_ID = "lj1995/VoiceConversionWebUI"

# HF repo relative path -> local_dir (directory under BASE_DIR) such that
# file ends up at BASE_DIR / local_dir / ... matching RVC layout.
MINIMAL_SPECS = [
    ("hubert_base.pt", "assets/hubert"),
    ("rmvpe.pt", "assets/rmvpe"),
    ("pretrained/f0G40k.pth", "assets"),
    ("pretrained/f0D40k.pth", "assets"),
]


def format_size(size_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _mirror_base() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _requests_url(repo_file: str) -> str:
    base = f"{_mirror_base()}/{DEFAULT_REPO_ID}/resolve/main/"
    return base + repo_file.lstrip("/")


def _retry_sleep(attempt: int) -> None:
    delay = min(RETRY_DELAY_CAP, RETRY_DELAY_BASE * (2 ** (attempt - 1)))
    print(f"  Retrying in {delay}s...")
    time.sleep(delay)


def _parse_content_range_total(content_range: str | None) -> int | None:
    """Parse 'bytes 0-9/100' → 100."""
    if not content_range or "/" not in content_range:
        return None
    try:
        return int(content_range.rsplit("/", 1)[-1].strip())
    except ValueError:
        return None


def dl_model_requests(url: str, file_path: Path) -> None:
    """Stream download with HTTP Range resume, long read timeout, exponential backoff."""
    import requests

    session = requests.Session()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = (CONNECT_TIMEOUT, READ_TIMEOUT)

    total_known: int | None = None
    try:
        h = session.head(url, allow_redirects=True, timeout=timeout)
        if h.ok:
            cl = h.headers.get("Content-Length")
            if cl and cl.isdigit():
                total_known = int(cl)
    except requests.exceptions.RequestException:
        pass

    def on_disk() -> int:
        return file_path.stat().st_size if file_path.exists() else 0

    start = on_disk()
    if total_known is not None and start == total_known:
        print(f"  Already complete: {format_size(start)} — {file_path.name}")
        return
    if total_known is not None and start > total_known:
        file_path.unlink(missing_ok=True)
        start = 0

    print(f"  URL: {url}")

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        start = on_disk()
        headers: dict[str, str] = {}

        if total_known is not None and 0 < start < total_known:
            headers["Range"] = f"bytes={start}-"
            print(f"  Resuming from {format_size(start)} / {format_size(total_known)} (Range request)")
        elif start > 0:
            # HEAD may fail on some mirrors; still try Range — 206 gives Content-Range / total size
            headers["Range"] = f"bytes={start}-"
            print(
                f"  Resuming from {format_size(start)}"
                + (f" / {format_size(total_known)}" if total_known else " (total size unknown until response)")
                + " (Range request)"
            )

        r = None
        try:
            r = session.get(url, stream=True, headers=headers, timeout=timeout)
            if r.status_code == 416:
                sz = on_disk()
                if total_known is not None and sz >= total_known:
                    print(f"  Already complete (416): {file_path.name}")
                    return
                verified = total_known
                if verified is None and sz > 0:
                    try:
                        rz = session.get(
                            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout
                        )
                        verified = _parse_content_range_total(rz.headers.get("Content-Range"))
                        rz.close()
                    except requests.exceptions.RequestException:
                        verified = None
                if verified is not None and sz == verified:
                    print(f"  Already complete: {file_path.name} ({format_size(sz)})")
                    return
                file_path.unlink(missing_ok=True)
                if verified is not None:
                    total_known = verified
                attempt -= 1
                continue

            if headers.get("Range") and r.status_code == 200:
                print("  Server ignored Range; restarting full download")
                file_path.unlink(missing_ok=True)
                attempt -= 1
                continue

            r.raise_for_status()

            cr_total = _parse_content_range_total(r.headers.get("Content-Range"))
            if cr_total is not None:
                total_known = cr_total

            cl = r.headers.get("Content-Length")
            if r.status_code == 200 and not headers.get("Range") and cl and cl.isdigit():
                total_known = int(cl)

            mode = "ab" if r.status_code == 206 else "wb"
            if mode == "wb" and on_disk() > 0 and not headers.get("Range"):
                file_path.unlink(missing_ok=True)

            downloaded = on_disk() if mode == "ab" else 0
            total_for_bar = total_known
            if total_for_bar:
                print(f"  Total file size: {format_size(total_for_bar)}")

            bytes_since_fsync = 0
            with open(file_path, mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    bytes_since_fsync += len(chunk)
                    if bytes_since_fsync >= FSYNC_EVERY_BYTES:
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
                        bytes_since_fsync = 0

                    if total_for_bar:
                        pct = min(100.0, (downloaded / total_for_bar) * 100)
                        sys.stdout.write(
                            f"\r  Progress: {pct:.1f}% ({format_size(downloaded)}/{format_size(total_for_bar)})"
                        )
                    else:
                        sys.stdout.write(f"\r  Downloaded: {format_size(downloaded)}")
                    sys.stdout.flush()

                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            print()
            final = on_disk()
            if total_known is not None and final < total_known:
                # Trigger retry (picked up as transient below)
                raise ConnectionError(
                    f"Incomplete download: {final} bytes on disk, expected {total_known}"
                )
            return

        except (
            requests.exceptions.RequestException,
            OSError,
            ConnectionError,
            BrokenPipeError,
            ConnectionResetError,
        ) as e:
            err = type(e).__name__
            print(f"\n  Attempt {attempt}/{MAX_RETRIES} failed: {err}")
            if attempt < MAX_RETRIES:
                _retry_sleep(attempt)
            else:
                print(f"  ERROR: Max retries reached, giving up on {file_path.name}")
                print(f"  Tip: re-run with  --requests --only <repo-path>  to resume this file only.")
                raise
        finally:
            if r is not None:
                r.close()


def _local_path_for_repo_file(hf_path: str) -> Path:
    """Map Hugging Face repo-relative path to RVC WebUI assets layout."""
    hf_path = hf_path.strip().lstrip("/")
    if "/" in hf_path:
        return BASE_DIR / "assets" / hf_path
    if hf_path == "hubert_base.pt":
        return BASE_DIR / "assets/hubert/hubert_base.pt"
    if hf_path == "rmvpe.pt":
        return BASE_DIR / "assets/rmvpe/rmvpe.pt"
    return BASE_DIR / "assets" / hf_path


def run_single_requests(hf_path: str) -> None:
    url = _requests_url(hf_path)
    out = _local_path_for_repo_file(hf_path)
    print(f"Single file: {hf_path} → {out}")
    dl_model_requests(url, out)


def run_single_hf(repo_id: str, hf_path: str) -> None:
    hf_path = hf_path.strip().lstrip("/")
    if "/" in hf_path:
        download_one_hf(repo_id, hf_path, BASE_DIR / "assets")
    elif hf_path == "hubert_base.pt":
        download_one_hf(repo_id, hf_path, BASE_DIR / "assets/hubert")
    elif hf_path == "rmvpe.pt":
        download_one_hf(repo_id, hf_path, BASE_DIR / "assets/rmvpe")
    else:
        download_one_hf(repo_id, hf_path, BASE_DIR / "assets")


def download_one_hf(repo_id: str, filename: str, local_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    print(f"  OK: {(local_dir / filename).resolve()}")


def run_minimal_hf(repo_id: str) -> None:
    print("Minimal downloads (hubert + rmvpe + f0G40k/f0D40k only):")
    for hf_name, rel_local_parent in MINIMAL_SPECS:
        print(f"Downloading {hf_name} …")
        download_one_hf(repo_id, hf_name, BASE_DIR / rel_local_parent)


def run_minimal_requests() -> None:
    base = f"{_mirror_base()}/{DEFAULT_REPO_ID}/resolve/main/"
    print(f"Using requests (no resume). Base: {base}")
    targets = [
        ("hubert_base.pt", BASE_DIR / "assets/hubert/hubert_base.pt"),
        ("rmvpe.pt", BASE_DIR / "assets/rmvpe/rmvpe.pt"),
        ("pretrained/f0G40k.pth", BASE_DIR / "assets/pretrained/f0G40k.pth"),
        ("pretrained/f0D40k.pth", BASE_DIR / "assets/pretrained/f0D40k.pth"),
    ]
    for rel, out in targets:
        print(f"Downloading {rel} …")
        dl_model_requests(base + rel, out)


def run_full_requests() -> None:
    """Original full bundle via requests (mirrors HF_ENDPOINT)."""
    HF_MIRROR = _mirror_base()
    RVC_DOWNLOAD_LINK = f"{HF_MIRROR}/{DEFAULT_REPO_ID}/resolve/main/"
    print("=" * 60)
    print(f"HuggingFace Mirror: {HF_MIRROR}")
    print(f"Download Link: {RVC_DOWNLOAD_LINK}")
    print("=" * 60)

    print("\nDownloading hubert_base.pt...")
    dl_model_requests(RVC_DOWNLOAD_LINK + "hubert_base.pt", BASE_DIR / "assets/hubert/hubert_base.pt")
    print("Downloading rmvpe.pt...")
    dl_model_requests(RVC_DOWNLOAD_LINK + "rmvpe.pt", BASE_DIR / "assets/rmvpe/rmvpe.pt")
    print("Downloading vocals.onnx...")
    dl_model_requests(
        RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
        BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
    )

    rvc_models_dir = BASE_DIR / "assets/pretrained"
    print("Downloading pretrained models:")
    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model_requests(RVC_DOWNLOAD_LINK + "pretrained/" + model, rvc_models_dir / model)

    rvc_models_dir = BASE_DIR / "assets/pretrained_v2"
    print("Downloading pretrained models v2:")
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model_requests(RVC_DOWNLOAD_LINK + "pretrained_v2/" + model, rvc_models_dir / model)

    rvc_models_dir = BASE_DIR / "assets/uvr5_weights"
    print("Downloading uvr5_weights:")
    model_names = [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model_requests(RVC_DOWNLOAD_LINK + "uvr5_weights/" + model, rvc_models_dir / model)

    print("All models downloaded!")


def run_full_hf(repo_id: str) -> None:
    """Download the same set as run_full_requests using hf_hub_download."""
    from huggingface_hub import hf_hub_download

    jobs: list[tuple[str, Path]] = []
    root_assets = BASE_DIR / "assets"

    def add(fn: str, local_dir: Path) -> None:
        jobs.append((fn, local_dir))

    add("hubert_base.pt", BASE_DIR / "assets/hubert")
    add("rmvpe.pt", BASE_DIR / "assets/rmvpe")
    add("uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx", root_assets)
    for sub, folder in (
        ("pretrained", "pretrained"),
        ("pretrained_v2", "pretrained_v2"),
    ):
        for name in (
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ):
            add(f"{sub}/{name}", root_assets)
    for name in (
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ):
        add(f"uvr5_weights/{name}", root_assets)

    print("=" * 60)
    print(f"huggingface_hub repo: {repo_id}")
    print(f"HF_ENDPOINT: {_mirror_base()}")
    print("=" * 60)
    for fn, local_dir in jobs:
        print(f"Downloading {fn} …")
        local_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=fn, local_dir=str(local_dir))
    print("All models downloaded!")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download RVC WebUI models (default: minimal set for myshot / 40k f0 training)"
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Download the entire RVC asset bundle (not needed for myshot pipeline.py)",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="Same as default: hubert + rmvpe + pretrained f0G40k/f0D40k only",
    )
    p.add_argument(
        "--requests",
        action="store_true",
        help="Use requests instead of huggingface_hub (supports HTTP Range resume + long timeouts)",
    )
    p.add_argument(
        "--only",
        metavar="REPO_PATH",
        help="Download a single repo-relative path, e.g. pretrained/D32k.pth (resume friendly with --requests)",
    )
    p.add_argument(
        "--repo-id",
        default=os.environ.get("RVC_HF_REPO_ID", DEFAULT_REPO_ID),
        help="Hugging Face repo id (default: lj1995/VoiceConversionWebUI)",
    )
    args = p.parse_args()

    if args.full and args.minimal:
        p.error("Choose at most one of --full and --minimal")

    use_full = args.full

    if args.only:
        only = args.only.strip()
        if not only:
            p.error("--only requires a non-empty path")
        if args.requests:
            run_single_requests(only)
        else:
            try:
                run_single_hf(args.repo_id, only)
            except ImportError:
                print(
                    "huggingface_hub is not installed. For single-file resume over HTTP, use:\n"
                    "  pip install huggingface_hub\n"
                    "or:\n"
                    f"  python {Path(__file__).name} --requests --only {only!r}\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        return

    if args.requests:
        if use_full:
            run_full_requests()
        else:
            run_minimal_requests()
        return

    try:
        if use_full:
            run_full_hf(args.repo_id)
        else:
            run_minimal_hf(args.repo_id)
    except ImportError:
        print(
            "huggingface_hub is not installed. Install with:\n"
            "  pip install huggingface_hub\n"
            "Or re-run with --requests (fragile on slow mirrors).\n"
            "Default download is minimal (myshot only); add --full for the entire RVC bundle.\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
