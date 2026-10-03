#!/usr/bin/env python3
"""
Model Downloader for Mini VeeVee (Lens Diffusion Model, CLIP Text Encoder, and VAE)
===================================================================================

This script downloads the required models for running the Mini VeeVee ComfyUI pipeline:
1. UNET Diffusion Model: lens_bf16.safetensors (~7.6 GB)
2. CLIP Text Encoder:    gpt_oss_20b_nvfp4.safetensors (~12.3 GB)
3. VAE Model:            flux2-vae.safetensors (~320 MB)

Usage Modes:
  1. Remote Trigger (Default for RunPod ComfyUI):
     Triggers fast server-side download directly on the RunPod pod and monitors progress.
     python3 download_models.py --remote

  2. Direct Local / Pod Download:
     Downloads model files directly to your local ComfyUI models folder or local disk.
     python3 download_models.py --local --output-dir /workspace/runpod-slim/ComfyUI/models

  3. Status Check:
     Check current download progress and model availability on the remote pod.
     python3 download_models.py --status
"""

import os
import sys
import time
import json
import ssl
import argparse
import urllib.request
import urllib.error

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DEFAULT_COMFY_URL = os.environ.get("COMFY_URL", "https://7a1raczqkrw81q-8188.proxy.runpod.net").rstrip("/")
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

MODELS = [
    {
        "name": "Diffusion Model (Lens BF16)",
        "filename": "lens_bf16.safetensors",
        "save_path": "diffusion_models",
        "local_subdir": "unet/lens",
        "url": "https://huggingface.co/Comfy-Org/Lens/resolve/main/diffusion_models/lens_bf16.safetensors",
        "size_bytes": 8159392256,
        "size_str": "7.6 GB",
    },
    {
        "name": "CLIP Text Encoder (GPT OSS 20B NVFP4)",
        "filename": "gpt_oss_20b_nvfp4.safetensors",
        "save_path": "text_encoders",
        "local_subdir": "text_encoders",
        "url": "https://huggingface.co/Comfy-Org/Lens/resolve/main/text_encoders/gpt_oss_20b_nvfp4.safetensors",
        "size_bytes": 13210459392,
        "size_str": "12.3 GB",
    },
    {
        "name": "Flux2 VAE",
        "filename": "flux2-vae.safetensors",
        "save_path": "vae",
        "local_subdir": "vae",
        "url": "https://huggingface.co/Comfy-Org/Lens/resolve/main/vae/flux2-vae.safetensors",
        "size_bytes": 336213556,
        "size_str": "320 MB",
    },
]


def format_bytes(bytes_num):
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_num) < 1024.0:
            return f"{bytes_num:3.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} PB"


def _http_get(url, timeout=10):
    """Perform HTTP GET with requests or urllib fallback."""
    if HAS_REQUESTS:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            return r.status_code, r.json() if "application/json" in r.headers.get("Content-Type", "") or r.text.startswith(("{", "[")) else r.text
        except Exception as e:
            return None, str(e)

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(text)
            except Exception:
                return resp.status, text
    except Exception as e:
        return None, str(e)


def _http_post_json(url, payload, timeout=15):
    """Perform HTTP POST JSON with requests or urllib fallback."""
    if HAS_REQUESTS:
        try:
            r = requests.post(url, json=payload, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            try:
                data = r.json()
            except Exception:
                data = r.text
            return r.status_code, data
        except Exception as e:
            return None, str(e)

    ctx = ssl._create_unverified_context()
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(text)
            except Exception:
                return resp.status, text
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, err_body
    except Exception as e:
        return None, str(e)


def check_remote_health(comfy_url):
    """Check if the ComfyUI remote instance is reachable."""
    status, data = _http_get(f"{comfy_url}/system_stats", timeout=8)
    if status == 200 and isinstance(data, dict):
        return True, data
    return False, data or "Unknown error"


def get_remote_models(comfy_url):
    """Check currently loaded models in ComfyUI."""
    status, data = _http_get(f"{comfy_url}/object_info", timeout=12)
    if status == 200 and isinstance(data, dict):
        unet = data.get("UNETLoader", {}).get("input", {}).get("required", {}).get("unet_name", [[]])[0]
        clip = data.get("CLIPLoader", {}).get("input", {}).get("required", {}).get("clip_name", [[]])[0]
        vae = data.get("VAELoader", {}).get("input", {}).get("required", {}).get("vae_name", [[]])[0]
        return {"unet": unet, "clip": clip, "vae": vae}
    return {"unet": [], "clip": [], "vae": []}


def get_remote_download_status(comfy_url):
    """Fetch status of active and completed downloads on the remote pod."""
    status, data = _http_get(f"{comfy_url}/server_download/status", timeout=8)
    if status == 200 and isinstance(data, dict):
        return data
    return {}


def trigger_remote_download(comfy_url, model_info, token=None):
    """Trigger a server-side download via ComfyUI-RunpodDirect extension."""
    payload = {
        "url": model_info["url"],
        "save_path": model_info["save_path"],
        "filename": model_info["filename"],
    }
    if token:
        payload["token"] = token

    status_code, res = _http_post_json(f"{comfy_url}/server_download/start", payload, timeout=15)
    if isinstance(res, dict):
        return res.get("success", False), res.get("download_id", ""), res.get("message", res.get("error", ""))
    return False, "", str(res)


def run_remote_download(comfy_url, hf_token=None):
    """Trigger and track remote downloads on RunPod pod."""
    print(f"\n========================================================")
    print(f"  Mini VeeVee Remote Model Downloader")
    print(f"  Pod Endpoint: {comfy_url}")
    print(f"========================================================\n")

    print("[*] Connecting to RunPod ComfyUI pod...")
    healthy, stats_or_err = check_remote_health(comfy_url)
    if not healthy:
        print(f"[!] Error: Unable to reach ComfyUI pod at {comfy_url}")
        print(f"    Details: {stats_or_err}")
        print("[!] If you want to download locally, run with: python3 download_models.py --local")
        sys.exit(1)

    devices = stats_or_err.get("devices", [])
    gpu_name = devices[0].get("name", "GPU") if devices else "CPU/Pod"
    print(f"[+] Connected successfully! Pod Target GPU: {gpu_name}")

    # Check already loaded models and active downloads
    loaded = get_remote_models(comfy_url)
    current_status = get_remote_download_status(comfy_url)

    tasks_to_start = []
    for model in MODELS:
        fname = model["filename"]
        download_key = f"{model['save_path']}/{fname}"

        # Check if already present in ComfyUI loaded list
        already_loaded = False
        if model["save_path"] == "diffusion_models" and any(fname in m for m in loaded.get("unet", [])):
            already_loaded = True
        elif model["save_path"] == "text_encoders" and any(fname in m for m in loaded.get("clip", [])):
            already_loaded = True
        elif model["save_path"] == "vae" and any(fname in m for m in loaded.get("vae", [])):
            already_loaded = True

        dl_info = current_status.get(download_key, {})
        dl_status = dl_info.get("status")

        if already_loaded or dl_status == "completed":
            print(f"  [✓] {model['name']} ({fname}) is already available.")
        elif dl_status in ["downloading", "queued"]:
            progress = dl_info.get("progress", 0.0)
            print(f"  [~] {model['name']} is already downloading ({progress:.1f}% complete)...")
            tasks_to_start.append(model)
        else:
            tasks_to_start.append(model)

    if not tasks_to_start:
        print("\n🎉 All required models are already installed and ready on the ComfyUI pod!")
        return

    print(f"\n[*] Triggering downloads for {len(tasks_to_start)} model(s) directly on the pod...")
    for model in tasks_to_start:
        fname = model["filename"]
        download_key = f"{model['save_path']}/{fname}"
        dl_info = current_status.get(download_key, {})
        if dl_info.get("status") in ["downloading", "queued"]:
            continue

        print(f"  -> Queueing {model['name']} ({model['size_str']})...")
        ok, dl_id, msg = trigger_remote_download(comfy_url, model, token=hf_token)
        if ok:
            print(f"     [Queued] ID: {dl_id}")
        else:
            print(f"     [!] Notice: {msg}")

    print("\n[*] Monitoring live download progress (pod downloads directly at high speed)...")
    start_time = time.time()
    while True:
        status_map = get_remote_download_status(comfy_url)
        all_done = True
        progress_lines = []

        for model in MODELS:
            fname = model["filename"]
            dl_key = f"{model['save_path']}/{fname}"
            info = status_map.get(dl_key, {})
            status = info.get("status", "pending")
            progress = float(info.get("progress", 0.0))
            downloaded = info.get("downloaded", 0)
            total = info.get("total", model["size_bytes"])

            if status != "completed":
                all_done = False

            if status == "completed":
                badge = "[100.0% COMPLETED]"
            elif status == "downloading":
                badge = f"[{progress:5.1f}% DOWNLOADING {format_bytes(downloaded)}/{format_bytes(total)}]"
            elif status == "queued":
                badge = "[   QUEUED   ]"
            else:
                badge = f"[{status.upper()}]"

            progress_lines.append(f"  * {model['filename'][:30]:<30} {badge}")

        # Print snapshot
        elapsed = int(time.time() - start_time)
        output_block = f"\nElapsed: {elapsed}s | Active Downloads:\n" + "\n".join(progress_lines)
        print(output_block)

        if all_done:
            print("\n========================================================")
            print("🎉 ALL MODELS SUCCESSFULLY DOWNLOADED ON THE RUNPOD POD!")
            print("========================================================")
            print("Your Mini VeeVee ComfyUI instance is fully armed and ready.")
            break

        time.sleep(5)


def download_file_locally(url, destination_path, expected_size=0):
    """Download a large file directly with resume support and terminal progress."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    temp_path = destination_path + ".part"

    existing_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    headers = {"User-Agent": USER_AGENT}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    if HAS_REQUESTS:
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            mode = "ab" if existing_bytes > 0 and r.status_code == 206 else "wb"
            if mode == "wb":
                existing_bytes = 0

            content_length = r.headers.get("content-length")
            total_size = int(content_length) + existing_bytes if content_length else expected_size

            downloaded = existing_bytes
            start_t = time.time()

            with open(temp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        elapsed = max(0.1, time.time() - start_t)
                        speed = (downloaded - existing_bytes) / elapsed
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        bar_len = 30
                        filled = int(bar_len * percent / 100)
                        bar = "=" * filled + "-" * (bar_len - filled)

                        sys.stdout.write(
                            f"\r  [{bar}] {percent:5.1f}% | {format_bytes(downloaded)}/{format_bytes(total_size)} | {format_bytes(speed)}/s"
                        )
                        sys.stdout.flush()

            print()
            os.rename(temp_path, destination_path)
            return True
        except Exception as e:
            print(f"\n[!] Download error: {e}")
            return False

    # Fallback to urllib
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            content_length = resp.headers.get("Content-Length")
            total_size = int(content_length) + existing_bytes if content_length else expected_size

            mode = "ab" if existing_bytes > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                existing_bytes = 0

            downloaded = existing_bytes
            chunk_size = 1024 * 1024

            start_t = time.time()
            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    elapsed = max(0.1, time.time() - start_t)
                    speed = (downloaded - existing_bytes) / elapsed
                    percent = (downloaded / total_size * 100) if total_size > 0 else 0
                    bar_len = 30
                    filled = int(bar_len * percent / 100)
                    bar = "=" * filled + "-" * (bar_len - filled)

                    sys.stdout.write(
                        f"\r  [{bar}] {percent:5.1f}% | {format_bytes(downloaded)}/{format_bytes(total_size)} | {format_bytes(speed)}/s"
                    )
                    sys.stdout.flush()

        print()
        os.rename(temp_path, destination_path)
        return True

    except Exception as e:
        print(f"\n[!] Download error: {e}")
        return False


def run_local_download(output_base_dir):
    """Download models directly to a local directory structure."""
    print(f"\n========================================================")
    print(f"  Mini VeeVee Local Model Downloader")
    print(f"  Output Directory: {os.path.abspath(output_base_dir)}")
    print(f"========================================================\n")

    for idx, model in enumerate(MODELS, 1):
        target_dir = os.path.join(output_base_dir, model["local_subdir"])
        target_file = os.path.join(target_dir, model["filename"])

        print(f"[{idx}/{len(MODELS)}] {model['name']}")
        print(f"  Target File: {target_file}")
        print(f"  Size:        {model['size_str']}")

        if os.path.exists(target_file):
            size = os.path.getsize(target_file)
            if size > 1024 * 1024:
                print(f"  [✓] Model already downloaded ({format_bytes(size)}). Skipping.\n")
                continue

        print(f"  Downloading from: {model['url']}")
        success = download_file_locally(model["url"], target_file, expected_size=model["size_bytes"])
        if success:
            print(f"  [✓] Completed: {model['filename']}\n")
        else:
            print(f"  [✗] Failed to download {model['filename']}\n")

    print("========================================================")
    print("Download process completed.")
    print("========================================================")


def print_status(comfy_url):
    """Print current status of ComfyUI and models."""
    print(f"\n[*] Checking RunPod ComfyUI pod status at: {comfy_url}")
    healthy, stats = check_remote_health(comfy_url)
    if not healthy:
        print(f"[!] Pod is not reachable: {stats}")
        return

    print("[+] Pod is ONLINE.")
    models = get_remote_models(comfy_url)
    downloads = get_remote_download_status(comfy_url)

    print("\n--- Model Availability ---")
    print(f"  UNET / Diffusion: {models.get('unet', [])}")
    print(f"  CLIP / Encoders:  {models.get('clip', [])}")
    print(f"  VAE:              {models.get('vae', [])}")

    if downloads:
        print("\n--- Active/Recent Downloads on Pod ---")
        for key, info in downloads.items():
            print(f"  * {key}: status={info.get('status')} progress={info.get('progress')}%")
    else:
        print("\n--- No background downloads recorded ---")


def main():
    parser = argparse.ArgumentParser(description="Download required models for Mini VeeVee ComfyUI")
    parser.add_argument("--remote", action="store_true", help="Trigger fast server-side download on RunPod ComfyUI pod (default)")
    parser.add_argument("--local", action="store_true", help="Download models directly to local directory")
    parser.add_argument("--status", action="store_true", help="Check download status on remote ComfyUI pod")
    parser.add_argument("--url", default=DEFAULT_COMFY_URL, help=f"ComfyUI base URL (default: {DEFAULT_COMFY_URL})")
    parser.add_argument("--output-dir", default="./models", help="Base directory for local downloads (default: ./models)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face access token")

    args = parser.parse_args()

    if args.status:
        print_status(args.url)
    elif args.local:
        run_local_download(args.output_dir)
    else:
        healthy, _ = check_remote_health(args.url)
        if healthy:
            run_remote_download(args.url, hf_token=args.hf_token)
        else:
            print(f"Remote pod at {args.url} is not reachable. Falling back to local download...")
            run_local_download(args.output_dir)


if __name__ == "__main__":
    main()
