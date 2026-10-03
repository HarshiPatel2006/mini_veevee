# generator/services.py
import os
import sys
import time
import json
import random
import logging
import signal
import atexit
import threading
import requests
from pathlib import Path
from django.conf import settings
from django.core.files.base import ContentFile
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load local .env if present
env_path = Path(settings.BASE_DIR) / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# RunPod Configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
DEFAULT_COMFY_URL = os.environ.get("COMFY_URL", "https://7a1raczqkrw81q-8188.proxy.runpod.net").rstrip("/")

# Target GPUs prioritized as requested: RTX PRO 4000, RTX 3090, L4
PREFERRED_GPUS = [
    "NVIDIA RTX PRO 4000 Blackwell",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA L4",
]

# Required models manifest for the Lens txt2img ComfyUI pipeline
REQUIRED_MODELS = [
    {
        "name": "Diffusion Model (Lens BF16)",
        "filename": "lens_bf16.safetensors",
        "save_path": "diffusion_models",
        "url": "https://huggingface.co/Comfy-Org/Lens/resolve/main/diffusion_models/lens_bf16.safetensors",
        "type": "unet"
    },
    {
        "name": "CLIP Text Encoder (GPT OSS 20B NVFP4)",
        "filename": "gpt_oss_20b_nvfp4.safetensors",
        "save_path": "text_encoders",
        "url": "https://huggingface.co/Comfy-Org/Lens/resolve/main/text_encoders/gpt_oss_20b_nvfp4.safetensors",
        "type": "clip"
    },
    {
        "name": "Flux2 VAE",
        "filename": "flux2-vae.safetensors",
        "save_path": "vae",
        "url": "https://huggingface.co/Comfy-Org/Lens/resolve/main/vae/flux2-vae.safetensors",
        "type": "vae"
    }
]

# Track active pods for lifecycle management
ACTIVE_POD_IDS = set()
LAST_HEARTBEAT_TIME = time.time()
_HEARTBEAT_LOCK = threading.Lock()
_ACTIVE_GENERATION_COUNT = 0
_GEN_LOCK = threading.Lock()


def get_runpod_api_key():
    """Dynamically fetch the RunPod API key from environment or .env file."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env_f = Path(settings.BASE_DIR) / '.env'
        if env_f.exists():
            load_dotenv(dotenv_path=env_f, override=True)
            key = os.environ.get("RUNPOD_API_KEY")
    return key


def get_runpod_client():
    """Lazily configure and return runpod module."""
    try:
        import runpod
        api_key = get_runpod_api_key()
        if api_key:
            runpod.api_key = api_key
        return runpod
    except ImportError:
        logger.warning("runpod package not installed or import failed.")
        return None



def terminate_pod(pod_id):
    """Terminate a specific RunPod pod immediately to avoid extra costs."""
    if not pod_id:
        return False
    rp = get_runpod_client()
    if not rp:
        return False
    try:
        logger.info(f"[RunPod] Terminating pod: {pod_id}")
        rp.terminate_pod(pod_id)
        ACTIVE_POD_IDS.discard(pod_id)
        logger.info(f"[RunPod] Pod {pod_id} successfully terminated.")
        return True
    except Exception as e:
        logger.error(f"[RunPod Error] Failed to terminate pod {pod_id}: {e}")
        return False


def terminate_all_active_pods():
    """Terminate all tracked active RunPod pods."""
    pods_to_kill = list(ACTIVE_POD_IDS)
    for pid in pods_to_kill:
        terminate_pod(pid)


# Register exit handlers so closing the terminal / Django process terminates the pods immediately
def _exit_cleanup(signum=None, frame=None):
    logger.info("[Lifecycle] Process exit detected. Terminating any active RunPod pods...")
    terminate_all_active_pods()
    if signum is not None:
        sys.exit(0)

atexit.register(terminate_all_active_pods)
try:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _exit_cleanup)
        signal.signal(signal.SIGTERM, _exit_cleanup)
except Exception:
    pass


# Heartbeat & Web Disconnect Watchdog
def record_heartbeat():
    """Record client activity from the web UI."""
    global LAST_HEARTBEAT_TIME
    with _HEARTBEAT_LOCK:
        LAST_HEARTBEAT_TIME = time.time()


def handle_browser_disconnect():
    """Called when user closes tab or browser window via sendBeacon."""
    global LAST_HEARTBEAT_TIME
    with _HEARTBEAT_LOCK:
        LAST_HEARTBEAT_TIME = 0  # Mark expired immediately
    with _GEN_LOCK:
        active_gen = _ACTIVE_GENERATION_COUNT
    if active_gen == 0 and ACTIVE_POD_IDS:
        logger.info("[Lifecycle] Browser tab closed and no generation running. Terminating pods now...")
        terminate_all_active_pods()


def _heartbeat_watchdog_loop():
    """Background monitor: terminates active pods if website was closed for > 45 seconds."""
    while True:
        time.sleep(10)
        with _HEARTBEAT_LOCK:
            idle_seconds = time.time() - LAST_HEARTBEAT_TIME
        with _GEN_LOCK:
            active_gen = _ACTIVE_GENERATION_COUNT

        if ACTIVE_POD_IDS and active_gen == 0 and idle_seconds > 45:
            logger.info(f"[Lifecycle] Inactive for {int(idle_seconds)}s with no open website. Auto-terminating pods...")
            terminate_all_active_pods()

# Start background watchdog daemon
_watchdog_thread = threading.Thread(target=_heartbeat_watchdog_loop, daemon=True)
_watchdog_thread.start()


def get_pod_comfy_url(pod_id=None):
    """Construct proxy URL for a pod ID, or return fallback COMFY_URL."""
    if pod_id:
        return f"https://{pod_id}-8188.proxy.runpod.net"
    return os.environ.get("COMFY_URL", DEFAULT_COMFY_URL).rstrip("/")


def check_pod_health(comfy_url):
    """Check if ComfyUI on pod is responsive and return hardware stats."""
    try:
        res = requests.get(f"{comfy_url}/system_stats", timeout=5)
        if res.status_code == 200:
            data = res.json()
            devices = data.get("devices", [])
            gpu_name = devices[0].get("name", "NVIDIA GPU") if devices else "NVIDIA GPU"
            vram_total = round(devices[0].get("vram_total", 0) / (1024 ** 3), 1) if devices else 0
            vram_free = round(devices[0].get("vram_free", 0) / (1024 ** 3), 1) if devices else 0
            return True, {
                "gpu_name": gpu_name,
                "vram_total": vram_total,
                "vram_free": vram_free,
                "system": data.get("system", {})
            }
    except Exception as e:
        return False, str(e)
    return False, "Non-200 status code"


def get_loaded_models(comfy_url):
    """Retrieve recognized model filenames from ComfyUI object_info."""
    models = {"unet": [], "clip": [], "vae": []}
    try:
        r = requests.get(f"{comfy_url}/object_info", timeout=8)
        if r.status_code == 200:
            data = r.json()
            unet_req = data.get("UNETLoader", {}).get("input", {}).get("required", {})
            clip_req = data.get("CLIPLoader", {}).get("input", {}).get("required", {})
            vae_req = data.get("VAELoader", {}).get("input", {}).get("required", {})

            if "unet_name" in unet_req and unet_req["unet_name"]:
                models["unet"] = unet_req["unet_name"][0]
            if "clip_name" in clip_req and clip_req["clip_name"]:
                models["clip"] = clip_req["clip_name"][0]
            if "vae_name" in vae_req and vae_req["vae_name"]:
                models["vae"] = vae_req["vae_name"][0]
    except Exception as e:
        logger.warning(f"Error querying ComfyUI models: {e}")
    return models


def ensure_models_on_pod(comfy_url, progress_callback=None):
    """
    Ensure required Lens models are downloaded and recognized by ComfyUI.
    Uses ComfyUI-RunpodDirect /server_download/start API directly on the pod for 500MB/s speeds.
    """
    loaded = get_loaded_models(comfy_url)
    
    has_unet = any("lens" in str(m).lower() for m in loaded.get("unet", []))
    has_clip = any("gpt_oss" in str(m).lower() or "lens" in str(m).lower() for m in loaded.get("clip", []))
    has_vae = any("flux" in str(m).lower() for m in loaded.get("vae", []))

    if has_unet and has_clip and has_vae:
        logger.info("[RunPod] All required models already present in ComfyUI!")
        if progress_callback:
            progress_callback(100, "All models loaded and verified.")
        return True

    # Need to trigger downloads for missing models
    missing = []
    if not has_unet:
        missing.append(REQUIRED_MODELS[0])
    if not has_clip:
        missing.append(REQUIRED_MODELS[1])
    if not has_vae:
        missing.append(REQUIRED_MODELS[2])

    logger.info(f"[RunPod] Triggering automated download of {len(missing)} missing models: {[m['filename'] for m in missing]}")

    for idx, model in enumerate(missing):
        if progress_callback:
            progress_callback(int((idx / len(missing)) * 50), f"Initiating download: {model['name']}...")
        payload = {
            "url": model["url"],
            "save_path": model["save_path"],
            "filename": model["filename"],
        }
        try:
            requests.post(f"{comfy_url}/server_download/start", json=payload, timeout=15)
        except Exception as e:
            logger.warning(f"Could not trigger /server_download/start for {model['filename']}: {e}")

    # Poll /server_download/status until downloads finish
    for poll_step in range(120):  # up to 10 minutes
        time.sleep(5)
        try:
            # Refresh ComfyUI extra model paths periodically
            try:
                requests.post(f"{comfy_url}/extra_model_paths", timeout=5)
            except Exception:
                pass

            loaded = get_loaded_models(comfy_url)
            has_unet = any("lens" in str(m).lower() for m in loaded.get("unet", []))
            has_clip = any("gpt_oss" in str(m).lower() or "lens" in str(m).lower() for m in loaded.get("clip", []))
            has_vae = any("flux" in str(m).lower() for m in loaded.get("vae", []))

            # Fetch download status for detail
            dl_status = {}
            try:
                dl_status = requests.get(f"{comfy_url}/server_download/status", timeout=5).json()
            except Exception:
                pass

            if has_unet and has_clip and has_vae:
                logger.info("[RunPod] All models successfully verified on ComfyUI pod!")
                if progress_callback:
                    progress_callback(100, "All models downloaded and ready!")
                return True

            active_info = []
            for k, v in dl_status.items():
                if isinstance(v, dict) and v.get("status") == "downloading":
                    prog = round(v.get("progress", 0), 1)
                    active_info.append(f"{k.split('/')[-1]} ({prog}%)")

            detail_str = f"Downloading models: {', '.join(active_info)}" if active_info else "Downloading models in background..."
            if progress_callback:
                progress_callback(min(90, 20 + poll_step), detail_str)

        except Exception as e:
            logger.warning(f"Polling model status error: {e}")

    # Final check
    loaded = get_loaded_models(comfy_url)
    return any("lens" in str(m).lower() for m in loaded.get("unet", []))


def provision_pod(gpu_candidates=None, task_callback=None):
    """
    Provision a high-performance RunPod GPU pod matching RTX PRO 4000 / RTX 3090 / L4.
    """
    rp = get_runpod_client()
    if not rp:
        raise Exception("RunPod SDK not configured. Please set RUNPOD_API_KEY.")

    if not gpu_candidates:
        gpu_candidates = PREFERRED_GPUS

    startup_script = "bash -c 'mkdir -p /workspace/runpod-slim/ComfyUI/models/diffusion_models /workspace/runpod-slim/ComfyUI/models/unet/lens /workspace/runpod-slim/ComfyUI/models/text_encoders /workspace/runpod-slim/ComfyUI/models/clip /workspace/runpod-slim/ComfyUI/models/vae; [ -d /workspace/ComfyUI ] && [ ! -d /workspace/runpod-slim/ComfyUI ] && ln -sf /workspace/ComfyUI /workspace/runpod-slim/ComfyUI || true; [ -d /workspace/runpod-slim/ComfyUI ] && [ ! -d /workspace/ComfyUI ] && ln -sf /workspace/runpod-slim/ComfyUI /workspace/ComfyUI || true; /start.sh & sleep 5; echo ComfyUI_Started'"

    pod = None
    last_err = None

    for gpu_id in gpu_candidates:
        try:
            if task_callback:
                task_callback(10, f"Attempting to provision {gpu_id}...")
            logger.info(f"[RunPod] Attempting provision with GPU: {gpu_id}")
            pod = rp.create_pod(
                name="veevee-pro-studio",
                image_name="runpod/comfyui:latest",
                gpu_type_id=gpu_id,
                ports="8188/http",
                container_disk_in_gb=100,
                volume_in_gb=0,
                docker_args=startup_script
            )
            if pod and "id" in pod:
                logger.info(f"[RunPod] Successfully provisioned pod {pod['id']} with {gpu_id}")
                break
        except Exception as e:
            last_err = e
            logger.warning(f"[RunPod] GPU {gpu_id} unavailable or failed: {e}")
            if "no longer any instances available" in str(e).lower() or "capacity" in str(e).lower():
                continue

    if not pod or "id" not in pod:
        raise Exception(f"Unable to provision requested GPU pod ({PREFERRED_GPUS}): {last_err}")

    pod_id = pod["id"]
    ACTIVE_POD_IDS.add(pod_id)
    pod_url = get_pod_comfy_url(pod_id)

    # Poll until ComfyUI proxy responds
    logger.info(f"[RunPod] Waiting for ComfyUI pod {pod_id} to become reachable at {pod_url}...")
    for attempt in range(60):  # up to 5 minutes
        time.sleep(5)
        if task_callback:
            task_callback(15 + int(attempt * 0.5), f"Initializing cloud container ({attempt+1}/60)...")
        healthy, _ = check_pod_health(pod_url)
        if healthy:
            logger.info(f"[RunPod] Pod {pod_id} is online!")
            break
    else:
        terminate_pod(pod_id)
        raise Exception(f"Pod {pod_id} started but ComfyUI service was unreachable.")

    # Automatically install and verify models
    if task_callback:
        task_callback(30, "Checking and downloading required pipeline models...")
    ensure_models_on_pod(pod_url, task_callback)

    return pod_id, pod_url


def build_image_workflow(prompt, aspect_ratio="1:1 (Square)", seed=None, steps=20, cfg=5.0, negative_prompt="", available_models=None):
    """
    Build clean ComfyUI workflow from workflow_api.json.
    - Strips all note / markdown cards (nodes lacking class_type).
    - Injects dynamic user prompt into Node 18.
    - Sets aspect ratio on Node 6.
    - Sets random or custom seed on Node 7.
    - Sets custom steps on Node 3 and cfg on Node 7.
    - Dynamically selects correct UNET model name matching loaded files.
    """
    json_path = os.path.join(settings.BASE_DIR, "workflow_api.json")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_wf = json.load(f)

    # CRUCIAL FIX: Exclude any nodes that do not have 'class_type' (e.g. note cards 13, 14)
    # ComfyUI throws HTTP 400 'missing_node_type' if passed nodes with class_type: null
    clean_wf = {
        k: v for k, v in raw_wf.items()
        if isinstance(v, dict) and v.get("class_type")
    }

    # 1. Update Positive Prompt (Node 18)
    if "18" in clean_wf:
        clean_wf["18"]["inputs"]["text"] = prompt

    # 2. Update Negative Prompt (Node 17)
    if "17" in clean_wf:
        clean_wf["17"]["inputs"]["text"] = negative_prompt or ""

    # 3. Update Aspect Ratio (Node 6)
    if "6" in clean_wf:
        clean_wf["6"]["inputs"]["aspect_ratio"] = aspect_ratio

    # 4. Update Random Seed & CFG (Node 7)
    if seed is None:
        seed = random.randint(1, 10**15)
    if "7" in clean_wf:
        clean_wf["7"]["inputs"]["noise_seed"] = seed
        clean_wf["7"]["inputs"]["cfg"] = float(cfg)

    # 5. Update Sampling Steps (Node 3)
    if "3" in clean_wf:
        clean_wf["3"]["inputs"]["steps"] = int(steps)

    # 6. Resolve UNET filename in Node 1
    if "1" in clean_wf:
        unet_name = "lens_bf16.safetensors"
        if available_models and available_models.get("unet"):
            loaded_unets = available_models.get("unet", [])
            for opt in ["lens_bf16.safetensors", "lens/lens_bf16.safetensors", "lens\\lens_bf16.safetensors"]:
                if opt in loaded_unets:
                    unet_name = opt
                    break
        clean_wf["1"]["inputs"]["unet_name"] = unet_name

    return clean_wf, seed


def process_generation_on_runpod(task_id, auto_terminate=False):
    """
    Full async worker:
    1. Checks if a ComfyUI pod is already online or provisions one on-demand (RTX 4000 Ada / RTX 3090 / L4).
    2. Automatically ensures models are downloaded.
    3. Executes the sanitized Lens workflow.
    4. Downloads output image and stores to GenerationTask.
    5. Optionally terminates the pod immediately if auto_terminate is True or when local website closes.
    """
    def worker():
        global _ACTIVE_GENERATION_COUNT
        with _GEN_LOCK:
            _ACTIVE_GENERATION_COUNT += 1

        from .models import GenerationTask
        task = GenerationTask.objects.get(id=task_id)
        pod_id = None
        provisioned_here = False

        def update_progress(percent, detail):
            try:
                task.refresh_from_db()
                task.progress_percent = percent
                task.status_detail = detail
                task.save(update_fields=['progress_percent', 'status_detail', 'updated_at'])
            except Exception:
                pass

        try:
            task.status = 'PROVISIONING'
            task.save()
            update_progress(5, "Connecting to RunPod GPU...")

            # 1. Determine target pod endpoint
            # First check if the fallback COMFY_URL is online and healthy
            pod_url = get_pod_comfy_url()
            healthy, _ = check_pod_health(pod_url)
            api_key = get_runpod_api_key()

            if not healthy and api_key:
                # Provision a new pod on-demand with preferred GPUs
                update_progress(10, "Deploying GPU pod (RTX PRO 4000 / RTX 3090 / L4)...")
                pod_id, pod_url = provision_pod(PREFERRED_GPUS, update_progress)
                provisioned_here = True
                task.pod_id = pod_id
                task.save()
            elif not healthy:
                raise Exception(
                    "Default ComfyUI pod is unreachable and RUNPOD_API_KEY is not set. "
                    "Please provide a valid RUNPOD_API_KEY in .env or configure COMFY_URL."
                )

            # 2. Ensure models are loaded and ready
            task.status = 'DOWNLOADING'
            task.save()
            update_progress(40, "Verifying Lens models on pod...")
            models_ok = ensure_models_on_pod(pod_url, update_progress)
            if not models_ok:
                raise Exception("Failed to download or verify required models on ComfyUI pod.")

            # 3. Build sanitized workflow from workflow_api.json
            task.status = 'GENERATING'
            task.save()
            update_progress(55, "Dispatching workflow to ComfyUI...")

            loaded_models = get_loaded_models(pod_url)
            workflow, used_seed = build_image_workflow(
                prompt=task.prompt,
                aspect_ratio=task.aspect_ratio or "1:1 (Square)",
                seed=task.seed,
                steps=task.steps or 20,
                cfg=task.cfg or 5.0,
                negative_prompt=task.negative_prompt,
                available_models=loaded_models
            )
            task.seed = used_seed
            task.save()

            # Submit prompt to ComfyUI
            resp = requests.post(f"{pod_url}/prompt", json={"prompt": workflow}, timeout=30)
            if resp.status_code != 200:
                raise Exception(f"ComfyUI rejected prompt: {resp.text}")

            prompt_res = resp.json()
            prompt_id = prompt_res.get("prompt_id")
            if not prompt_id:
                raise Exception(f"No prompt_id returned by ComfyUI: {prompt_res}")

            logger.info(f"[RunPod] Prompt submitted successfully. ID: {prompt_id}")
            update_progress(65, "Rendering image via Lens Diffusion...")

            # 4. Poll history until generation finishes
            output_filename = None
            subfolder = ""
            output_type = "output"

            for poll_idx in range(120):  # up to 6 minutes
                time.sleep(3)
                progress_step = min(95, 65 + int(poll_idx * 0.5))
                update_progress(progress_step, f"Processing latents... ({poll_idx*3}s)")

                try:
                    hist_res = requests.get(f"{pod_url}/history/{prompt_id}", timeout=5).json()
                    if prompt_id in hist_res:
                        outputs = hist_res[prompt_id].get("outputs", {})
                        for _, node_output in outputs.items():
                            if "images" in node_output and node_output["images"]:
                                img_info = node_output["images"][0]
                                output_filename = img_info.get("filename")
                                subfolder = img_info.get("subfolder", "")
                                output_type = img_info.get("type", "output")
                                break
                        break
                except Exception:
                    continue

            if not output_filename:
                raise Exception("Generation timed out or produced no output image.")

            # 5. Download rendered image and store in DB
            update_progress(96, "Saving high-res image...")
            view_url = f"{pod_url}/view?filename={output_filename}&subfolder={subfolder}&type={output_type}"
            img_bytes = requests.get(view_url, timeout=30).content

            saved_name = f"lens_{task.id}_{output_filename}"
            task.output_file.save(saved_name, ContentFile(img_bytes))
            task.status = 'COMPLETED'
            task.progress_percent = 100
            task.status_detail = "Generation finished successfully!"
            task.save()
            logger.info(f"[RunPod] Task #{task.id} completed successfully!")

        except Exception as e:
            logger.error(f"[RunPod Worker Error] Task #{task.id} failed: {e}")
            task.status = 'FAILED'
            task.error_message = str(e)
            task.progress_percent = 0
            task.status_detail = f"Error: {e}"
            task.save()

        finally:
            with _GEN_LOCK:
                _ACTIVE_GENERATION_COUNT = max(0, _ACTIVE_GENERATION_COUNT - 1)

            # Auto-terminate if requested to save costs
            if auto_terminate and pod_id and provisioned_here:
                terminate_pod(pod_id)

    threading.Thread(target=worker, daemon=True).start()