import os
import re
import json
import random
import logging
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

logger = logging.getLogger(__name__)

COMFY_URL = "https://7a1raczqkrw81q-8188.proxy.runpod.net/".rstrip("/")
REQUEST_TIMEOUT = 30

VALID_ASPECT_RATIOS = [
    "1:1 (Square)",
    "16:9 (Landscape)",
    "9:16 (Portrait)",
    "4:3 (Landscape)",
    "3:4 (Portrait)",
]


def index(request):
    """Render the Apple-inspired AI Generator UI."""
    return render(request, "generator/index.html")


def _get_comfy_available_models():
    """Query ComfyUI to discover currently loaded model filenames."""
    models = {"unet": [], "clip": [], "vae": []}
    try:
        r = requests.get(f"{COMFY_URL}/object_info", timeout=10)
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
        logger.warning(f"Could not fetch object_info from ComfyUI: {e}")
    return models


def _resolve_model_name(available_list, candidates, default_name):
    """Pick the best matching model filename from what ComfyUI actually sees."""
    for cand in candidates:
        if cand in available_list:
            return cand
    return default_name


@csrf_exempt
def system_status(request):
    """
    Check the health of the RunPod ComfyUI pod, GPU status,
    and active model downloads.
    """
    status_data = {
        "connected": False,
        "pod_url": COMFY_URL,
        "gpu_name": "RunPod GPU",
        "vram_free_gb": 0,
        "vram_total_gb": 0,
        "models_ready": False,
        "downloads": {},
    }

    try:
        stats_res = requests.get(f"{COMFY_URL}/system_stats", timeout=6)
        if stats_res.status_code == 200:
            status_data["connected"] = True
            stats = stats_res.json()
            devices = stats.get("devices", [])
            if devices:
                dev = devices[0]
                status_data["gpu_name"] = dev.get("name", "NVIDIA GPU")
                vram_total = dev.get("vram_total", 0) / (1024 ** 3)
                vram_free = dev.get("vram_free", 0) / (1024 ** 3)
                status_data["vram_total_gb"] = round(vram_total, 1)
                status_data["vram_free_gb"] = round(vram_free, 1)
    except Exception:
        status_data["connected"] = False

    # Check downloads progress
    try:
        dl_res = requests.get(f"{COMFY_URL}/server_download/status", timeout=6)
        if dl_res.status_code == 200:
            status_data["downloads"] = dl_res.json()
    except Exception:
        pass

    # Check if key models are loaded
    models = _get_comfy_available_models()
    has_unet = any("lens" in m.lower() for m in models.get("unet", []))
    has_clip = any("gpt_oss" in m.lower() or "lens" in m.lower() for m in models.get("clip", []))
    has_vae = any("flux" in m.lower() for m in models.get("vae", []))
    status_data["models_ready"] = has_unet and has_clip and has_vae
    status_data["available_models"] = models

    return JsonResponse(status_data)


@csrf_exempt
def generate_image(request):
    """Handle generation requests by validating inputs and dispatching workflow to RunPod ComfyUI."""
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method not allowed. POST required."}, status=405)

    user_prompt = request.POST.get("prompt", "").strip()
    if not user_prompt:
        return JsonResponse({"status": "error", "message": "Please enter a descriptive prompt."}, status=400)

    if len(user_prompt) > 3000:
        return JsonResponse({"status": "error", "message": "Prompt is too long (maximum 3000 characters)."}, status=400)

    aspect_ratio = request.POST.get("aspect_ratio", "1:1 (Square)")
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        aspect_ratio = "1:1 (Square)"

    json_path = os.path.join(settings.BASE_DIR, "workflow_api.json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # Strip UI/Note nodes (nodes missing 'class_type')
        clean_workflow = {
            node_id: node_data
            for node_id, node_data in workflow.items()
            if isinstance(node_data, dict) and node_data.get("class_type") is not None
        }

        # Query ComfyUI for actual recognized model names
        available = _get_comfy_available_models()

        # 1. Update Lens Positive Prompt (Node 18)
        if "18" in clean_workflow:
            clean_workflow["18"]["inputs"]["text"] = user_prompt

        # 2. Update Resolution Selector (Node 6)
        if "6" in clean_workflow:
            clean_workflow["6"]["inputs"]["aspect_ratio"] = aspect_ratio

        # 3. Dynamic seed for fresh generations (Node 7)
        if "7" in clean_workflow:
            clean_workflow["7"]["inputs"]["noise_seed"] = random.randint(1, 10**15)

        # 4. Resolve UNET model path
        if "1" in clean_workflow:
            unet_choice = _resolve_model_name(
                available.get("unet", []),
                ["lens_bf16.safetensors", "lens/lens_bf16.safetensors", "lens\\lens_bf16.safetensors"],
                "lens_bf16.safetensors"
            )
            clean_workflow["1"]["inputs"]["unet_name"] = unet_choice

        # 5. Resolve CLIP model path
        if "9" in clean_workflow:
            clip_choice = _resolve_model_name(
                available.get("clip", []),
                ["gpt_oss_20b_nvfp4.safetensors", "lens/gpt_oss_20b_nvfp4.safetensors"],
                "gpt_oss_20b_nvfp4.safetensors"
            )
            clean_workflow["9"]["inputs"]["clip_name"] = clip_choice

        # 6. Resolve VAE model path
        if "10" in clean_workflow:
            vae_choice = _resolve_model_name(
                available.get("vae", []),
                ["flux2-vae.safetensors", "lens/flux2-vae.safetensors"],
                "flux2-vae.safetensors"
            )
            clean_workflow["10"]["inputs"]["vae_name"] = vae_choice

        # 7. Dispatch payload to ComfyUI
        payload = {"prompt": clean_workflow}
        response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=REQUEST_TIMEOUT)

        try:
            res_data = response.json()
        except ValueError:
            return JsonResponse({
                "status": "error",
                "message": f"RunPod returned non-JSON response (HTTP {response.status_code})"
            }, status=502)

        prompt_id = res_data.get("prompt_id")

        if not prompt_id:
            # Parse node errors for clear user-facing explanation
            error_details = []
            if "node_errors" in res_data:
                for node_id, node_err in res_data["node_errors"].items():
                    class_type = node_err.get("class_type", f"Node {node_id}")
                    for err in node_err.get("errors", []):
                        if err.get("type") == "value_not_in_list":
                            received = err.get("extra_info", {}).get("received_value", "")
                            error_details.append(f"{class_type}: Model file '{received}' is still downloading or not ready in ComfyUI.")
                        else:
                            error_details.append(f"{class_type}: {err.get('message', 'Validation error')}")

            if error_details:
                user_msg = " | ".join(error_details)
            else:
                user_msg = res_data.get("error", {}).get("message") if isinstance(res_data.get("error"), dict) else str(res_data)

            return JsonResponse({
                "status": "error",
                "message": user_msg or "Prompt was rejected by ComfyUI pipeline."
            }, status=400)

        return JsonResponse({
            "status": "success",
            "prompt_id": prompt_id
        })

    except FileNotFoundError:
        return JsonResponse({"status": "error", "message": "workflow_api.json not found in project root"}, status=500)
    except requests.Timeout:
        return JsonResponse({"status": "error", "message": "Connection to RunPod timed out. Please verify pod status."}, status=504)
    except requests.ConnectionError:
        return JsonResponse({"status": "error", "message": "Could not reach RunPod. Verify pod is running and accessible."}, status=502)
    except Exception as e:
        logger.exception("Unexpected error during image generation")
        return JsonResponse({"status": "error", "message": f"Server error: {str(e)}"}, status=500)


@csrf_exempt
def check_status(request, prompt_id):
    """Poll generation status and return result image when completed."""
    # Sanitize prompt_id (UUID or alphanumeric with dashes/underscores)
    if not prompt_id or not re.match(r'^[a-zA-Z0-9_\-]+$', prompt_id):
        return JsonResponse({"status": "error", "message": "Invalid prompt ID format"}, status=400)

    try:
        # Check history first
        history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
        if history_res.status_code == 200:
            history_data = history_res.json()
            if prompt_id in history_data:
                item = history_data[prompt_id]
                status_info = item.get("status", {})

                # Check if prompt execution failed
                if status_info.get("status_str") == "error":
                    messages = status_info.get("messages", [])
                    err_text = messages[0][1] if messages and len(messages[0]) > 1 else "Generation failed during execution."
                    return JsonResponse({"status": "error", "message": str(err_text)})

                outputs = item.get("outputs", {})
                # Node 11 is SaveImage in Lens workflow
                if "11" in outputs and "images" in outputs["11"] and outputs["11"]["images"]:
                    image_info = outputs["11"]["images"][0]
                    filename = image_info.get("filename", "")
                    subfolder = image_info.get("subfolder", "")
                    image_type = image_info.get("type", "output")

                    image_url = f"{COMFY_URL}/view?filename={filename}&subfolder={subfolder}&type={image_type}"
                    return JsonResponse({
                        "status": "completed",
                        "image_url": image_url,
                        "filename": filename
                    })

        # Check queue to see if currently executing or pending
        queue_res = requests.get(f"{COMFY_URL}/queue", timeout=10)
        if queue_res.status_code == 200:
            queue_data = queue_res.json()
            running = queue_data.get("queue_running", [])
            pending = queue_data.get("queue_pending", [])

            # Check if running
            for job in running:
                if len(job) > 1 and job[1] == prompt_id:
                    return JsonResponse({"status": "processing", "stage": "rendering", "message": "Synthesizing pixels on GPU..."})

            # Check if pending
            for idx, job in enumerate(pending):
                if len(job) > 1 and job[1] == prompt_id:
                    return JsonResponse({"status": "processing", "stage": "queued", "message": f"Queued at position #{idx + 1}"})

        return JsonResponse({"status": "processing", "stage": "processing", "message": "Processing generation..."})

    except requests.Timeout:
        return JsonResponse({"status": "processing", "stage": "network", "message": "Checking pod..."})
    except requests.ConnectionError:
        return JsonResponse({"status": "error", "message": "Lost connection to RunPod instance."}, status=502)
    except Exception as e:
        logger.exception("Error checking status")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)