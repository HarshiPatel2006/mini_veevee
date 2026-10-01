import os
import json
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

COMFY_URL = "https://dm2wjkhr178irh-8188.proxy.runpod.net"

def index(request):
    return render(request, "generator/index.html")

@csrf_exempt
def generate_image(request):
    if request.method == "POST":
        user_prompt = request.POST.get("prompt", "")
        json_path = os.path.join(settings.BASE_DIR, "workflow_api.json")

        try:
            with open(json_path, "r") as f:
                workflow = json.load(f)

            # Strip UI/Note nodes (nodes missing 'class_type')
            clean_workflow = {
                node_id: node_data
                for node_id, node_data in workflow.items()
                if isinstance(node_data, dict) and node_data.get("class_type") is not None
            }

            # 1. Update Lens Positive Prompt (Node 18)
            if "18" in clean_workflow:
                clean_workflow["18"]["inputs"]["text"] = user_prompt

            # 2. Fix model path slashes for Linux
            if "1" in clean_workflow:
                clean_workflow["1"]["inputs"]["unet_name"] = "lens/lens_bf16.safetensors"

            # 3. Dispatch cleaned payload to ComfyUI
            payload = {"prompt": clean_workflow}
            response = requests.post(f"{COMFY_URL}/prompt", json=payload)
            res_data = response.json()

            prompt_id = res_data.get("prompt_id")

            if not prompt_id:
                return JsonResponse({
                    "status": "error",
                    "message": f"ComfyUI rejected prompt: {res_data}"
                }, status=400)

            return JsonResponse({
                "status": "success",
                "prompt_id": prompt_id
            })

        except FileNotFoundError:
            return JsonResponse({"status": "error", "message": "workflow_api.json not found in project root"}, status=400)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def check_status(request, prompt_id):
    try:
        history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}")
        history_data = history_res.json()

        if prompt_id in history_data:
            outputs = history_data[prompt_id].get("outputs", {})
            
            # Node 11 is SaveImage in Lens workflow
            if "11" in outputs and "images" in outputs["11"]:
                image_info = outputs["11"]["images"][0]
                filename = image_info["filename"]
                subfolder = image_info.get("subfolder", "")
                image_type = image_info.get("type", "output")

                image_url = f"{COMFY_URL}/view?filename={filename}&subfolder={subfolder}&type={image_type}"
                return JsonResponse({"status": "completed", "image_url": image_url})

        return JsonResponse({"status": "processing"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)