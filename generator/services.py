# generator/services.py
import os
import time
import requests
import threading
import runpod
from django.core.files.base import ContentFile



import os

# REPLACE THIS:
# RUNPOD_API_KEY = "rnp_xxxx_your_actual_key"

# WITH THIS:
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# generator/services.py


# generator/services.py

# generator/services.py

def get_startup_script(workflow_type='IMAGE'):
    """
    Bash script executed by RunPod.
    - Completely avoids nested quotes to prevent RunPod SDK GraphQL crash.
    - Downloads only required models for the specified workflow.
    - Ends with '/start.sh' to ensure ComfyUI launches after downloading.
    """
    base = "/workspace/runpod-slim/ComfyUI/models"
    dirs = [
        f"mkdir -p {base}/unet/lens {base}/text_encoders {base}/clip {base}/vae",
        "[ -d /workspace/ComfyUI ] && [ ! -d /workspace/runpod-slim/ComfyUI ] && ln -sf /workspace/ComfyUI /workspace/runpod-slim/ComfyUI || true",
        "[ -d /workspace/runpod-slim/ComfyUI ] && [ ! -d /workspace/ComfyUI ] && ln -sf /workspace/runpod-slim/ComfyUI /workspace/ComfyUI || true",
    ]

    downloads = []
    if workflow_type == 'AUDIO':
        downloads = [
            f"[ -f {base}/unet/acestep_v1.5_xl_turbo_bf16.safetensors ] || wget -nc -q -O {base}/unet/acestep_v1.5_xl_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/AceStep_v1.5_XL_Turbo/resolve/main/acestep_v1.5_xl_turbo_bf16.safetensors",
            f"[ -f {base}/text_encoders/qwen_0.6b_ace15.safetensors ] || wget -nc -q -O {base}/text_encoders/qwen_0.6b_ace15.safetensors https://huggingface.co/Comfy-Org/AceStep_v1.5_XL_Turbo/resolve/main/qwen_0.6b_ace15.safetensors",
            f"[ -f {base}/text_encoders/qwen_4b_ace15.safetensors ] || wget -nc -q -O {base}/text_encoders/qwen_4b_ace15.safetensors https://huggingface.co/Comfy-Org/AceStep_v1.5_XL_Turbo/resolve/main/qwen_4b_ace15.safetensors",
            f"[ -f {base}/vae/ace_1.5_vae.safetensors ] || wget -nc -q -O {base}/vae/ace_1.5_vae.safetensors https://huggingface.co/Comfy-Org/AceStep_v1.5_XL_Turbo/resolve/main/ace_1.5_vae.safetensors",
        ]
    else:
        downloads = [
            f"[ -f {base}/unet/lens/lens_bf16.safetensors ] || wget -nc -q -O {base}/unet/lens/lens_bf16.safetensors https://huggingface.co/Comfy-Org/Lens/resolve/main/lens_bf16.safetensors",
            f"[ -f {base}/text_encoders/gpt_oss_20b_nvfp4.safetensors ] || wget -nc -q -O {base}/text_encoders/gpt_oss_20b_nvfp4.safetensors https://huggingface.co/Comfy-Org/Lens/resolve/main/gpt_oss_20b_nvfp4.safetensors",
            f"[ -f {base}/vae/flux2-vae.safetensors ] || wget -nc -q -O {base}/vae/flux2-vae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors",
        ]

    post_setup = [
        f"ln -sf {base}/text_encoders/* {base}/clip/ 2>/dev/null || true",
        "/start.sh"
    ]

    all_cmds = " && ".join(dirs + downloads + post_setup)
    script = f"bash -c '{all_cmds}'"
    return script.replace('"', '\\"')
    
def build_image_workflow(prompt):
    """
    Image Workflow (Lens txt2img)
    Injected user prompt directly into Node "18" (Positive CLIP Text Encode)
    """
    return {
      "1": {
        "inputs": {
          "unet_name": "lens/lens_bf16.safetensors",
          "weight_dtype": "default"
        },
        "class_type": "UNETLoader"
      },
      "2": {
        "inputs": {"sampler_name": "euler"},
        "class_type": "KSamplerSelect"
      },
      "3": {
        "inputs": {
          "scheduler": "simple",
          "steps": 20,
          "denoise": 1,
          "model": ["16", 0]
        },
        "class_type": "BasicScheduler"
      },
      "4": {
        "inputs": {
          "expression": "a & -8",
          "values.a": ["6", 1]
        },
        "class_type": "ComfyMathExpression"
      },
      "5": {
        "inputs": {
          "width": ["12", 1],
          "height": ["4", 1],
          "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
      },
      "6": {
        "inputs": {
          "aspect_ratio": "1:1 (Square)",
          "megapixels": 2,
          "multiple": 8
        },
        "class_type": "ResolutionSelector"
      },
      "7": {
        "inputs": {
          "add_noise": True,
          "noise_seed": 937888105118440,
          "cfg": 5,
          "model": ["15", 0],
          "positive": ["18", 0],
          "negative": ["17", 0],
          "sampler": ["2", 0],
          "sigmas": ["3", 0],
          "latent_image": ["5", 0]
        },
        "class_type": "SamplerCustom"
      },
      "8": {
        "inputs": {
          "samples": ["7", 0],
          "vae": ["10", 0]
        },
        "class_type": "VAEDecode"
      },
      "9": {
        "inputs": {
          "clip_name": "gpt_oss_20b_nvfp4.safetensors",
          "type": "lens",
          "device": "default"
        },
        "class_type": "CLIPLoader"
      },
      "10": {
        "inputs": {"vae_name": "flux2-vae.safetensors"},
        "class_type": "VAELoader"
      },
      "11": {
        "inputs": {
          "filename_prefix": "img",
          "images": ["8", 0]
        },
        "class_type": "SaveImage"
      },
      "12": {
        "inputs": {
          "expression": "a & -8",
          "values.a": ["6", 0]
        },
        "class_type": "ComfyMathExpression"
      },
      "15": {
        "inputs": {
          "strength": 1,
          "pre_cfg": False,
          "model": ["16", 0]
        },
        "class_type": "CFGNorm"
      },
      "16": {
        "inputs": {
          "max_shift": 1.15,
          "base_shift": 0.5,
          "width": ["12", 1],
          "height": ["4", 1],
          "model": ["1", 0]
        },
        "class_type": "ModelSamplingFlux"
      },
      "17": {
        "inputs": {
          "text": "",
          "clip": ["9", 0]
        },
        "class_type": "CLIPTextEncode"
      },
      "18": {
        "inputs": {
          "text": prompt,  # <--- INJECTED DYNAMIC USER PROMPT HERE
          "clip": ["9", 0]
        },
        "class_type": "CLIPTextEncode"
      }
    }


def build_audio_workflow(prompt):
    """
    Music Workflow (Acestep v1.5 xl Turbo Text to Music)
    Injected user prompt directly into Node "6" (TextEncodeAceStepAudio1.5 tags)
    """
    return {
      "3": {
        "inputs": {
          "unet_name": "acestep_v1.5_xl_turbo_bf16.safetensors",
          "weight_dtype": "default"
        },
        "class_type": "UNETLoader"
      },
      "4": {
        "inputs": {
          "clip_name1": "qwen_0.6b_ace15.safetensors",
          "clip_name2": "qwen_4b_ace15.safetensors",
          "type": "ace",
          "device": "default"
        },
        "class_type": "DualCLIPLoader"
      },
      "5": {
        "inputs": {"vae_name": "ace_1.5_vae.safetensors"},
        "class_type": "VAELoader"
      },
      "6": {
        "inputs": {
          "tags": prompt,  # <--- INJECTED DYNAMIC USER PROMPT HERE
          "lyrics": "",
          "seed": ["14", 0],
          "bpm": ["15", 0],
          "duration": ["11", 0],
          "timesignature": "4",
          "language": "en",
          "keyscale": "E minor",
          "generate_audio_codes": True,
          "cfg_scale": 2,
          "temperature": 0.85,
          "top_p": 0.9,
          "top_k": 0,
          "min_p": 0,
          "clip": ["4", 0]
        },
        "class_type": "TextEncodeAceStepAudio1.5"
      },
      "7": {
        "inputs": {
          "seconds": ["11", 0],
          "batch_size": 1
        },
        "class_type": "EmptyAceStep1.5LatentAudio"
      },
      "8": {
        "inputs": {
          "shift": 3,
          "sampling": "flow",
          "model": ["3", 0]
        },
        "class_type": "ModelSamplingAuraFlow"
      },
      "9": {
        "inputs": {
          "seed": ["14", 0],
          "steps": 8,
          "cfg": 1,
          "sampler_name": "euler",
          "scheduler": "simple",
          "denoise": 1,
          "model": ["8", 0],
          "positive": ["6", 0],
          "negative": ["10", 0],
          "latent_image": ["7", 0]
        },
        "class_type": "KSampler"
      },
      "10": {
        "inputs": {"conditioning": ["6", 0]},
        "class_type": "ConditioningZeroOut"
      },
      "11": {
        "inputs": {"value": 20},
        "class_type": "PrimitiveFloat"
      },
      "12": {
        "inputs": {
          "samples": ["9", 0],
          "vae": ["5", 0]
        },
        "class_type": "VAEDecodeAudio"
      },
      "13": {
        "inputs": {
          "filename_prefix": "audio/acestep",
          "quality": "V0",
          "audio": ["12", 0]
        },
        "class_type": "SaveAudioMP3"  # Output audio node
      },
      "14": {
        "inputs": {"value": 500},
        "class_type": "PrimitiveInt"
      },
      "15": {
        "inputs": {"value": 116},
        "class_type": "PrimitiveInt"
      }
    }
# generator/services.py

def process_generation_on_runpod(task_id):
    def worker():
        from .models import GenerationTask
        task = GenerationTask.objects.get(id=task_id)
        pod_id = None

        try:
            task.status = 'PROVISIONING'
            task.save()

            # 1. Provision Pod with candidate GPUs in case of capacity limitations
            gpu_candidates = [
                "NVIDIA GeForce RTX 3090",
                "NVIDIA GeForce RTX 4090",
                "NVIDIA RTX A5000",
                "NVIDIA RTX A6000",
                "NVIDIA GeForce RTX 3080",
            ]
            pod = None
            last_err = None
            startup_script = get_startup_script(task.workflow_type)

            for gpu_id in gpu_candidates:
                try:
                    pod = runpod.create_pod(
                        name=f"veevee-job-{task.id}",
                        image_name="runpod/comfyui:latest",
                        gpu_type_id=gpu_id,
                        ports="8188/http",
                        container_disk_in_gb=100,
                        volume_in_gb=0,
                        docker_args=startup_script
                    )
                    if pod and "id" in pod:
                        break
                except Exception as e:
                    last_err = e
                    if "no longer any instances available" in str(e).lower():
                        continue
                    raise e

            if not pod or "id" not in pod:
                raise Exception(f"Unable to provision GPU pod: {last_err}")

            pod_id = pod["id"]
            print(f"[RunPod] Pod Created with 100GB Disk: {pod_id}")

            # 2. Poll until Pod HTTP proxy is reachable AND models are fully downloaded
            pod_url = f"https://{pod_id}-8188.proxy.runpod.net"
            active = False
            target_model = "acestep_v1.5_xl_turbo_bf16.safetensors" if task.workflow_type == 'AUDIO' else "lens/lens_bf16.safetensors"
            
            for attempt in range(120): # Up to 10 minutes waiting for 20GB downloads
                time.sleep(5)
                try:
                    # Ask ComfyUI to refresh its internal model folder list
                    try:
                        requests.post(f"{pod_url}/extra_model_paths", timeout=5)
                    except Exception:
                        pass
                    
                    # Check object_info to confirm loaders see the downloaded models
                    info = requests.get(f"{pod_url}/object_info", timeout=5).json()
                    unet_models = info.get("UNETLoader", {}).get("input", {}).get("required", {}).get("unet_name", [[]])[0]
                    
                    if target_model in unet_models or any(target_model.split('/')[-1] in str(m) for m in unet_models):
                        print(f"[RunPod] Model {target_model} loaded into ComfyUI successfully!")
                        active = True
                        break
                    else:
                        print(f"[RunPod] Waiting for background model downloads... ({attempt+1}/120)")
                except Exception:
                    continue

            if not active:
                raise Exception("RunPod engine timed out while downloading models or starting ComfyUI.")

            # 3. Proceed with workflow generation
            task.status = 'GENERATING'
            task.save()

            if task.workflow_type == 'AUDIO':
                workflow = build_audio_workflow(task.prompt)
                ext = "mp3"
            else:
                workflow = build_image_workflow(task.prompt)
                ext = "png"

            # Post workflow to ComfyUI
            response = requests.post(f"{pod_url}/prompt", json={"prompt": workflow}, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"ComfyUI rejected prompt payload: {response.text}")
                
            prompt_res = response.json()
            prompt_id = prompt_res.get("prompt_id")

            # 4. Wait for execution completion
            history_url = f"{pod_url}/history/{prompt_id}"
            output_filename = None
            subfolder = ""
            output_type = "output"
            
            for _ in range(120): # Up to 6 minutes polling for high-res generation
                time.sleep(3)
                try:
                    hist_res = requests.get(history_url, timeout=5).json()
                    if prompt_id in hist_res:
                        outputs = hist_res[prompt_id].get("outputs", {})
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                output_filename = node_output["images"][0]["filename"]
                                subfolder = node_output["images"][0].get("subfolder", "")
                                output_type = node_output["images"][0].get("type", "output")
                            elif "audio" in node_output:
                                output_filename = node_output["audio"][0]["filename"]
                                subfolder = node_output["audio"][0].get("subfolder", "")
                                output_type = node_output["audio"][0].get("type", "output")
                        break
                except Exception:
                    continue

            if not output_filename:
                raise Exception("Generation finished or failed, but no media output file was produced.")

            # 5. Download output and store in DB
            view_url = f"{pod_url}/view?filename={output_filename}&subfolder={subfolder}&type={output_type}"
            file_data = requests.get(view_url).content

            saved_filename = f"task_{task.id}_result.{ext}"
            task.output_file.save(saved_filename, ContentFile(file_data))
            task.status = 'COMPLETED'
            task.save()

        except Exception as e:
            print(f"[RunPod Worker Error] {str(e)}")
            task.status = 'FAILED'
            task.error_message = str(e)
            task.save()

        finally:
            # Always terminate the pod immediately to save money
            if pod_id:
                try:
                    runpod.terminate_pod(pod_id)
                    print(f"[RunPod] Pod {pod_id} shut down successfully.")
                except Exception as cleanup_err:
                    print(f"[RunPod Cleanup Error] {cleanup_err}")

    threading.Thread(target=worker).start()