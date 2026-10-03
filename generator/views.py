# generator/views.py
import json
import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import GenerationTask
from .services import (
    process_generation_on_runpod,
    check_pod_health,
    get_loaded_models,
    get_pod_comfy_url,
    get_active_running_pod,
    terminate_all_active_pods,
    terminate_pod,
    record_heartbeat,
    handle_browser_disconnect,
    ACTIVE_POD_IDS,
    provision_pod,
    PREFERRED_GPUS
)

logger = logging.getLogger(__name__)

PROMPT_STORE = [
    {
        "id": "burger-gourmet",
        "category": "Culinary & Food",
        "title": "Artisanal Wagyu Burger",
        "tags": ["Food", "Macro", "Chiaroscuro"],
        "prompt": "A delicious juicy rustic wagyu burger on charred dark oak wood, glossy golden brioche bun, thick grilled beef patty, melted aged cheddar dripping down the sides, crispy smoked bacon, heirloom tomato slices, caramelized shallots, dramatic moody chiaroscuro lighting, deep shadows, warm specular highlights, rustic artisanal food photography, 85mm f/1.4, ultra-detailed textures, shallow depth of field"
    },
    {
        "id": "apple-spatial",
        "category": "Future Tech & Hardware",
        "title": "Apple Glass Spatial Device",
        "tags": ["Apple", "Minimalist", "Industrial"],
        "prompt": "A futuristic next-generation Apple spatial computing device carved from matte titanium and curved sapphire glass, floating on a minimal slate pedestal, displaying ethereal volumetric macOS interface ribbons, studio product lighting, clean minimalist Apple design language, raytraced reflections, 8k resolution"
    },
    {
        "id": "haute-couture-neon",
        "category": "Editorial Fashion",
        "title": "Cyberpunk Haute Couture",
        "tags": ["Fashion", "Neon", "Rain"],
        "prompt": "Editorial avant-garde fashion photography, model in reflective metallic structured trench coat standing in a glistening rain-slicked neon street, dramatic volumetric teal and magenta backlight, water droplet trails, cinematic bokeh, Hasselblad medium format capture"
    },
    {
        "id": "alpine-villa",
        "category": "Architecture & Spaces",
        "title": "Fjord Cantilever Villa",
        "tags": ["Architecture", "Brutalist", "Atmospheric"],
        "prompt": "Hyper-luxurious modernist cantilever villa anchored into sheer granite cliff over a misty Scandinavian fjord, massive floor-to-ceiling glass panels, warm ambient interior illumination against twilight fog, wet basalt rock textures, architectural digest photography"
    },
    {
        "id": "macro-hummingbird",
        "category": "Wildlife & Nature",
        "title": "Prismatic Hummingbird Dewdrop",
        "tags": ["Macro", "Nature", "Prismatic"],
        "prompt": "Extreme macro close-up of a crystalline morning dewdrop resting on an iridescent hummingbird feather, microscopic water surface tension, rainbow chromatic refraction, razor-sharp focus, velvety dark background, National Geographic award-winning photography"
    },
    {
        "id": "tokyo-hypercar",
        "category": "Automotive",
        "title": "Neo-Tokyo Aero Hypercar",
        "tags": ["Automotive", "Cyberpunk", "Speed"],
        "prompt": "Low-slung carbon-fiber aerodynamic hypercar drifting through neo-Tokyo expressway tunnel, glowing cyan underglow, light trails from holographic billboards reflecting on glossy wet asphalt, motion blur in background, photorealistic unreal engine 5 render"
    },
    {
        "id": "zen-loft",
        "category": "Interior Design",
        "title": "Scandinavian Sunlit Loft",
        "tags": ["Interior", "Serene", "Minimalist"],
        "prompt": "Warm minimalist living sanctuary, double-height ceiling, polished light concrete flooring, custom Scandinavian white oak furniture, large linen drapery diffusing golden morning sunlight, architectural shadows, peaceful serene atmosphere, 35mm interior photography"
    },
    {
        "id": "celestial-islands",
        "category": "Surrealist & 3D Art",
        "title": "Levitating Celestial Archipelago",
        "tags": ["Surreal", "Cosmic", "Octane"],
        "prompt": "Surreal dreamscape of mossy floating islands suspended in a violet cosmic nebula, glowing waterfalls cascading into empty space, ancient bioluminescent willow trees, surrealist fantasy art, ethereal celestial atmosphere, octane render, 8k"
    },
    {
        "id": "kyoto-rain",
        "category": "Cinematic Street",
        "title": "Kyoto Lantern Dusk",
        "tags": ["Cinematic", "Kyoto", "35mm Film"],
        "prompt": "Atmospheric candid street photography in Gion Kyoto at dusk during autumn rain, wet cobblestone reflections, glowing amber paper lanterns casting soft glow, silhouette of figure under umbrella, rich film grain, cinematic Kodak Portra 400 aesthetic"
    },
    {
        "id": "tourbillon-watch",
        "category": "Luxury Product",
        "title": "Titanium Skeleton Tourbillon",
        "tags": ["Horology", "Luxury", "Macro"],
        "prompt": "Precision-milled titanium luxury skeleton watch resting on black volcanic sand, intricate exposed mechanical tourbillon gears, anti-reflective domed sapphire crystal, dramatic rim lighting, macro horology photography"
    },
    {
        "id": "cosmic-nursery",
        "category": "Cosmic & Astro",
        "title": "James Webb Deep Nebula",
        "tags": ["Space", "Astrophotography", "JWST"],
        "prompt": "Ultra-deep space James Webb space telescope view of a vibrant stellar nursery, swirling cosmic dust pillars illuminated in emerald and crimson by newborn stars, gravitational light lensing, immense scale, scientific astronomical fidelity"
    },
    {
        "id": "moss-dragon",
        "category": "Mythical Fantasy",
        "title": "Slumbering Forest Wyrm",
        "tags": ["Fantasy", "Creature", "Mythical"],
        "prompt": "Ancient mythical moss-clad dragon slumbering curled inside a hollow redwood cathedral, glowing cyan spores drifting through shafts of morning sunbeams, hyper-detailed scales, enchanted atmosphere, fantasy concept art"
    },
    {
        "id": "rembrandt-sea",
        "category": "Cinematic Portrait",
        "title": "The Old Mariner",
        "tags": ["Portrait", "Rembrandt", "Cinematic"],
        "prompt": "Close-up portrait of an elderly sea captain with weathered skin and silver beard, intense contemplative gaze, dramatic Rembrandt single-source side lighting, deep textural shadows, cinematic 70mm anamorphic lens"
    },
    {
        "id": "mercury-ribbon",
        "category": "Abstract 3D",
        "title": "Prismatic Mercury Ribbon",
        "tags": ["Abstract", "Fluid", "Prismatic"],
        "prompt": "Abstract dynamic sculpture of liquid mercury and iridescent chrome ribbon twisting through weightless void, prismatic spectral reflections, high-speed fluid dynamics, sleek minimalist composition, 3D digital art"
    },
    {
        "id": "sky-metropolis",
        "category": "Sci-Fi Landscapes",
        "title": "Dystopian Sky Metropolis",
        "tags": ["Sci-Fi", "Cityscape", "Atmospheric"],
        "prompt": "Panoramic wide shot of a dense futuristic megalopolis shrouded in low-hanging smog and neon glow, multi-tiered skyways with autonomous transit pods, colossal architectural spires reaching through cloud decks at dusk"
    },
    {
        "id": "roadster-67",
        "category": "Vintage Automotive",
        "title": "1967 Big Sur Roadster",
        "tags": ["Vintage", "Roadster", "Golden Hour"],
        "prompt": "Classic cherry-red 1967 convertible roadster parked on Big Sur coastal cliffside during golden sunset, polished chrome bumpers gleaming, gentle ocean breeze, pastel gradient sky, 1970s vintage 35mm film photography"
    },
    {
        "id": "michelin-dessert",
        "category": "Culinary & Food",
        "title": "Deconstructed Matcha Zen",
        "tags": ["Gastronomy", "Fine Dining", "Studio"],
        "prompt": "Michelin-starred deconstructed black sesame and matcha dessert on handmade charcoal ceramic plate, spun sugar sphere, edible gold dust accents, precision culinary presentation, crisp studio food photography"
    },
    {
        "id": "snow-leopard",
        "category": "Wildlife & Nature",
        "title": "Himalayan Snow Leopard",
        "tags": ["Wildlife", "Snow", "Portrait"],
        "prompt": "Majestic snow leopard perched on jagged ice crag amidst a fierce mountain blizzard, intense piercing green eyes, frosted fur covered in snowflakes, photorealistic wildlife documentary photography"
    },
    {
        "id": "biophilic-dome",
        "category": "Architecture & Spaces",
        "title": "Biophilic Botanical Dome",
        "tags": ["Eco Architecture", "Timber", "Greenery"],
        "prompt": "Futuristic organic architectural pavilion built from curved glued-laminated timber, lush cascading vertical gardens, solar skylights casting geometric dappled light on visitors, sustainable futuristic design"
    },
    {
        "id": "android-repose",
        "category": "Future Tech & Hardware",
        "title": "Android in Apple Gallery",
        "tags": ["Android", "Porcelain", "Minimalist"],
        "prompt": "Elegant humanoid android with polished porcelain white shell and discreet copper seams, seated in contemplative silence in an Apple-inspired minimalist white gallery, soft diffused natural daylight, contemplative high-tech realism"
    },
    {
        "id": "cyber-market",
        "category": "Cinematic Street",
        "title": "Midnight Noodle Alley",
        "tags": ["Cyberpunk", "Street", "Neon"],
        "prompt": "Bustling futuristic night market in a dense Asian megacity, steaming noodle stalls under holographic neon signs, rain umbrellas, wet pavement reflecting crimson and blue neon, cinematic depth and rich detail"
    },
    {
        "id": "diamond-beach",
        "category": "Wildlife & Nature",
        "title": "Icelandic Ice Crystals",
        "tags": ["Landscape", "Iceland", "Fine Art"],
        "prompt": "Diamonds of glacial ice glistening on volcanic black sand beach at Diamond Beach Iceland, dramatic overcast twilight, crashing turquoise ocean waves with sea mist, long exposure fine-art landscape"
    }
]


def index(request):
    """Render the Apple Pro Studio interface."""
    recent_tasks = GenerationTask.objects.all().order_by('-created_at')[:10]
    return render(request, 'generator/index.html', {
        'recent_tasks': recent_tasks,
        'prompts': PROMPT_STORE
    })


def api_prompts(request):
    """Return prompt library stores for UI."""
    return JsonResponse({"prompts": PROMPT_STORE})


def api_system_status(request):
    """Check live RunPod pod telemetry, loaded models, and VRAM stats."""
    pod_id, pod_url = get_active_running_pod()
    if not pod_url:
        pod_url = get_pod_comfy_url()
    healthy, stats = check_pod_health(pod_url)

    models = get_loaded_models(pod_url) if healthy else {"unet": [], "clip": [], "vae": []}
    has_unet = any("lens" in str(m).lower() for m in models.get("unet", []))
    has_clip = any("gpt_oss" in str(m).lower() or "lens" in str(m).lower() for m in models.get("clip", []))
    has_vae = any("flux" in str(m).lower() for m in models.get("vae", []))

    return JsonResponse({
        "connected": healthy,
        "pod_id": pod_id,
        "pod_url": pod_url if healthy else None,
        "gpu_name": stats.get("gpu_name", "RunPod Cloud GPU") if healthy else "Cloud GPU Offline",
        "vram_total_gb": stats.get("vram_total", 0) if healthy else 0,
        "vram_free_gb": stats.get("vram_free", 0) if healthy else 0,
        "models_ready": has_unet and has_clip and has_vae,
        "active_pods_count": len(ACTIVE_POD_IDS) if (healthy or len(ACTIVE_POD_IDS) > 0) else (1 if pod_id else 0),
        "available_models": models
    })


@csrf_exempt
def api_generate(request):
    """Handle generation requests with full parameter support."""
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "POST method required"}, status=405)

    try:
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({"status": "error", "message": "Please provide a text prompt"}, status=400)

        negative_prompt = request.POST.get("negative_prompt", "").strip()
        aspect_ratio = request.POST.get("aspect_ratio", "1:1 (Square)")
        
        seed_str = request.POST.get("seed", "").strip()
        seed = int(seed_str) if seed_str.isdigit() else None
        
        steps = int(request.POST.get("steps", 20))
        cfg = float(request.POST.get("cfg", 5.0))
        auto_terminate = request.POST.get("auto_terminate", "false").lower() in ["true", "1", "yes"]

        task = GenerationTask.objects.create(
            workflow_type='IMAGE',
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            steps=steps,
            cfg=cfg,
            status='PENDING',
            status_detail='Queued for execution...'
        )

        process_generation_on_runpod(task.id, auto_terminate=auto_terminate)

        return JsonResponse({
            "status": "ok",
            "task_id": task.id,
            "message": "Generation initiated successfully."
        })
    except Exception as e:
        logger.error(f"Error in api_generate: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def api_task_status(request, task_id):
    """Poll generation progress for a specific task."""
    task = get_object_or_404(GenerationTask, pk=task_id)
    return JsonResponse({
        "id": task.id,
        "status": task.status,
        "progress_percent": task.progress_percent,
        "status_detail": task.status_detail,
        "error_message": task.error_message,
        "prompt": task.prompt,
        "aspect_ratio": task.aspect_ratio,
        "seed": task.seed,
        "output_url": task.output_file.url if task.output_file else None,
        "created_at": task.created_at.strftime("%H:%M:%S")
    })


@csrf_exempt
def api_heartbeat(request):
    """Keepalive ping from open browser tab."""
    record_heartbeat()
    return JsonResponse({"status": "ok"})


@csrf_exempt
def api_terminate_on_close(request):
    """Called by browser beacon when window or tab closes."""
    handle_browser_disconnect()
    return JsonResponse({"status": "terminated"})


@csrf_exempt
def api_terminate_pod(request):
    """Manual 1-click pod termination."""
    terminate_all_active_pods()
    return JsonResponse({
        "status": "ok",
        "message": "All active RunPod pods have been terminated."
    })


@csrf_exempt
def api_provision_pod(request):
    """Manual trigger to spin up a pod with RTX 4000 Ada / RTX 3090 / L4 (or reuse actively running pod)."""
    try:
        active_id, active_url = get_active_running_pod(force_refresh=True)
        if active_id and active_url:
            return JsonResponse({
                "status": "ok",
                "pod_id": active_id,
                "pod_url": active_url,
                "message": f"Active GPU pod {active_id} is already running and ready."
            })
        pod_id, pod_url = provision_pod(PREFERRED_GPUS)
        return JsonResponse({
            "status": "ok",
            "pod_id": pod_id,
            "pod_url": pod_url,
            "message": f"Successfully deployed GPU pod {pod_id}"
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def task_detail(request, task_id):
    """Detail view for single job."""
    task = get_object_or_404(GenerationTask, pk=task_id)
    return render(request, 'generator/detail.html', {'task': task})