# generator/views.py
from django.shortcuts import render, redirect, get_object_or_404
from .models import GenerationTask
from .services import process_generation_on_runpod

def index(request):
    if request.method == 'POST':
        workflow_type = request.POST.get('workflow_type', 'IMAGE')
        prompt = request.POST.get('prompt')
        image = request.FILES.get('image')
        audio = request.FILES.get('audio')

        task = GenerationTask.objects.create(
            workflow_type=workflow_type,
            prompt=prompt,
            input_image=image,
            input_audio=audio
        )
        
        # Trigger dynamic Pod provision & generation
        process_generation_on_runpod(task.id)
        
        return redirect('task_detail', task_id=task.id)

    tasks = GenerationTask.objects.all().order_by('-created_at')[:5]
    return render(request, 'generator/index.html', {'tasks': tasks})

def task_detail(request, task_id):
    task = get_object_or_404(GenerationTask, pk=task_id)
    return render(request, 'generator/detail.html', {'task': task})