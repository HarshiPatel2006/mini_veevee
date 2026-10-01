from django.test import TestCase, Client
from django.urls import reverse
from .models import GenerationTask
from .services import get_startup_script, build_image_workflow, build_audio_workflow
from runpod.api.mutations import pods

class GeneratorServiceTests(TestCase):
    def test_startup_script_image_graphql_safe(self):
        script = get_startup_script('IMAGE')
        mutation = pods.generate_pod_deployment_mutation(
            'test', 'runpod/comfyui:latest', gpu_type_id='NVIDIA GeForce RTX 3090', docker_args=script
        )
        self.assertIn('dockerArgs:', mutation)
        # Ensure lens model is included and flux vae
        self.assertIn('lens_bf16.safetensors', script)
        self.assertIn('flux2-vae.safetensors', script)

    def test_startup_script_audio_graphql_safe(self):
        script = get_startup_script('AUDIO')
        mutation = pods.generate_pod_deployment_mutation(
            'test', 'runpod/comfyui:latest', gpu_type_id='NVIDIA GeForce RTX 3090', docker_args=script
        )
        self.assertIn('dockerArgs:', mutation)
        self.assertIn('acestep_v1.5_xl_turbo_bf16.safetensors', script)

    def test_build_image_workflow_path(self):
        workflow = build_image_workflow('a golden retriever')
        unet_name = workflow['1']['inputs']['unet_name']
        self.assertEqual(unet_name, 'lens/lens_bf16.safetensors')
        self.assertEqual(workflow['18']['inputs']['text'], 'a golden retriever')

    def test_build_audio_workflow(self):
        workflow = build_audio_workflow('ambient chill beats')
        self.assertEqual(workflow['6']['inputs']['tags'], 'ambient chill beats')
        self.assertEqual(workflow['3']['inputs']['unet_name'], 'acestep_v1.5_xl_turbo_bf16.safetensors')

class GeneratorViewsTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_view_get(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'VEEVEE UI')

    def test_task_detail_view(self):
        task = GenerationTask.objects.create(workflow_type='IMAGE', prompt='test prompt')
        response = self.client.get(reverse('task_detail', args=[task.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'test prompt')
