from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
from .models import GenerationTask
from .services import (
    build_image_workflow,
    PREFERRED_GPUS,
    REQUIRED_MODELS,
    get_active_running_pod,
    ACTIVE_POD_IDS,
)

class GeneratorServiceTests(TestCase):
    def tearDown(self):
        ACTIVE_POD_IDS.clear()

    def test_preferred_gpus_configuration(self):
        """Verify requested GPUs are configured (RTX PRO 4000, RTX 3090, L4)."""
        self.assertIn("NVIDIA RTX PRO 4000 Blackwell", PREFERRED_GPUS)
        self.assertIn("NVIDIA RTX 4000 Ada Generation", PREFERRED_GPUS)
        self.assertIn("NVIDIA GeForce RTX 3090", PREFERRED_GPUS)
        self.assertIn("NVIDIA L4", PREFERRED_GPUS)

    def test_required_models_manifest(self):
        """Verify lens unet, gpt-oss clip, and flux2-vae are included."""
        filenames = [m["filename"] for m in REQUIRED_MODELS]
        self.assertIn("lens_bf16.safetensors", filenames)
        self.assertIn("gpt_oss_20b_nvfp4.safetensors", filenames)
        self.assertIn("flux2-vae.safetensors", filenames)

    def test_build_image_workflow_sanitization(self):
        """Ensure workflow strips unhandled note nodes without class_type and injects prompt."""
        workflow, seed = build_image_workflow("a luxury titanium camera", aspect_ratio="16:9 (Landscape)")
        self.assertNotIn("13", workflow)  # note node stripped
        self.assertNotIn("14", workflow)  # note node stripped
        self.assertEqual(workflow["18"]["inputs"]["text"], "a luxury titanium camera")
        self.assertEqual(workflow["6"]["inputs"]["aspect_ratio"], "16:9 (Landscape)")
        self.assertIsNotNone(seed)

    @patch('generator.services.get_runpod_client')
    @patch('generator.services.check_pod_health')
    def test_get_active_running_pod_when_pod_running(self, mock_health, mock_get_client):
        """Verify active running pod is selected and not creating a new one."""
        ACTIVE_POD_IDS.clear()
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_pods.return_value = [
            {'id': 'pod-active-123', 'desiredStatus': 'RUNNING', 'runtime': {'ports': []}, 'name': 'veevee-pro-studio'}
        ]
        mock_health.return_value = (True, {'gpu_name': 'NVIDIA RTX 4000 Ada', 'vram_total': 24, 'vram_free': 20})

        pod_id, pod_url = get_active_running_pod(force_refresh=True)
        self.assertEqual(pod_id, 'pod-active-123')
        self.assertIn('pod-active-123', pod_url)

    @patch('generator.services.get_runpod_client')
    @patch('generator.services.check_pod_health')
    def test_get_active_running_pod_when_no_pod_running(self, mock_health, mock_get_client):
        """Verify None returned when no pod is running."""
        ACTIVE_POD_IDS.clear()
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_pods.return_value = [
            {'id': 'pod-old-999', 'desiredStatus': 'EXITED', 'runtime': None}
        ]
        mock_health.return_value = (False, 'Offline')

        pod_id, pod_url = get_active_running_pod(force_refresh=True)
        self.assertIsNone(pod_id)
        self.assertIsNone(pod_url)


class GeneratorViewsTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_view_get(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'VEEVEE Studio')

    def test_system_status_api(self):
        response = self.client.get(reverse('api_system_status'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('connected', data)
        self.assertIn('gpu_name', data)

    def test_prompts_store_api(self):
        response = self.client.get(reverse('api_prompts'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('prompts', data)
        self.assertGreaterEqual(len(data['prompts']), 20)

    def test_task_detail_view(self):
        task = GenerationTask.objects.create(workflow_type='IMAGE', prompt='test prompt')
        response = self.client.get(reverse('task_detail', args=[task.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'test prompt')

    def test_heartbeat_api(self):
        response = self.client.post(reverse('api_heartbeat'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch('generator.views.get_active_running_pod')
    def test_api_provision_reuses_active_pod(self, mock_active):
        """Verify manual provision API returns running pod instead of double-provisioning."""
        mock_active.return_value = ('pod-active-123', 'https://pod-active-123-8188.proxy.runpod.net')
        response = self.client.post(reverse('api_provision_pod'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['pod_id'], 'pod-active-123')
        self.assertIn('already running', data['message'])

