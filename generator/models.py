# generator/models.py
from django.db import models

class GenerationTask(models.Model):
    WORKFLOW_CHOICES = [
        ('IMAGE', 'Image Generation'),
        ('AUDIO', 'Audio / Music Generation'),
    ]

    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROVISIONING', 'Deploying Pod...'),
        ('GENERATING', 'Processing ComfyUI...'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    workflow_type = models.CharField(max_length=10, choices=WORKFLOW_CHOICES, default='IMAGE')
    prompt = models.TextField()
    
    # Input uploads
    input_image = models.ImageField(upload_to='inputs/images/', null=True, blank=True)
    input_audio = models.FileField(upload_to='inputs/audio/', null=True, blank=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    # Generated outputs saved back to DB
    output_file = models.FileField(upload_to='results/', null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_audio(self):
        return self.workflow_type == 'AUDIO' or (self.output_file and self.output_file.name.endswith(('.mp3', '.wav', '.flac')))
