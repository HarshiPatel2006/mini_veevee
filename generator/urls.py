# generator/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/generate/', views.api_generate, name='api_generate'),
    path('api/status/<int:task_id>/', views.api_task_status, name='api_task_status'),
    path('api/system/status/', views.api_system_status, name='api_system_status'),
    path('api/prompts/', views.api_prompts, name='api_prompts'),
    path('api/pod/terminate/', views.api_terminate_pod, name='api_terminate_pod'),
    path('api/pod/provision/', views.api_provision_pod, name='api_provision_pod'),
    path('api/pod/heartbeat/', views.api_heartbeat, name='api_heartbeat'),
    path('api/pod/terminate_on_close/', views.api_terminate_on_close, name='api_terminate_on_close'),
    path('task/<int:task_id>/', views.task_detail, name='task_detail'),
]