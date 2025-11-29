"""
URL configuration for chess_backend project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('api/', include('game.urls')),
    path('api/ai/', include('ai_engine.urls')),
]
