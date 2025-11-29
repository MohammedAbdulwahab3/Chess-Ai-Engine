from django.urls import path
from . import views
from .hint_views import get_hint

urlpatterns = [
    path('games/<int:game_id>/ai-move/', views.get_ai_move, name='ai-move'),
    path('games/<int:game_id>/hint/', get_hint, name='get-hint'),
]
