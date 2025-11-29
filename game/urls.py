from django.urls import path
from . import views

urlpatterns = [
    path('games/', views.GameListCreateView.as_view(), name='game-list-create'),
    path('games/<int:pk>/', views.GameDetailView.as_view(), name='game-detail'),
    path('games/<int:pk>/join/', views.join_game, name='game-join'),
    path('games/<int:pk>/move/', views.make_move, name='game-move'),
    path('games/<int:pk>/status/', views.game_status, name='game-status'),
    path('games/available/', views.available_games, name='available-games'),
]
