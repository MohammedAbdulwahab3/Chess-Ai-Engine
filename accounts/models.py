from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Custom User model with additional fields for chess game"""
    email = models.EmailField(unique=True)
    rating = models.IntegerField(default=1200)  # ELO rating
    games_played = models.IntegerField(default=0)
    games_won = models.IntegerField(default=0)
    games_lost = models.IntegerField(default=0)
    games_drawn = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

    @property
    def win_rate(self):
        """Calculate win rate percentage"""
        if self.games_played == 0:
            return 0
        return (self.games_won / self.games_played) * 100
