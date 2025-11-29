from django.db import models
from django.contrib.auth import get_user_model
import chess
import json

User = get_user_model()


class Game(models.Model):
    """Chess game model"""
    GAME_STATUS_CHOICES = [
        ('waiting', 'Waiting for opponent'),
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
    ]

    GAME_RESULT_CHOICES = [
        ('white_wins', 'White Wins'),
        ('black_wins', 'Black Wins'),
        ('draw', 'Draw'),
        ('ongoing', 'Ongoing'),
    ]

    GAME_MODE_CHOICES = [
        ('pvp', 'Player vs Player'),
        ('ai', 'Player vs AI'),
    ]

    white_player = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='games_as_white',
        null=True,
        blank=True
    )
    black_player = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='games_as_black',
        null=True,
        blank=True
    )
    
    game_mode = models.CharField(max_length=10, choices=GAME_MODE_CHOICES, default='pvp')
    ai_difficulty = models.CharField(max_length=10, null=True, blank=True)  # easy, medium, hard
    ai_color = models.CharField(max_length=5, null=True, blank=True)  # white or black
    
    board_state = models.TextField(default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')  # FEN notation
    current_turn = models.CharField(max_length=5, default='white')
    status = models.CharField(max_length=20, choices=GAME_STATUS_CHOICES, default='waiting')
    result = models.CharField(max_length=20, choices=GAME_RESULT_CHOICES, default='ongoing')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        if self.game_mode == 'ai':
            player = self.white_player if self.ai_color == 'black' else self.black_player
            return f"Game {self.id}: {player.username} vs AI ({self.ai_difficulty})"
        return f"Game {self.id}: {self.white_player.username if self.white_player else 'Waiting'} vs {self.black_player.username if self.black_player else 'Waiting'}"

    def get_board(self):
        """Get python-chess Board object from FEN"""
        return chess.Board(self.board_state)

    def is_player_turn(self, user):
        """Check if it's the user's turn"""
        if self.game_mode == 'ai':
            if self.ai_color == 'white':
                return self.current_turn == 'black' and user == self.black_player
            else:
                return self.current_turn == 'white' and user == self.white_player
        
        if self.current_turn == 'white':
            return user == self.white_player
        return user == self.black_player

    class Meta:
        ordering = ['-created_at']


class Move(models.Model):
    """Chess move model"""
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='moves')
    player = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    is_ai_move = models.BooleanField(default=False)
    
    move_notation = models.CharField(max_length=10)  # e.g., "e2e4"
    san_notation = models.CharField(max_length=10)  # Standard Algebraic Notation, e.g., "e4"
    fen_after_move = models.TextField()  # Board state after this move
    
    move_number = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Move {self.move_number} in Game {self.game.id}: {self.san_notation}"

    class Meta:
        ordering = ['move_number']
