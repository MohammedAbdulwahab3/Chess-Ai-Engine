from rest_framework import serializers
from .models import Game, Move
from accounts.serializers import UserSerializer


class MoveSerializer(serializers.ModelSerializer):
    """Serializer for Move model"""
    player = UserSerializer(read_only=True)
    
    class Meta:
        model = Move
        fields = ['id', 'player', 'is_ai_move', 'move_notation', 'san_notation', 
                  'fen_after_move', 'move_number', 'timestamp']
        read_only_fields = ['id', 'timestamp']


class GameSerializer(serializers.ModelSerializer):
    """Serializer for Game model"""
    white_player = UserSerializer(read_only=True)
    black_player = UserSerializer(read_only=True)
    moves = MoveSerializer(many=True, read_only=True)
    
    class Meta:
        model = Game
        fields = ['id', 'white_player', 'black_player', 'game_mode', 'ai_difficulty', 
                  'ai_color', 'board_state', 'current_turn', 'status', 'result', 
                  'created_at', 'updated_at', 'completed_at', 'moves']
        read_only_fields = ['id', 'board_state', 'current_turn', 'status', 'result', 
                           'created_at', 'updated_at', 'completed_at']


class GameListSerializer(serializers.ModelSerializer):
    """Simplified serializer for game lists"""
    white_player = UserSerializer(read_only=True)
    black_player = UserSerializer(read_only=True)
    move_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Game
        fields = ['id', 'white_player', 'black_player', 'game_mode', 'ai_difficulty',
                  'status', 'result', 'created_at', 'move_count']
    
    def get_move_count(self, obj):
        return obj.moves.count()


class CreateGameSerializer(serializers.Serializer):
    """Serializer for creating a new game"""
    game_mode = serializers.ChoiceField(choices=['pvp', 'ai'])
    ai_difficulty = serializers.ChoiceField(
        choices=['easy', 'medium', 'hard'], 
        required=False, 
        allow_null=True
    )
    ai_color = serializers.ChoiceField(
        choices=['white', 'black'], 
        required=False, 
        allow_null=True
    )
    
    def validate(self, data):
        if data['game_mode'] == 'ai':
            if not data.get('ai_difficulty'):
                raise serializers.ValidationError("AI difficulty is required for AI games")
            if not data.get('ai_color'):
                raise serializers.ValidationError("AI color is required for AI games")
        return data


class MakeMoveSerializer(serializers.Serializer):
    """Serializer for making a move"""
    move = serializers.CharField(max_length=10, help_text="Move in UCI format (e.g., 'e2e4')")
