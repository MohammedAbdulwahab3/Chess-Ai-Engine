from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Game, Move
from .serializers import (
    GameSerializer, 
    GameListSerializer, 
    CreateGameSerializer,
    MakeMoveSerializer,
    MoveSerializer
)
from .chess_logic import ChessLogic


class GameListCreateView(generics.ListCreateAPIView):
    """List all games or create a new game"""
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return CreateGameSerializer
        return GameListSerializer
    
    def get_queryset(self):
        user = self.request.user
        return Game.objects.filter(
            white_player=user
        ) | Game.objects.filter(
            black_player=user
        )
    
    def create(self, request, *args, **kwargs):
        serializer = CreateGameSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        game_mode = serializer.validated_data['game_mode']
        
        if game_mode == 'ai':
            # Create AI game
            ai_difficulty = serializer.validated_data['ai_difficulty']
            ai_color = serializer.validated_data['ai_color']
            
            game = Game.objects.create(
                game_mode='ai',
                ai_difficulty=ai_difficulty,
                ai_color=ai_color,
                status='active'
            )
            
            # Assign player to the opposite color
            if ai_color == 'white':
                game.black_player = request.user
            else:
                game.white_player = request.user
            game.save()
        else:
            # Create PvP game
            game = Game.objects.create(
                white_player=request.user,
                game_mode='pvp',
                status='waiting'
            )
        
        return Response(
            GameSerializer(game).data,
            status=status.HTTP_201_CREATED
        )


class GameDetailView(generics.RetrieveAPIView):
    """Get game details"""
    queryset = Game.objects.all()
    serializer_class = GameSerializer
    permission_classes = [IsAuthenticated]


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def join_game(request, pk):
    """Join a waiting game"""
    game = get_object_or_404(Game, pk=pk)
    
    if game.status != 'waiting':
        return Response(
            {'error': 'Game is not waiting for players'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if game.white_player == request.user:
        return Response(
            {'error': 'You are already in this game'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    game.black_player = request.user
    game.status = 'active'
    game.save()
    
    return Response(GameSerializer(game).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def make_move(request, pk):
    """Make a move in a game"""
    game = get_object_or_404(Game, pk=pk)
    
    if game.status != 'active':
        return Response(
            {'error': 'Game is not active'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Debug logging
    print(f"Received move request data: {request.data}")
    
    serializer = MakeMoveSerializer(data=request.data)
    if not serializer.is_valid():
        print(f"Serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    move_uci = serializer.validated_data['move']
    print(f"Extracted move UCI: {move_uci}")
    
    success, message, move_instance = ChessLogic.make_move(game, request.user, move_uci)
    
    if not success:
        print(f"Move failed: {message}")
        return Response({'error': message}, status=status.HTTP_400_BAD_REQUEST)
    
    return Response({
        'move': MoveSerializer(move_instance).data,
        'game': GameSerializer(game).data,
        'game_status': ChessLogic.get_game_status(game)
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def game_status(request, pk):
    """Get current game status"""
    game = get_object_or_404(Game, pk=pk)
    
    return Response({
        'game': GameSerializer(game).data,
        'status': ChessLogic.get_game_status(game)
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def available_games(request):
    """Get list of games waiting for players"""
    games = Game.objects.filter(status='waiting').exclude(white_player=request.user)
    return Response(GameListSerializer(games, many=True).data)
