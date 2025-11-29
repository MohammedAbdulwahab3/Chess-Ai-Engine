from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from game.models import Game, Move
from game.chess_logic import ChessLogic
from .stockfish_engine import StockfishEngine
import chess


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_ai_move(request, game_id):
    """
    Get AI move for a game
    """
    try:
        game = Game.objects.get(id=game_id)
    except Game.DoesNotExist:
        return Response({'error': 'Game not found'}, status=status.HTTP_404_NOT_FOUND)

    # Verify it's an AI game
    if game.game_mode != 'ai':
        return Response({'error': 'Not an AI game'}, status=status.HTTP_400_BAD_REQUEST)

    # Verify it's AI's turn
    is_ai_turn = (game.current_turn == 'white' and game.ai_color == 'white') or \
                 (game.current_turn == 'black' and game.ai_color == 'black')
    
    if not is_ai_turn:
        return Response({'error': 'Not AI turn'}, status=status.HTTP_400_BAD_REQUEST)

    # Get AI move
    engine = StockfishEngine(difficulty=game.ai_difficulty or 'medium')
    best_move = engine.get_best_move(game.board_state)

    if not best_move:
        return Response({'error': 'Could not generate AI move'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Make the move
    board = game.get_board()
    move = chess.Move.from_uci(best_move)
    san_notation = board.san(move)
    board.push(move)

    # Create move record
    move_number = game.moves.count() + 1
    move_instance = Move.objects.create(
        game=game,
        player=None,
        is_ai_move=True,
        move_notation=best_move,
        san_notation=san_notation,
        fen_after_move=board.fen(),
        move_number=move_number
    )

    # Update game state
    game.board_state = board.fen()
    game.current_turn = 'black' if game.current_turn == 'white' else 'white'

    # Check for game end conditions
    if board.is_checkmate():
        game.status = 'completed'
        winner = 'white' if board.turn == chess.BLACK else 'black'
        game.result = 'white_wins' if winner == 'white' else 'black_wins'
        ChessLogic._update_player_stats(game, winner)
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        game.status = 'completed'
        game.result = 'draw'
        ChessLogic._update_player_stats(game, 'draw')

    game.save()

    return Response({
        'move': best_move,
        'san': san_notation,
        'board_state': game.board_state,
        'game_status': game.status,
        'result': game.result
    })
