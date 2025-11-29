from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from game.models import Game
import os

# Try to import RL engine - fallback if PyTorch not installed
try:
    from .rl_model.chess_network import ChessRLEngine
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'rl_chess_v1.pth')
    # Use the larger model architecture as requested (256 channels, 8 blocks)
    rl_hint_engine = ChessRLEngine(
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        num_channels=256,
        num_res_blocks=8
    )
    RL_MODEL_AVAILABLE = True
except ImportError as e:
    RL_MODEL_AVAILABLE = False
    RL_IMPORT_ERROR = str(e)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_hint(request, game_id):
    """
    Get a move hint/suggestion using YOUR custom RL model!
    
    NOTE: This uses YOUR trained model (not Stockfish)
    - Starts weak initially (random moves)
    - Improves as you train it
    - Great for learning how RL models work!
    
    Stockfish is still used for the AI opponent.
    """
    # Check if RL model is available
    if not RL_MODEL_AVAILABLE:
        return Response({
            'error': 'RL model not available',
            'reason': 'PyTorch not installed in venv',
            'solution': 'Install PyTorch: pip install torch torchvision',
            'import_error': RL_IMPORT_ERROR,
            'note': 'Your RL model requires PyTorch. Install it or wait for current install to complete.'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        game = Game.objects.get(id=game_id)
    except Game.DoesNotExist:
        return Response({'error': 'Game not found'}, status=status.HTTP_404_NOT_FOUND)

    if game.status != 'active':
        return Response({'error': 'Game is not active'}, status=status.HTTP_400_BAD_REQUEST)

    # Get hint from YOUR RL model
    try:
        best_move = rl_hint_engine.get_best_move(game.board_state)
        evaluation = rl_hint_engine.evaluate_position(game.board_state)
    except Exception as e:
        return Response({
            'error': f'RL model error: {str(e)}',
            'hint': 'Train your model first! See training scripts.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    if not best_move:
        return Response({'error': 'Could not generate hint from RL model'}, 
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Parse move to from/to squares
    from_square = best_move[:2]
    to_square = best_move[2:4]
    promotion = best_move[4:] if len(best_move) > 4 else None
    
    # Get model info for debugging
    model_status = "untrained" if not os.path.exists(MODEL_PATH) else "trained"
    
    return Response({
        'hint_move': best_move,
        'from_square': from_square,
        'to_square': to_square,
        'promotion': promotion,
        'evaluation': evaluation,
        'evaluation_text': _get_evaluation_text(evaluation),
        'model_status': model_status,
        'note': f'Hint from YOUR custom RL model ({model_status})'
    })


def _get_evaluation_text(eval_value):
    """Convert evaluation to human-readable text"""
    if eval_value > 300:
        return "White is winning (according to your model)"
    elif eval_value > 100:
        return "White is better (according to your model)"
    elif eval_value > 50:
        return "White is slightly better (according to your model)"
    elif eval_value > -50:
        return "Position is equal (according to your model)"
    elif eval_value > -100:
        return "Black is slightly better (according to your model)"
    elif eval_value > -300:
        return "Black is better (according to your model)"
    else:
        return "Black is winning (according to your model)"

