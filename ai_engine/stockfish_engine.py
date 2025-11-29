import chess
from stockfish import Stockfish
from django.conf import settings
import os


class StockfishEngine:
    """Wrapper for Stockfish chess engine"""
    
    DIFFICULTY_SETTINGS = {
        'easy': {
            'depth': 5,
            'skill_level': 5,
        },
        'medium': {
            'depth': 10,
            'skill_level': 10,
        },
        'hard': {
            'depth': 15,
            'skill_level': 20,
        }
    }

    def __init__(self, difficulty='medium'):
        """
        Initialize Stockfish engine
        Args:
            difficulty: 'easy', 'medium', or 'hard'
        """
        self.difficulty = difficulty
        stockfish_path = settings.STOCKFISH_PATH
        
        # Check if Stockfish is installed
        if not os.path.exists(stockfish_path):
            # Try common locations
            common_paths = [
                '/usr/games/stockfish',
                '/usr/local/bin/stockfish',
                '/usr/bin/stockfish',
                'stockfish'
            ]
            for path in common_paths:
                if os.path.exists(path) or path == 'stockfish':
                    stockfish_path = path
                    break
        
        try:
            self.engine = Stockfish(path=stockfish_path)
            self._configure_difficulty()
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            self.engine = None

    def _configure_difficulty(self):
        """Configure engine based on difficulty level"""
        if self.engine and self.difficulty in self.DIFFICULTY_SETTINGS:
            settings = self.DIFFICULTY_SETTINGS[self.difficulty]
            self.engine.set_depth(settings['depth'])
            self.engine.set_skill_level(settings['skill_level'])

    def get_best_move(self, fen):
        """
        Get best move for current position
        Args:
            fen: Board position in FEN notation
        Returns:
            str: Best move in UCI format (e.g., "e2e4")
        """
        if not self.engine:
            # Fallback: return a random legal move
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0].uci()
            return None

        try:
            self.engine.set_fen_position(fen)
            best_move = self.engine.get_best_move()
            return best_move
        except Exception as e:
            print(f"Error getting best move: {e}")
            # Fallback to random legal move
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if legal_moves:
                import random
                return random.choice(legal_moves).uci()
            return None

    def evaluate_position(self, fen):
        """
        Evaluate current position
        Args:
            fen: Board position in FEN notation
        Returns:
            int: Evaluation in centipawns (positive = white advantage)
        """
        if not self.engine:
            return 0

        try:
            self.engine.set_fen_position(fen)
            evaluation = self.engine.get_evaluation()
            if evaluation['type'] == 'cp':
                return evaluation['value']
            elif evaluation['type'] == 'mate':
                # Mate in X moves
                return 10000 if evaluation['value'] > 0 else -10000
            return 0
        except Exception as e:
            print(f"Error evaluating position: {e}")
            return 0
