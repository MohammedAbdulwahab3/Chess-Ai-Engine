import chess
from .models import Game, Move


class ChessLogic:
    """Chess game logic and move validation"""
    
    @staticmethod
    def validate_move(game, move_uci):
        """
        Validate a chess move
        Args:
            game: Game instance
            move_uci: Move in UCI format (e.g., "e2e4")
        Returns:
            tuple: (is_valid, error_message, move_object)
        """
        try:
            board = game.get_board()
            move = chess.Move.from_uci(move_uci)
            
            if move not in board.legal_moves:
                return False, "Illegal move", None
            
            return True, None, move
        except Exception as e:
            return False, str(e), None

    @staticmethod
    def make_move(game, player, move_uci):
        """
        Make a move in the game
        Args:
            game: Game instance
            player: User making the move
            move_uci: Move in UCI format
        Returns:
            tuple: (success, message, move_instance)
        """
        # Validate it's the player's turn
        if not game.is_player_turn(player):
            return False, "Not your turn", None

        # Validate the move
        is_valid, error, move = ChessLogic.validate_move(game, move_uci)
        if not is_valid:
            return False, error, None

        # Apply the move
        board = game.get_board()
        san_notation = board.san(move)
        board.push(move)

        # Create move record
        move_number = game.moves.count() + 1
        move_instance = Move.objects.create(
            game=game,
            player=player,
            move_notation=move_uci,
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

        return True, "Move successful", move_instance

    @staticmethod
    def _update_player_stats(game, result):
        """Update player statistics after game completion"""
        if game.game_mode == 'ai':
            # Only update the human player's stats
            player = game.white_player if game.ai_color == 'black' else game.black_player
            if player:
                player.games_played += 1
                if result == 'draw':
                    player.games_drawn += 1
                elif (result == 'white_wins' and game.white_player == player) or \
                     (result == 'black_wins' and game.black_player == player):
                    player.games_won += 1
                else:
                    player.games_lost += 1
                player.save()
        else:
            # Update both players' stats
            for player in [game.white_player, game.black_player]:
                if player:
                    player.games_played += 1
                    if result == 'draw':
                        player.games_drawn += 1
                    elif (result == 'white_wins' and player == game.white_player) or \
                         (result == 'black_wins' and player == game.black_player):
                        player.games_won += 1
                    else:
                        player.games_lost += 1
                    player.save()

    @staticmethod
    def get_legal_moves(game):
        """Get all legal moves for current position"""
        board = game.get_board()
        return [move.uci() for move in board.legal_moves]

    @staticmethod
    def get_game_status(game):
        """Get detailed game status"""
        board = game.get_board()
        return {
            'is_check': board.is_check(),
            'is_checkmate': board.is_checkmate(),
            'is_stalemate': board.is_stalemate(),
            'is_game_over': board.is_game_over(),
            'current_turn': game.current_turn,
            'legal_moves': ChessLogic.get_legal_moves(game)
        }
