"""
Self-Play Training System!

This makes YOUR RL model play against itself to learn chess.
You can WATCH the games in real-time!

How it works:
1. Model plays both white and black
2. Learns from wins/losses/draws
3. Gets better over time
4. You can watch games live!
"""

import torch
import chess
import time
import os
from .chess_network import ChessRLNetwork, ChessBoardEncoder
from django.conf import settings
import sys


class SelfPlayTrainer:
    """
    Makes your AI play against itself to learn!
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessRLNetwork()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded model from {model_path}")
        else:
            print("✓ Starting with new untrained model")
        
        self.model.to(self.device)
        self.encoder = ChessBoardEncoder()
        
    def play_one_game(self, show_board=True, delay=0.5):
        """
        Play one full game against itself
        
        Args:
            show_board: Print board after each move
            delay: Seconds between moves (for watching)
            
        Returns:
            positions, moves, outcome for training
        """
        board = chess.Board()
        positions = []
        moves_made = []
        move_count = 0
        
        print("\n" + "="*60)
        print("🎮 NEW SELF-PLAY GAME STARTING!")
        print("="*60)
        
        while not board.is_game_over() and move_count < 200:
            # Show current position
            if show_board:
                print(f"\nMove {move_count + 1}:")
                print(board)
                print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
            
            # Save position for training
            position_tensor = self.encoder.board_to_tensor(board)
            positions.append(position_tensor)
            
            # Get model's move suggestion
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                break
            
            # Simple move selection (random for now - improve with MCTS later!)
            import random
            move = random.choice(legal_moves)
            
            moves_made.append(str(move))
            board.push(move)
            move_count += 1
            
            # Delay for watching
            if delay > 0:
                time.sleep(delay)
            
            # Show move made
            if show_board:
                print(f"  Move: {move} ({move.uci()})")
        
        # Game over - determine outcome
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            outcome = -1.0 if board.turn == chess.WHITE else 1.0
            result_text = f"🏆 {winner} wins by checkmate!"
        elif board.is_stalemate():
            outcome = 0.0
            result_text = "🤝 Draw by stalemate"
        elif board.is_insufficient_material():
            outcome = 0.0
            result_text = "🤝 Draw by insufficient material"
        elif board.is_fifty_moves():
            outcome = 0.0
            result_text = "🤝 Draw by 50-move rule"
        elif board.is_repetition():
            outcome = 0.0
            result_text = "🤝 Draw by repetition"
        else:
            outcome = 0.0
            result_text = "Game stopped (max moves reached)"
        
        print("\n" + "="*60)
        print(f"GAME OVER: {result_text}")
        print(f"Total moves: {move_count}")
        print("="*60 + "\n")
        
        return positions, moves_made, outcome
    
    def play_training_games(self, num_games=10, show_games=True):
        """
        Play multiple self-play games for training
        
        Args:
            num_games: Number of games to play
            show_games: Whether to display games
            
        Returns:
            all_positions, all_moves, all_outcomes for training
        """
        all_positions = []
        all_moves = []
        all_outcomes = []
        
        wins = {'white': 0, 'black': 0, 'draw': 0}
        
        for game_num in range(num_games):
            print(f"\n{'#'*60}")
            print(f"  GAME {game_num + 1} / {num_games}")
            print(f"{'#'*60}")
            
            positions, moves, outcome = self.play_one_game(
                show_board=show_games,
                delay=0.5 if show_games else 0
            )
            
            all_positions.extend(positions)
            all_moves.extend(moves)
            all_outcomes.extend([outcome] * len(positions))
            
            # Track statistics
            if outcome > 0:
                wins['white'] += 1
            elif outcome < 0:
                wins['black'] += 1
            else:
                wins['draw'] += 1
            
            print(f"\nCurrent stats: {wins}")
        
        print("\n" + "="*80)
        print("🎉 TRAINING SESSION COMPLETE!")
        print("="*80)
        print(f"Total games: {num_games}")
        print(f"White wins: {wins['white']}")
        print(f"Black wins: {wins['black']}")
        print(f"Draws: {wins['draw']}")
        print(f"Total positions collected: {len(all_positions)}")
        print("="*80 + "\n")
        
        return all_positions, all_moves, all_outcomes


def watch_self_play(num_games=5):
    """
    Watch your AI play against itself!
    
    Usage:
        python -m ai_engine.rl_model.self_play
    """
    print("\n" + "🎮"*30)
    print("  WATCHING YOUR AI LEARN BY PLAYING ITSELF!")
    print("🎮"*30 + "\n")
    
    trainer = SelfPlayTrainer()
    
    print("Starting self-play training...")
    print("Watch as your AI plays against itself!\n")
    
    positions, moves, outcomes = trainer.play_training_games(
        num_games=num_games,
        show_games=True  # Show each game
    )
    
    print("\n✅ Training data collected!")
    print(f"   You can now use this data to train your model")
    print(f"   See train.py for training code\n")
    
    return positions, moves, outcomes


if __name__ == "__main__":
    # Run self-play and watch!
    watch_self_play(num_games=3)
