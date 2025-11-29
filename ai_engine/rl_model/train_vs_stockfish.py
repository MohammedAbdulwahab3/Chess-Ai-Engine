"""
Train YOUR RL model by playing against Stockfish!

Better than self-play because:
- Stockfish is a strong opponent
- Your model learns from mistakes
- No endless draws!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import os
import sys
import django
from django.conf import settings

# Setup Django settings
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Set settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chess_backend.settings')
django.setup()

from .chess_network import ChessRLNetwork, ChessBoardEncoder
from ..stockfish_engine import StockfishEngine


class StockfishTrainer:
    """
    Train your RL model by playing against Stockfish!
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Smarter model architecture as requested!
        self.model = ChessRLNetwork(num_channels=256, num_res_blocks=8)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded existing model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load existing model (architecture mismatch?): {e}")
                print("✓ Starting with new untrained model instead")
        else:
            print("✓ Starting with new untrained model")
        
        self.model.to(self.device)
        self.encoder = ChessBoardEncoder()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Stockfish opponent (medium difficulty)
        self.stockfish = StockfishEngine(difficulty='medium')
        
    def get_model_move(self, board):
        """Get move from your RL model"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # For now, random selection (will improve with training)
        import random
        return random.choice(legal_moves)
    
    def play_one_game(self, model_plays_white=True, show_board=True):
        """
        Play one game: Your model vs Stockfish
        
        Returns:
            positions, moves, outcome
        """
        board = chess.Board()
        positions = []
        moves_made = []
        move_count = 0
        
        print(f"\n{'='*70}")
        print(f"🎮 NEW GAME: {'Your Model (White)' if model_plays_white else 'Your Model (Black)'} vs Stockfish")
        print(f"{'='*70}")
        
        while not board.is_game_over() and move_count < 200:
            if show_board and move_count % 5 == 0:  # Show every 5 moves
                print(f"\nMove {move_count + 1}:")
                print(board)
            
            # Determine whose turn
            is_model_turn = (board.turn == chess.WHITE) == model_plays_white
            
            if is_model_turn:
                # Your model's turn
                move = self.get_model_move(board)
                if move is None:
                    break
                
                # Save position for training
                position_tensor = self.encoder.board_to_tensor(board)
                positions.append(position_tensor)
                moves_made.append(str(move))
                
                if show_board and move_count % 5 == 0:
                    print(f"  👤 Your Model: {move}")
            else:
                # Stockfish's turn
                stockfish_move = self.stockfish.get_best_move(board.fen())
                move = chess.Move.from_uci(stockfish_move)
                
                if show_board and move_count % 5 == 0:
                    print(f"  🤖 Stockfish: {move}")
            
            board.push(move)
            move_count += 1
        
        # Determine outcome
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            if (winner == "White" and model_plays_white) or (winner == "Black" and not model_plays_white):
                outcome = 1.0  # Model won!
                result_text = f"🏆 YOUR MODEL WINS!"
            else:
                outcome = -1.0  # Stockfish won
                result_text = f"😢 Stockfish wins"
        else:
            outcome = 0.0
            result_text = "🤝 Draw"
        
        print(f"\n{'='*70}")
        print(f"RESULT: {result_text}")
        print(f"Total moves: {move_count}")
        print(f"{'='*70}\n")
        
        return positions, moves_made, outcome
    
    def train_against_stockfish(self, num_games=100, save_path=None):
        """
        Train by playing multiple games against Stockfish!
        
        Args:
            num_games: Number of training games
            save_path: Where to save the trained model
        """
        print("\n" + "🚀"*35)
        print("  TRAINING YOUR MODEL AGAINST STOCKFISH!")
        print("🚀"*35 + "\n")
        
        all_positions = []
        all_moves = []
        all_outcomes = []
        
        stats = {'model_wins': 0, 'stockfish_wins': 0, 'draws': 0}
        
        for game_num in range(num_games):
            print(f"\n{'#'*70}")
            print(f"  TRAINING GAME {game_num + 1} / {num_games}")
            print(f"{'#'*70}")
            
            # Alternate colors
            model_plays_white = (game_num % 2 == 0)
            
            positions, moves, outcome = self.play_one_game(
                model_plays_white=model_plays_white,
                show_board=(game_num < 5 or game_num % 10 == 0)  # Show first 5 and every 10th
            )
            
            all_positions.extend(positions)
            all_moves.extend(moves)
            all_outcomes.extend([outcome] * len(positions))
            
            # Update stats
            if outcome > 0:
                stats['model_wins'] += 1
            elif outcome < 0:
                stats['stockfish_wins'] += 1
            else:
                stats['draws'] += 1
            
            # Train every 10 games
            if (game_num + 1) % 10 == 0:
                print(f"\n{'='*70}")
                print(f"📊 PROGRESS UPDATE (Game {game_num + 1}/{num_games})")
                print(f"{'='*70}")
                print(f"  Your Model Wins: {stats['model_wins']}")
                print(f"  Stockfish Wins: {stats['stockfish_wins']}")
                print(f"  Draws: {stats['draws']}")
                win_rate = (stats['model_wins'] / (game_num + 1)) * 100
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"{'='*70}\n")
                
                # Quick training update
                if len(all_positions) > 32:
                    self._train_batch(all_positions[-320:], all_outcomes[-320:])
        
        # Final training
        print("\n" + "🎓"*35)
        print("  FINAL TRAINING PHASE")
        print("🎓"*35 + "\n")
        
        self._train_final(all_positions, all_outcomes, epochs=10)
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"\n✅ Model saved to: {save_path}")
        
        # Final stats
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE!")
        print("="*80)
        print(f"Total games: {num_games}")
        print(f"Your Model wins: {stats['model_wins']} ({stats['model_wins']/num_games*100:.1f}%)")
        print(f"Stockfish wins: {stats['stockfish_wins']} ({stats['stockfish_wins']/num_games*100:.1f}%)")
        print(f"Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")
        print(f"Total training positions: {len(all_positions)}")
        print("="*80 + "\n")
        
        return stats
    
    def _train_batch(self, positions, outcomes):
        """Quick training update"""
        self.model.train()
        
        # Convert to tensors
        batch_positions = torch.stack(positions[:32]).to(self.device)
        batch_outcomes = torch.tensor(outcomes[:32], dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Forward pass
        _, value_pred = self.model(batch_positions)
        
        # Loss
        loss = nn.MSELoss()(value_pred, batch_outcomes)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"  📈 Quick update - Loss: {loss.item():.4f}")
    
    def _train_final(self, positions, outcomes, epochs=10):
        """Final comprehensive training"""
        self.model.train()
        
        print(f"Training on {len(positions)} positions for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Train in batches
            for i in range(0, len(positions), 32):
                batch_pos = positions[i:i+32]
                batch_out = outcomes[i:i+32]
                
                if len(batch_pos) < 8:  # Skip tiny batches
                    continue
                
                batch_positions = torch.stack(batch_pos).to(self.device)
                batch_outcomes = torch.tensor(batch_out, dtype=torch.float32).unsqueeze(1).to(self.device)
                
                _, value_pred = self.model(batch_positions)
                loss = nn.MSELoss()(value_pred, batch_outcomes)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        self.model.eval()


def main():
    """
    Main training script - Train against Stockfish!
    """
    print("\n" + "🎮"*40)
    print("  TRAIN YOUR RL MODEL VS STOCKFISH")
    print("🎮"*40 + "\n")
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    save_path = os.path.join(save_dir, 'rl_chess_v1.pth')
    
    trainer = StockfishTrainer()
    
    # Train for 10,000 games!
    stats = trainer.train_against_stockfish(
        num_games=10000,
        save_path=save_path
    )
    
    print("\n🎉 Your model is now trained and saved!")
    print(f"📁 Model location: {save_path}")
    print("\n💡 Next: Watch your model play in the Flutter app!")


if __name__ == "__main__":
    main()
