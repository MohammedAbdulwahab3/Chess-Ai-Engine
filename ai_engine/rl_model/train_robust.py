"""
Robust Training Script for Chess RL

Features:
1. Imitation Learning (Pre-training vs Stockfish)
2. RL Training with Policy Gradient + Value Loss + Entropy
3. Replay Buffer
4. Proper Move Encoding/Masking
5. Evaluation Routine
6. Checkpointing & Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import os
import sys
import numpy as np
import time

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import local modules
from ai_engine.rl_model.chess_network import ChessRLNetwork, ChessBoardEncoder
from ai_engine.stockfish_engine import StockfishEngine
from ai_engine.rl_model.move_encoding import MoveLabeler
from ai_engine.rl_model.replay_buffer import ReplayBuffer

# Minimal Django setup for StockfishEngine if needed
# (StockfishEngine uses settings.STOCKFISH_PATH)
try:
    import django
    from django.conf import settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chess_backend.settings')
    django.setup()
except:
    pass # Might already be setup or not needed if StockfishEngine is modified

class RobustTrainer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model (LARGER for better learning)
        self.model = ChessRLNetwork(num_channels=256, num_res_blocks=8)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded model from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model (architecture mismatch?): {e}")
                print("Starting with fresh model")
        
        self.model.to(self.device)
        self.encoder = ChessBoardEncoder()
        self.labeler = MoveLabeler()
        
        # Higher learning rate for faster convergence
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        
        # Stockfish for opponent/teacher (START EASY!)
        try:
            self.stockfish_easy = StockfishEngine(difficulty='easy')
            self.stockfish_medium = StockfishEngine(difficulty='medium')
            self.stockfish = self.stockfish_easy  # Start with easy
        except:
            print("⚠️ Stockfish init failed (Django settings?). Using fallback/mock if possible.")
            # Fallback or error handling
            
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.save_path = model_path
        
    def get_model_move_from_policy(self, board, temperature=1.0, greedy=False):
        """
        Select move using policy logits and masking.
        """
        state = self.encoder.board_to_tensor(board)
        state_tensor = state.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            logits = logits.squeeze(0) # [vocab_size]
            
            # Mask illegal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None, None
                
            mask = torch.ones_like(logits) * float('-inf')
            legal_indices = []
            legal_moves_map = {}
            
            for m in legal_moves:
                idx = self.labeler.encode(m)
                if idx is not None and idx < logits.shape[0]:
                    legal_indices.append(idx)
                    legal_moves_map[idx] = m
            
            if not legal_indices:
                # Fallback: Random legal move
                import random
                move = random.choice(legal_moves)
                return move, self.labeler.encode(move)
                
            mask[legal_indices] = 0
            masked_logits = logits + mask
            
            if greedy:
                # Argmax
                action_idx = torch.argmax(masked_logits).item()
            else:
                # Sample with temperature
                probs = F.softmax(masked_logits / temperature, dim=0)
                
                # Safety check for NaNs
                if torch.isnan(probs).any() or probs.sum() == 0:
                     action_idx = legal_indices[0] # Fallback
                else:
                    action_idx = torch.multinomial(probs, 1).item()
            
            move = legal_moves_map.get(action_idx)
            if move is None:
                # Fallback
                move = legal_moves[0]
                action_idx = self.labeler.encode(move)
                
            return move, action_idx

    def pretrain_imitation(self, num_positions=5000):
        """
        Imitation Learning: Train policy to mimic Stockfish (FAST VERSION)
        """
        print(f"\n🤖 Starting Imitation Learning ({num_positions} positions)...")
        self.model.train()
        
        board = chess.Board()
        positions_collected = 0
        
        # Collection Phase
        while positions_collected < num_positions:
            if board.is_game_over():
                board.reset()
                
            try:
                # Get Stockfish move
                sf_move_uci = self.stockfish.get_best_move(board.fen())
                if not sf_move_uci:
                    board.push(list(board.legal_moves)[0])
                    continue
                    
                sf_move = chess.Move.from_uci(sf_move_uci)
                move_idx = self.labeler.encode(sf_move)
                
                if move_idx is not None:
                    state = self.encoder.board_to_tensor(board)
                    # Store for training (value=0 as placeholder)
                    self.replay_buffer.push(state, move_idx, 0.0)
                    positions_collected += 1
                    
                    if positions_collected % 200 == 0:
                        print(f"  Collected {positions_collected}/{num_positions}")
                
                board.push(sf_move)
                
            except Exception as e:
                # print(f"Error: {e}")
                board.reset()
                
        # Training Phase (OPTIMIZED!)
        print("  Training on collected data...")
        num_epochs = 25  # Increased for better learning
        batch_size = 64  # Larger batches
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Only train on 500 random samples per epoch (not all data)
            # This is much faster and still effective
            iterations = min(20, len(self.replay_buffer) // batch_size)
            
            for _ in range(iterations):
                states, actions, _ = self.replay_buffer.sample(batch_size)
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                logits, _ = self.model(states)
                
                loss = nn.CrossEntropyLoss()(logits, actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"  Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
            
    def train_rl(self, num_games=500, use_curriculum=True):
        """
        RL Training Loop with Curriculum Learning
        """
        print(f"\n⚔️ Starting RL Training ({num_games} games)...")
        if use_curriculum:
            print("📚 Curriculum: Easy Stockfish (0-300 games) -> Medium (300+ games)")
        
        wins = 0
        draws = 0
        losses = 0
        
        for game_idx in range(num_games):
            # Curriculum: Switch to medium Stockfish after 300 games
            if use_curriculum and game_idx == 300:
                print("\n🔥 Difficulty increased! Switching to Medium Stockfish...")
                self.stockfish = self.stockfish_medium
            
            board = chess.Board()
            game_memory = []
            
            # Use lower temperature from the start (less random!)
            temperature = max(0.3, 0.7 - (game_idx / num_games) * 0.4)
            
            # Play Game
            while not board.is_game_over(claim_draw=True) and len(game_memory) < 150:
                # Model Move
                move, action_idx = self.get_model_move_from_policy(board, temperature=temperature)
                
                if move is None:
                    break
                    
                state = self.encoder.board_to_tensor(board)
                game_memory.append((state, action_idx))
                board.push(move)
                
                # Opponent Move (Stockfish)
                if not board.is_game_over():
                    try:
                        sf_move_uci = self.stockfish.get_best_move(board.fen())
                        if sf_move_uci:
                            board.push(chess.Move.from_uci(sf_move_uci))
                    except:
                        break
            
            # Result
            result = board.result(claim_draw=True)
            if result == "1-0": # White (Model) wins
                outcome = 1.0
                wins += 1
            elif result == "0-1": # Black (Stockfish) wins
                outcome = -1.0
                losses += 1
            else:
                outcome = 0.0
                draws += 1
                
            # Store in buffer
            for state, action_idx in game_memory:
                self.replay_buffer.push(state, action_idx, outcome)
                
            # Train MORE frequently (every game after buffer has data)
            if len(self.replay_buffer) > 64:
                # Multiple training steps per game
                for _ in range(3):
                    stats = self._train_step()
                
            # Logging
            if (game_idx + 1) % 10 == 0:
                win_rate = (wins / (game_idx + 1)) * 100
                draw_rate = (draws / (game_idx + 1)) * 100
                print(f"Game {game_idx+1}: W={wins} ({win_rate:.1f}%), D={draws} ({draw_rate:.1f}%), L={losses}")
                if 'stats' in locals():
                    print(f"  Loss: {stats['loss']:.4f} (Pol: {stats['policy_loss']:.4f}, Val: {stats['value_loss']:.4f})")
                
                # Checkpoint
                if (game_idx + 1) % 100 == 0:
                    self.save_model()
                    # Quick eval
                    print("  Quick eval (5 games):")
                    self.evaluate_against_stockfish(num_games=5)
                    
    def _train_step(self):
        self.model.train()
        states, actions, values = self.replay_buffer.sample(64)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        values = values.to(self.device)
        
        logits, value_pred = self.model(states)
        
        # 1. Policy Loss (Cross Entropy)
        policy_loss = nn.CrossEntropyLoss()(logits, actions)
        
        # 2. Value Loss (MSE)
        value_loss = nn.MSELoss()(value_pred, values)
        
        # 3. Entropy Bonus (Regularization)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # Combined Loss
        # loss = policy + 0.5 * value - 0.01 * entropy
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def evaluate_against_stockfish(self, num_games=20):
        """
        Evaluate model vs Stockfish (Deterministic/Greedy)
        """
        print(f"\n📊 Evaluating vs Stockfish ({num_games} games)...")
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(num_games):
            board = chess.Board()
            moves = 0
            while not board.is_game_over() and moves < 150:
                # Model (White)
                move, _ = self.get_model_move_from_policy(board, greedy=True)
                if move:
                    board.push(move)
                else:
                    break
                    
                # Stockfish (Black)
                if not board.is_game_over():
                    sf_move = self.stockfish.get_best_move(board.fen())
                    if sf_move:
                        board.push(chess.Move.from_uci(sf_move))
                moves += 1
                
            res = board.result()
            if res == "1-0": wins += 1
            elif res == "0-1": losses += 1
            else: draws += 1
            
        print(f"Results: Wins={wins}, Draws={draws}, Losses={losses}")
        return wins, draws, losses

    def save_model(self):
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)
            print(f"✓ Saved model to {self.save_path}")

def main():
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    save_path = os.path.join(save_dir, 'rl_chess_v1.pth')
    
    trainer = RobustTrainer(save_path)
    
    # 1. Imitation Learning (Bootstrap from Stockfish) - More data!
    trainer.pretrain_imitation(num_positions=5000)  # Increased from 1000
    
    # 2. RL Training (WITH CURRICULUM!)
    trainer.train_rl(num_games=500, use_curriculum=True)
    
    # 3. Final Evaluation vs Easy
    print("\n" + "="*50)
    print("FINAL EVALUATION vs EASY Stockfish")
    print("="*50)
    trainer.stockfish = trainer.stockfish_easy
    trainer.evaluate_against_stockfish(num_games=20)
    
    # 4. Final Evaluation vs Medium
    print("\n" + "="*50)
    print("FINAL EVALUATION vs MEDIUM Stockfish")
    print("="*50)
    trainer.stockfish = trainer.stockfish_medium
    trainer.evaluate_against_stockfish(num_games=20)
    
    # Final Save
    trainer.save_model()

if __name__ == "__main__":
    main()
