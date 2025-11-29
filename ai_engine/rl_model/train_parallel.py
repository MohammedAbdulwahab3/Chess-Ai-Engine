"""
Parallel Training Script for Chess RL

Uses multiprocessing to play multiple games simultaneously.
Can utilize all CPU cores for faster training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import os
import sys
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, Queue
from functools import partial

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import local modules
from ai_engine.rl_model.chess_network import ChessRLNetwork, ChessBoardEncoder
from ai_engine.stockfish_engine import StockfishEngine
from ai_engine.rl_model.move_encoding import MoveLabeler
from ai_engine.rl_model.replay_buffer import ReplayBuffer

# Minimal Django setup
try:
    import django
    from django.conf import settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chess_backend.settings')
    django.setup()
except:
    pass

def play_game_worker(args):
    """
    Worker function to play one game.
    Returns list of (state, action_idx, outcome) tuples.
    """
    game_idx, model_state_dict, difficulty, temperature = args
    
    # Initialize model for this worker
    device = torch.device('cpu')  # Each worker uses CPU
    model = ChessRLNetwork(num_channels=256, num_res_blocks=8)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    encoder = ChessBoardEncoder()
    labeler = MoveLabeler()
    stockfish = StockfishEngine(difficulty=difficulty)
    
    board = chess.Board()
    game_memory = []
    
    # Play Game
    while not board.is_game_over(claim_draw=True) and len(game_memory) < 150:
        # Model Move
        state = encoder.board_to_tensor(board)
        state_tensor = state.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(state_tensor)
            logits = logits.squeeze(0)
            
            # Mask illegal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
                
            mask = torch.ones_like(logits) * float('-inf')
            legal_indices = []
            legal_moves_map = {}
            
            for m in legal_moves:
                idx = labeler.encode(m)
                if idx is not None and idx < logits.shape[0]:
                    legal_indices.append(idx)
                    legal_moves_map[idx] = m
            
            if not legal_indices:
                import random
                move = random.choice(legal_moves)
                action_idx = labeler.encode(move)
            else:
                mask[legal_indices] = 0
                masked_logits = logits + mask
                
                # Sample with temperature
                probs = F.softmax(masked_logits / temperature, dim=0)
                
                if torch.isnan(probs).any() or probs.sum() == 0:
                    action_idx = legal_indices[0]
                else:
                    action_idx = torch.multinomial(probs, 1).item()
                
                move = legal_moves_map.get(action_idx, legal_moves[0])
                if move is None:
                    move = legal_moves[0]
                    action_idx = labeler.encode(move)
        
        game_memory.append((state, action_idx))
        board.push(move)
        
        # Opponent Move
        if not board.is_game_over():
            try:
                sf_move_uci = stockfish.get_best_move(board.fen())
                if sf_move_uci:
                    board.push(chess.Move.from_uci(sf_move_uci))
            except:
                break
    
    # Result
    result = board.result(claim_draw=True)
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0
    
    # Return experiences
    experiences = [(state, action_idx, outcome) for state, action_idx in game_memory]
    return experiences, outcome


class ParallelTrainer:
    def __init__(self, model_path=None, num_workers=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Auto-detect CPU cores
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)  # Leave 1 core free
        else:
            self.num_workers = num_workers
            
        print(f"Using device: {self.device}")
        print(f"Parallel workers: {self.num_workers}")
        
        # Initialize model
        self.model = ChessRLNetwork(num_channels=256, num_res_blocks=8)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded model from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
        
        self.model.to(self.device)
        self.encoder = ChessBoardEncoder()
        self.labeler = MoveLabeler()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        
        # Stockfish for evaluation
        try:
            self.stockfish_easy = StockfishEngine(difficulty='easy')
            self.stockfish_medium = StockfishEngine(difficulty='medium')
        except:
            print("⚠️ Stockfish init failed")
            
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.save_path = model_path
        
    def pretrain_imitation(self, num_positions=1000):
        """Imitation Learning (FAST version)"""
        print(f"\n🤖 Starting Imitation Learning ({num_positions} positions)...")
        self.model.train()
        
        board = chess.Board()
        positions_collected = 0
        
        while positions_collected < num_positions:
            if board.is_game_over():
                board.reset()
                
            try:
                sf_move_uci = self.stockfish_easy.get_best_move(board.fen())
                if not sf_move_uci:
                    board.push(list(board.legal_moves)[0])
                    continue
                    
                sf_move = chess.Move.from_uci(sf_move_uci)
                move_idx = self.labeler.encode(sf_move)
                
                if move_idx is not None:
                    state = self.encoder.board_to_tensor(board)
                    self.replay_buffer.push(state, move_idx, 0.0)
                    positions_collected += 1
                    
                    if positions_collected % 200 == 0:
                        print(f"  Collected {positions_collected}/{num_positions}")
                
                board.push(sf_move)
                
            except Exception as e:
                board.reset()
                
        # Training Phase (FAST!)
        print("  Training on collected data...")
        for epoch in range(5):  # Only 5 epochs
            total_loss = 0
            num_batches = 0
            iterations = min(20, len(self.replay_buffer) // 64)  # Max 20 iterations
            
            for _ in range(iterations):
                states, actions, _ = self.replay_buffer.sample(64)
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
            print(f"  Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    def train_rl_parallel(self, num_games=500, use_curriculum=True, batch_size=None):
        """
        RL Training with Parallel Game Playing
        """
        if batch_size is None:
            batch_size = self.num_workers * 4  # Play multiple batches per iteration
            
        print(f"\n⚔️ Starting PARALLEL RL Training ({num_games} games)...")
        print(f"  Batch size: {batch_size} games at a time")
        if use_curriculum:
            print("📚 Curriculum: Easy Stockfish (0-300) -> Medium (300+)")
        
        wins = 0
        draws = 0
        losses = 0
        games_played = 0
        
        # Create process pool
        with Pool(processes=self.num_workers) as pool:
            while games_played < num_games:
                # Determine difficulty
                if use_curriculum and games_played >= 300:
                    difficulty = 'medium'
                else:
                    difficulty = 'easy'
                
                # Temperature annealing
                temperature = max(0.5, 1.0 - (games_played / num_games) * 0.5)
                
                # Prepare arguments for workers
                games_to_play = min(batch_size, num_games - games_played)
                model_state = self.model.state_dict()
                
                args_list = [
                    (i, model_state, difficulty, temperature)
                    for i in range(games_to_play)
                ]
                
                # Play games in parallel!
                print(f"\n🎮 Playing games {games_played+1} to {games_played+games_to_play} ({difficulty})...")
                results = pool.map(play_game_worker, args_list)
                
                # Collect results
                batch_wins = 0
                batch_draws = 0
                batch_losses = 0
                
                for experiences, outcome in results:
                    # Add to replay buffer
                    for exp in experiences:
                        self.replay_buffer.push(*exp)
                    
                    # Update stats
                    if outcome > 0:
                        wins += 1
                        batch_wins += 1
                    elif outcome < 0:
                        losses += 1
                        batch_losses += 1
                    else:
                        draws += 1
                        batch_draws += 1
                
                games_played += games_to_play
                
                # Train on collected data
                if len(self.replay_buffer) > 64:
                    print(f"  Training on replay buffer ({len(self.replay_buffer)} experiences)...")
                    for _ in range(10):  # Multiple training iterations
                        stats = self._train_step()
                
                # Logging
                win_rate = (wins / games_played) * 100
                draw_rate = (draws / games_played) * 100
                print(f"\n📊 Progress: {games_played}/{num_games}")
                print(f"  Overall: W={wins} ({win_rate:.1f}%), D={draws} ({draw_rate:.1f}%), L={losses}")
                print(f"  This batch: W={batch_wins}, D={batch_draws}, L={batch_losses}")
                if 'stats' in locals():
                    print(f"  Loss: {stats['loss']:.4f} (Pol: {stats['policy_loss']:.4f}, Val: {stats['value_loss']:.4f})")
                
                # Checkpoint every 100 games
                if games_played % 100 == 0:
                    self.save_model()
                    
    def _train_step(self):
        """Single training step"""
        self.model.train()
        states, actions, values = self.replay_buffer.sample(64)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        values = values.to(self.device)
        
        logits, value_pred = self.model(states)
        
        policy_loss = nn.CrossEntropyLoss()(logits, actions)
        value_loss = nn.MSELoss()(value_pred, values)
        
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def evaluate_against_stockfish(self, num_games=20, difficulty='easy'):
        """Evaluate model"""
        print(f"\n📊 Evaluating vs {difficulty} Stockfish ({num_games} games)...")
        
        stockfish = self.stockfish_easy if difficulty == 'easy' else self.stockfish_medium
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(num_games):
            board = chess.Board()
            moves = 0
            
            while not board.is_game_over() and moves < 150:
                state = self.encoder.board_to_tensor(board)
                state_tensor = state.unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    logits, _ = self.model(state_tensor)
                    logits = logits.squeeze(0)
                    
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    mask = torch.ones_like(logits) * float('-inf')
                    legal_indices = []
                    for m in legal_moves:
                        idx = self.labeler.encode(m)
                        if idx is not None and idx < logits.shape[0]:
                            legal_indices.append(idx)
                            mask[idx] = 0
                    
                    if legal_indices:
                        masked_logits = logits + mask
                        action_idx = torch.argmax(masked_logits).item()
                        move = self.labeler.decode(action_idx)
                        if move not in legal_moves:
                            move = legal_moves[0]
                    else:
                        move = legal_moves[0]
                
                board.push(move)
                
                if not board.is_game_over():
                    sf_move = stockfish.get_best_move(board.fen())
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
    # Force spawn method for multiprocessing (safer on Linux)
    mp.set_start_method('spawn', force=True)
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    save_path = os.path.join(save_dir, 'rl_chess_v1.pth')
    
    # Use all available cores
    trainer = ParallelTrainer(save_path, num_workers=None)
    
    # 1. Imitation Learning (FAST!)
    trainer.pretrain_imitation(num_positions=1000)
    
    # 2. PARALLEL RL Training
    trainer.train_rl_parallel(num_games=500, use_curriculum=True, batch_size=32)
    
    # 3. Final Evaluations
    print("\n" + "="*50)
    print("FINAL EVALUATION vs EASY Stockfish")
    print("="*50)
    trainer.evaluate_against_stockfish(num_games=20, difficulty='easy')
    
    print("\n" + "="*50)
    print("FINAL EVALUATION vs MEDIUM Stockfish")
    print("="*50)
    trainer.evaluate_against_stockfish(num_games=20, difficulty='medium')
    
    trainer.save_model()


if __name__ == "__main__":
    main()
