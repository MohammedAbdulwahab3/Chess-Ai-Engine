"""
Training script for YOUR custom chess RL model!

This will train your model to give good chess hints.
Start here to learn reinforcement learning!

Training approach: Supervised Learning from expert games
- Download chess games from Lichess
- Extract positions and moves
- Train your model to predict good moves
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import os
from .chess_network import ChessRLNetwork, ChessBoardEncoder


class ChessDataset(Dataset):
    """
    Dataset of chess positions and moves from real games
    """
    
    def __init__(self, positions, moves, outcomes):
        self.positions = positions
        self.moves = moves
        self.outcomes = outcomes
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'move': self.moves[idx],
            'outcome': self.outcomes[idx]
        }


def extract_training_data_from_pgn(pgn_file, max_games=1000):
    """
    Extract training data from PGN file
    
    Args:
        pgn_file: Path to PGN file with chess games
        max_games: Maximum number of games to process
        
    Returns:
        positions, moves, outcomes for training
    """
    positions = []
    moves = []
    outcomes = []
    
    encoder = ChessBoardEncoder()
    
    with open(pgn_file) as f:
        game_count = 0
        
        while game_count < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            # Get game outcome
            result = game.headers.get("Result", "*")
            if result == "1-0":
                outcome = 1.0  # White wins
            elif result == "0-1":
                outcome = -1.0  # Black wins
            elif result == "1/2-1/2":
                outcome = 0.0  # Draw
            else:
                continue  # Skip unfinished games
            
            # Extract positions and moves
            board = game.board()
            for move in game.mainline_moves():
                # Convert position to tensor
                position_tensor = encoder.board_to_tensor(board)
                positions.append(position_tensor)
                
                # Store move (simplified - need proper move encoding)
                moves.append(move.uci())
                
                # Store outcome  
                outcomes.append(outcome)
                
                # Make the move
                board.push(move)
            
            game_count += 1
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {len(positions)} positions")
    
    return positions, moves, outcomes


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Train your chess RL model!
    
    This uses supervised learning - the model learns from expert games.
    Later you can add self-play for true RL!
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    print(f"Training on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            positions = batch['position'].to(device)
            outcomes = batch['outcome'].float().to(device).unsqueeze(1)
            
            # Forward pass
            policy_pred, value_pred = model(positions)
            
            # Compute losses
            # Note: Simplified - need proper move encoding for policy loss
            value_loss = value_criterion(value_pred, outcomes)
            
            # Combined loss (for now just value loss)
            loss = value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Value Loss: {avg_value_loss:.4f}")
        print(f"{'='*50}\n")
    
    return model


def main():
    """
    Main training script
    """
    print("🚀 Training YOUR Custom Chess RL Model!")
    print("=" * 60)
    
    # Step 1: Load or create training data
    print("\n[1/4] Loading training data...")
    
    # For now, create dummy data as example
    # TODO: Download real PGN files from Lichess
    print("NOTE: Using random data for demonstration")
    print("Download real games from: https://database.lichess.org/")
    
    dummy_positions = [torch.randn(12, 8, 8) for _ in range(1000)]
    dummy_moves = ['e2e4'] * 1000
    dummy_outcomes = [torch.tensor([0.0]) for _ in range(1000)]
    
    dataset = ChessDataset(dummy_positions, dummy_moves, dummy_outcomes)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"✓ Loaded {len(dataset)} training positions")
    
    # Step 2: Create model
    print("\n[2/4] Creating neural network...")
    model = ChessRLNetwork(num_channels=128, num_res_blocks=4)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 3: Train
    print("\n[3/4] Training model...")
    print("This will take a while depending on your hardware!")
    trained_model = train_model(model, train_loader, num_epochs=100)
    
    # Step 4: Save model
    print("\n[4/4] Saving model...")
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rl_chess_v1.pth')
    
    torch.save(trained_model.state_dict(), save_path)
    print(f"✓ Model saved to: {save_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("Your model is now ready to give hints in the chess game!")
    print("=" * 60)


if __name__ == "__main__":
    main()
