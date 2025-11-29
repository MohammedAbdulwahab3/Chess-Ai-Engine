"""
Custom Reinforcement Learning Chess Engine
This is YOUR chess AI built from scratch to learn RL/ML!

Architecture:
- Input: Chess board (8x8x12 tensor - 12 piece types)
- Neural Network: Convolutional layers + Fully connected
- Output: 
  * Policy head: Move probabilities
  * Value head: Position evaluation

This will start weak but improve with training!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np


class ChessRLNetwork(nn.Module):
    """
    Your custom chess AI neural network!
    
    Similar to AlphaZero architecture but simpler for learning.
    """
    
    def __init__(self, num_channels=128, num_res_blocks=4):
        super(ChessRLNetwork, self).__init__()
        
        # Input: 8x8x12 (board representation)
        # 12 channels: 6 piece types × 2 colors
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(12, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual blocks (like in AlphaZero)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head (which move to make)
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # Vocab size from MoveLabeler is roughly 4256 (4096 + promotions)
        # We'll use 4608 to be safe/aligned or exact size.
        # Let's use 4256 to match the MoveLabeler logic (4096 + 64*2.5 approx)
        # Actually let's just use a safe large number or import it.
        # For simplicity in this file without circular imports, we'll use 4672 (AlphaZero size) or just 4300.
        # Let's use 4672 to be safe and cover everything.
        self.policy_vocab_size = 4672 
        self.policy_fc = nn.Linear(32 * 8 * 8, self.policy_vocab_size)
        
        # Value head (who's winning)
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Board tensor (batch_size, 12, 8, 8)
            
        Returns:
            policy_logits: Raw move scores (batch_size, 4096) - NO SOFTMAX
            value: Position evaluation (batch_size, 1)
        """
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        # NO SOFTMAX HERE! We need raw logits for CrossEntropyLoss and masking
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output between -1 and 1
        
        return policy, value


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections
    Helps the network learn better
    """
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        x = F.relu(x)
        return x


class ChessBoardEncoder:
    """
    Converts chess positions to neural network input
    """
    
    @staticmethod
    def board_to_tensor(board):
        """
        Convert python-chess board to tensor
        
        Returns:
            tensor: (12, 8, 8) representing the board
        """
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        # Piece type mapping
        piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - (square // 8)  # Flip row for correct orientation
                col = square % 8
                channel = piece_to_channel[piece.piece_type]
                
                # Add 6 for black pieces
                if piece.color == chess.BLACK:
                    channel += 6
                
                tensor[channel, row, col] = 1.0
        
        return torch.FloatTensor(tensor)
    
    @staticmethod
    def fen_to_tensor(fen):
        """
        Convert FEN string to tensor
        """
        board = chess.Board(fen)
        return ChessBoardEncoder.board_to_tensor(board)


class ChessRLEngine:
    """
    Your custom RL chess engine!
    This is what will play chess using your neural network.
    """
    
    def __init__(self, model_path=None, device=None, num_channels=128, num_res_blocks=4):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with specified architecture
        self.model = ChessRLNetwork(num_channels=num_channels, num_res_blocks=num_res_blocks)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except FileNotFoundError:
                print(f"Model file not found: {model_path}")
                print("Starting with untrained model")
            except Exception as e:
                print(f"Error loading model (architecture mismatch?): {e}")
                print("Starting with untrained model")
        
        self.model.to(self.device)
        self.model.eval()
        
    def get_best_move(self, fen):
        """
        Get the best move for a given position
        
        Args:
            fen: Position in FEN notation
            
        Returns:
            move: Best move in UCI format (e.g., "e2e4")
        """
        board = chess.Board(fen)
        
        # Convert board to tensor
        board_tensor = ChessBoardEncoder.fen_to_tensor(fen)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get model predictions (logits)
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
        
        # Squeeze batch dim
        policy_logits = policy_logits.squeeze(0) # [4096]
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Create mask for legal moves
        # We need to map each legal move to its index
        from .move_encoding import MoveLabeler
        labeler = MoveLabeler()
        
        legal_indices = []
        legal_moves_map = {} # index -> move
        
        for move in legal_moves:
            idx = labeler.encode(move)
            if idx is not None:
                legal_indices.append(idx)
                legal_moves_map[idx] = move
        
        if not legal_indices:
            # Fallback if encoding fails (shouldn't happen with standard moves)
            import random
            return str(random.choice(legal_moves))
            
        # Mask illegal moves (set to -inf)
        mask = torch.ones_like(policy_logits) * float('-inf')
        mask[legal_indices] = 0
        
        masked_logits = policy_logits + mask
        
        # Apply softmax to get probabilities
        probs = F.softmax(masked_logits, dim=0)
        
        # Sample from the distribution (stochastic)
        # Or use argmax for deterministic play
        try:
            move_idx = torch.multinomial(probs, 1).item()
            best_move = legal_moves_map.get(move_idx)
        except:
            # Fallback
            best_move = legal_moves[0]
            
        return str(best_move)
    
    def evaluate_position(self, fen):
        """
        Evaluate a chess position
        
        Args:
            fen: Position in FEN notation
            
        Returns:
            float: Evaluation score (-1 to +1)
                  Negative = black is winning
                  Positive = white is winning
        """
        board_tensor = ChessBoardEncoder.fen_to_tensor(fen)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.model(board_tensor)
        
        return value.item() * 100  # Scale to centipawns like Stockfish


# Test the model
if __name__ == "__main__":
    print("Testing Chess RL Network...")
    
    # Create model
    model = ChessRLNetwork()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 12, 8, 8)
    policy, value = model(dummy_input)
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item()}")
    
    # Test encoder
    encoder = ChessBoardEncoder()
    starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = encoder.fen_to_tensor(starting_position)
    print(f"Board tensor shape: {tensor.shape}")
    
    # Test engine
    engine = ChessRLEngine()
    move = engine.get_best_move(starting_position)
    evaluation = engine.evaluate_position(starting_position)
    print(f"Best move: {move}")
    print(f"Evaluation: {evaluation}")
    
    print("\n✅ RL Chess Engine initialized successfully!")
    print("Ready to train and improve! 🚀")
