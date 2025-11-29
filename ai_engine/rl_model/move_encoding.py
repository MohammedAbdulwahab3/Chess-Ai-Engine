"""
Move Encoding/Decoding for Chess RL

Maps chess moves to integer indices for the neural network policy head.
Supports standard moves and promotions.
"""

import chess

class MoveLabeler:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = {}
        self.vocab_size = 0
        self._generate_vocabulary()
        
    def _generate_vocabulary(self):
        """
        Generates vocabulary:
        0-4095: Standard moves (from_sq * 64 + to_sq)
        4096+: Promotions
        """
        # Standard moves (from_sq * 64 + to_sq)
        # This covers all from-to combinations, including illegal ones (simplifies logic)
        self.vocab_size = 4096
        
        # Promotions
        # We need to map (from_sq, to_sq, promotion_piece)
        # Only pawns promote, from rank 7->8 (white) or 2->1 (black).
        # But to keep it simple and cover all potential promotions:
        # We'll map promotions to a separate range.
        # Structure: 4096 + (from_file * 3 + direction) * 4 + promo_type
        # But the user suggested: 4096 + base*4 + promo_id
        
        # Let's use a simpler dense mapping for promotions to avoid huge sparse gaps if we used from*64+to
        # There are only limited promotion moves.
        # White: from rank 7 (8-15) to rank 8 (0-7)
        # Black: from rank 2 (48-55) to rank 1 (56-63)
        # Actually, let's just use a separate dictionary for promotions to keep indices compact if possible,
        # OR just append them.
        
        # User requested: "Handle promotions by reserving a promotion region"
        # Let's define:
        # Indices 0-4095: Standard moves (including castling, en passant)
        # Indices 4096-4200+: Promotions
        
        # Let's iterate all possible promotion moves to assign indices
        idx = 4096
        self.promotion_map = {} # (from, to, promo) -> idx
        self.idx_to_promotion = {} # idx -> (from, to, promo)
        
        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        
        # White promotions (Rank 7 -> 8)
        for from_file in range(8):
            from_sq = chess.square(from_file, 6) # Rank 7
            for to_file in range(8):
                if abs(from_file - to_file) <= 1: # Move or capture
                    to_sq = chess.square(to_file, 7) # Rank 8
                    for piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion=piece)
                        self.move_to_idx[self._move_key(move)] = idx
                        self.idx_to_move[idx] = move
                        idx += 1
                        
        # Black promotions (Rank 2 -> 1)
        for from_file in range(8):
            from_sq = chess.square(from_file, 1) # Rank 2
            for to_file in range(8):
                if abs(from_file - to_file) <= 1:
                    to_sq = chess.square(to_file, 0) # Rank 1
                    for piece in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion=piece)
                        self.move_to_idx[self._move_key(move)] = idx
                        self.idx_to_move[idx] = move
                        idx += 1
                        
        self.vocab_size = idx
        
    def _move_key(self, move):
        return (move.from_square, move.to_square, move.promotion)

    def encode(self, move: chess.Move) -> int:
        """Convert a chess.Move to an integer index"""
        if move.promotion:
            return self.move_to_idx.get(self._move_key(move))
        else:
            # Standard move
            return move.from_square * 64 + move.to_square

    def decode(self, idx: int) -> chess.Move:
        """Convert an integer index back to a chess.Move"""
        if idx < 4096:
            # Standard move
            from_sq = idx // 64
            to_sq = idx % 64
            return chess.Move(from_sq, to_sq)
        else:
            # Promotion
            return self.idx_to_move.get(idx)
            
    def get_vocab_size(self):
        return self.vocab_size
