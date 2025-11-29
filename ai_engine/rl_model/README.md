# 🧠 Robust Chess RL Engine

This directory contains a complete Reinforcement Learning system for Chess, built from scratch.

## 🌟 Features

- **AlphaZero-style Architecture**: ResNet backbone with Policy and Value heads.
- **Robust Training Pipeline**:
    - **Imitation Learning**: Pre-trains policy to mimic Stockfish (jumpstarts learning).
    - **Reinforcement Learning**: Improves through self-play/play-vs-Stockfish.
    - **Combined Loss**: Optimizes both Policy (CrossEntropy) and Value (MSE).
    - **Replay Buffer**: Stores and samples experiences for stable training.
- **Proper Move Encoding**: Maps UCI moves to network indices (0-4096) with masking for illegal moves.

## 🚀 How to Train

### 1. Robust Training (Recommended)
This runs the full pipeline: Imitation Learning -> RL Training.

```bash
cd /home/maw/Desktop/trying
source venv/bin/activate
python -m ai_engine.rl_model.train_robust
```

### 2. Watch It Learn
You can watch the training progress in the console.
- **Imitation Phase**: Loss should decrease rapidly as it learns legal moves.
- **RL Phase**: Win rate vs Stockfish should slowly increase.

## 📁 File Structure

- `train_robust.py`: **Main training script** (Imitation + RL).
- `chess_network.py`: Neural network architecture & inference logic.
- `move_encoding.py`: Maps chess moves <-> network indices.
- `replay_buffer.py`: Experience replay memory.
- `self_play.py`: Script to watch the model play itself.
- `train_vs_stockfish.py`: Legacy training script (deprecated).

## 🛠️ Architecture Details

- **Input**: 8x8x12 tensor (6 piece types x 2 colors).
- **Backbone**: ResNet (Configurable depth/width).
- **Policy Head**: Outputs logits for 4096 possible moves.
- **Value Head**: Outputs scalar evaluation (-1 to +1).

## 📈 Monitoring

The training script prints:
- **Policy Loss**: How well it predicts the "best" move.
- **Value Loss**: How well it predicts the game outcome.
- **Win Rate**: Performance against Stockfish.

Enjoy building your Chess AI! ♟️🤖
