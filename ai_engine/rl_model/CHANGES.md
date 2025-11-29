# Changes Made to Improve RL Training

## Issue
Model was losing 98% of games against medium Stockfish with 0 wins after 100 games.

## Improvements Made

### 1. Larger Model (chess_network.py)
- **Before**: 128 channels, 4 residual blocks
- **After**: 256 channels, 8 residual blocks
- **Why**: More capacity to learn complex chess patterns

### 2. Curriculum Learning (train_robust.py)
- **Before**: Training against medium Stockfish from start
- **After**: 
  - Games 1-300: Easy Stockfish
  - Games 301-500: Medium Stockfish
- **Why**: Gradual difficulty increase allows model to learn basics first

### 3. More Training Games
- **Before**: 100 RL games
- **After**: 500 RL games with curriculum
- **Why**: More experience = better learning

### 4. More Frequent Training
- **Before**: 1 training step per game
- **After**: 3 training steps per game
- **Why**: Better utilization of collected data

### 5. Temperature Annealing
- **Before**: Fixed temperature=1.0
- **After**: Starts at 1.0, decreases to 0.5 over training
- **Why**: Initial exploration, then exploitation

### 6. Higher Learning Rate
- **Before**: 1e-4
- **After**: 3e-4
- **Why**: Faster convergence on CPU

### 7. Better Evaluation
- **After**: Evaluate vs both Easy and Medium Stockfish separately
- **Why**: See progress at different difficulty levels

## Expected Results

After these changes:
- **vs Easy Stockfish**: 30-50% win rate
- **vs Medium Stockfish**: 5-15% win rate, 10-20% draws
- **Imitation loss**: Should still drop to ~0.10
- **Policy loss**: Should stabilize around 0.3-0.5
- **Value loss**: Should decrease over training

## Files Modified
1. `train_robust.py`: All improvements
2. No other files needed changes

Run with:
```bash
python -m ai_engine.rl_model.train_robust
```
