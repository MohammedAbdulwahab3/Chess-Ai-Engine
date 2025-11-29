import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_chess_board/flutter_chess_board.dart';
import 'dart:async';
import 'dart:math' show atan2, cos, sin;
import '../../data/repositories/game_repository.dart';
import '../../providers/game_provider.dart';

/// Screen to WATCH your trained RL model play against Stockfish!
/// This is like spectator mode - you just watch the game unfold.
class WatchModelPlayScreen extends ConsumerStatefulWidget {
  const WatchModelPlayScreen({super.key});

  @override
  ConsumerState<WatchModelPlayScreen> createState() => _WatchModelPlayScreenState();
}

class _WatchModelPlayScreenState extends ConsumerState<WatchModelPlayScreen> {
  late ChessBoardController _boardController;
  bool _isPlaying = false;
  bool _isPaused = false;
  int _moveCount = 0;
  String _currentStatus = "Ready to watch";
  Timer? _gameTimer;
  double _gameSpeed = 1.0; // Moves per second

  @override
  void initState() {
    super.initState();
    _boardController = ChessBoardController();
  }

  @override
  void dispose() {
    _gameTimer?.cancel();
    _boardController.dispose();
    super.dispose();
  }

  Future<void> _startWatching() async {
    setState(() {
      _isPlaying = true;
      _isPaused = false;
      _moveCount = 0;
      _currentStatus = "Creating game...";
    });

    try {
      // Create a new game: Your model vs Stockfish
      await ref.read(currentGameProvider.notifier).createGame(
        gameMode: 'ai',
        aiDifficulty: 'medium',
        aiColor: 'black', // Your model plays white
      );

      setState(() {
        _currentStatus = "Game started! Watching your model play...";
      });

      // Start the automated game loop
      _runGameLoop();
    } catch (e) {
      setState(() {
        _currentStatus = "Error: $e";
        _isPlaying = false;
      });
    }
  }

  void _runGameLoop() {
    _gameTimer = Timer.periodic(Duration(milliseconds: (1000 / _gameSpeed).round()), (timer) async {
      if (_isPaused) return;

      final gameState = ref.read(currentGameProvider);
      final game = gameState.game;

      if (game == null || !game.isActive) {
        timer.cancel();
        setState(() {
          _isPlaying = false;
          _currentStatus = game?.status == 'completed' 
              ? "Game Over! ${game?.result ?? 'Unknown'}"
              : "Game stopped";
        });
        return;
      }

      // Update board
      _boardController.loadFen(game.boardState);

      try {
        // Determine whose turn and make move
        if (game.currentTurn == 'white') {
          // Your model's turn (it plays white)
          setState(() {
            _currentStatus = "🤖 Your Model is thinking...";
          });
          
          // Get hint from your model (this uses your RL model!)
          final repository = ref.read(gameRepositoryProvider);
          final hint = await repository.getHint(game.id);
          final move = hint['hint_move'];
          
          // Make the move
          await ref.read(currentGameProvider.notifier).makeMove(move);
          
          setState(() {
            _moveCount++;
            _currentStatus = "🤖 Your Model played: $move";
          });
        } else {
          // Stockfish's turn (it plays black)
          setState(() {
            _currentStatus = "⚙️ Stockfish is thinking...";
          });
          
          // Let Stockfish make its move
          await ref.read(currentGameProvider.notifier).requestAiMove();
          
          setState(() {
            _moveCount++;
            _currentStatus = "⚙️ Stockfish played";
          });
        }
      } catch (e) {
        timer.cancel();
        setState(() {
          _isPlaying = false;
          _currentStatus = "Error: $e";
        });
      }
    });
  }

  void _togglePause() {
    setState(() {
      _isPaused = !_isPaused;
    });
  }

  void _stopGame() {
    _gameTimer?.cancel();
    setState(() {
      _isPlaying = false;
      _isPaused = false;
      _currentStatus = "Stopped";
    });
  }

  @override
  Widget build(BuildContext context) {
    final gameState = ref.watch(currentGameProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Watch Your Model Play'),
        actions: [
          IconButton(
            icon: Icon(_isPaused ? Icons.play_arrow : Icons.pause),
            onPressed: _isPlaying ? _togglePause : null,
          ),
          IconButton(
            icon: const Icon(Icons.stop),
            onPressed: _isPlaying ? _stopGame : null,
          ),
        ],
      ),
      body: Column(
        children: [
          // Status Card
          Card(
            margin: const EdgeInsets.all(16),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Text(
                    _currentStatus,
                    style: Theme.of(context).textTheme.titleMedium,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Moves: $_moveCount',
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('Speed: '),
                      Slider(
                        value: _gameSpeed,
                        min: 0.5,
                        max: 3.0,
                        divisions: 5,
                        label: '${_gameSpeed.toStringAsFixed(1)}x',
                        onChanged: (value) {
                          setState(() {
                            _gameSpeed = value;
                          });
                          if (_isPlaying) {
                            _gameTimer?.cancel();
                            _runGameLoop();
                          }
                        },
                      ),
                      Text('${_gameSpeed.toStringAsFixed(1)}x'),
                    ],
                  ),
                ],
              ),
            ),
          ),

          // Chess Board
          Expanded(
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: AspectRatio(
                  aspectRatio: 1.0,
                  child: Stack(
                    children: [
                      ChessBoard(
                        controller: _boardController,
                        boardColor: BoardColor.brown,
                        boardOrientation: PlayerColor.white,
                        enableUserMoves: false, // Spectator mode!
                      ),
                      
                      // Board coordinates
                      Positioned.fill(
                        child: Padding(
                          padding: const EdgeInsets.all(4.0),
                          child: _BoardCoordinates(),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          // Control Buttons
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                if (!_isPlaying)
                  ElevatedButton.icon(
                    onPressed: _startWatching,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Start Watching'),
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(200, 50),
                    ),
                  ),
                if (_isPlaying) ...[
                  ElevatedButton.icon(
                    onPressed: _togglePause,
                    icon: Icon(_isPaused ? Icons.play_arrow : Icons.pause),
                    label: Text(_isPaused ? 'Resume' : 'Pause'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _stopGame,
                    icon: const Icon(Icons.stop),
                    label: const Text('Stop'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red.shade700,
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Info
          if (!_isPlaying)
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Card(
                color: Colors.blue.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '🤖 How it works:',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text('• Your RL Model plays WHITE'),
                      const Text('• Stockfish plays BLACK'),
                      const Text('• Moves happen automatically'),
                      const Text('• Adjust speed with slider'),
                      const Text('• Pause/resume anytime'),
                      const SizedBox(height: 8),
                      const Text(
                        'This lets you see how your trained model performs!',
                        style: TextStyle(fontStyle: FontStyle.italic),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
     ),
    );
  }
}

// Board coordinates widget (same as game screen)
class _BoardCoordinates extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Files (a-h) at bottom
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Row(
            children: List.generate(8, (i) {
              return Expanded(
                child: Center(
                  child: Text(
                    String.fromCharCode('a'.codeUnitAt(0) + i),
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.8),
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      shadows: [
                        Shadow(
                          color: Colors.black,
                          blurRadius: 2,
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }),
          ),
        ),
        
        // Ranks (1-8) on left
        Positioned(
          top: 0,
          bottom: 0,
          left: 0,
          child: Column(
            children: List.generate(8, (i) {
              return Expanded(
                child: Center(
                  child: Text(
                    '${8 - i}',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.8),
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      shadows: [
                        Shadow(
                          color: Colors.black,
                          blurRadius: 2,
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }),
          ),
        ),
      ],
    );
  }
}
