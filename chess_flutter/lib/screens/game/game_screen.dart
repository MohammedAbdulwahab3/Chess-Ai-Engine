import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_chess_board/flutter_chess_board.dart';
import '../../providers/game_provider.dart';
import 'dart:math' show atan2, cos, sin;

class GameScreen extends ConsumerStatefulWidget {
  const GameScreen({super.key});

  @override
  ConsumerState<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends ConsumerState<GameScreen> {
  late ChessBoardController _boardController;
  bool _isProcessingMove = false;
  String? _hintMove;
  String? _hintEvaluation;

  @override
  void initState() {
    super.initState();
    _boardController = ChessBoardController();
  }

  @override
  void dispose() {
    _boardController.dispose();
    super.dispose();
  }

  Future<void> _getHint() async {
    final gameState = ref.read(currentGameProvider);
    if (gameState.game == null) return;

    try {
      final repository = ref.read(gameRepositoryProvider);
      final hint = await repository.getHint(gameState.game!.id);
      
      setState(() {
        _hintMove = hint['hint_move'];
        _hintEvaluation = hint['evaluation_text'];
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Hint: Move from ${hint['from_square']} to ${hint['to_square']}\n${hint['evaluation_text']}'),
            duration: const Duration(seconds: 5),
            action: SnackBarAction(
              label: 'Clear',
              onPressed: () {
                setState(() {
                  _hintMove = null;
                  _hintEvaluation = null;
                });
              },
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not get hint: $e')),
        );
      }
    }
  }

  void _onMove() {
    if (_isProcessingMove) return;
    
    setState(() {
      _isProcessingMove = true;
    });

    final gameState = ref.read(currentGameProvider);
    if (gameState.game != null) {
      final currentFen = _boardController.getFen();
      final gameFen = gameState.game!.boardState;
      
      if (currentFen != gameFen) {
        // A move was made
        // Get the SAN history from the board
        final sanHistory = _boardController.getSan();
        
        if (sanHistory.isNotEmpty) {
          // Get the last move in SAN format
          final lastSan = sanHistory.last;
          
          // Convert SAN to UCI using the board's internal game
          // We need to get the actual move from the game history
          final gameHistory = _boardController.game.history;
          
          if (gameHistory.isNotEmpty) {
            final lastMoveData = gameHistory.last;
            
            // Try to extract UCI from the move data
            // The move should have 'from' and 'to' as algebraic strings
            String moveUci = '';
            
            // Check if the move has algebraic notation directly
            final from = lastMoveData.move.fromAlgebraic;
            final to = lastMoveData.move.toAlgebraic;
            if (from != null && to != null) {
              moveUci = '$from$to';
              final promotion = lastMoveData.move.promotion;
              if (promotion != null) {
                moveUci += promotion.toString().toLowerCase();
              }
            } else {
              // Fallback: parse from SAN (send sanitized SAN to backend)
              final sanitizedSan = (lastSan ?? '').toLowerCase().replaceAll(RegExp(r'[+#x]'), '');
              moveUci = sanitizedSan;
            }
            
            print('Sending move: $moveUci (from SAN: ${lastSan ?? ''})'); // Debug
            
            // Send move to backend
            ref.read(currentGameProvider.notifier).makeMove(moveUci).then((_) {
              setState(() {
                _isProcessingMove = false;
              });
            }).catchError((error) {
              // Revert the move on error
              _boardController.loadFen(gameFen);
              setState(() {
                _isProcessingMove = false;
              });
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Invalid move: ${error.toString()}')),
              );
            });
          }
        } else {
          setState(() {
            _isProcessingMove = false;
          });
        }
      } else {
        setState(() {
          _isProcessingMove = false;
        });
      }
    }
  }

  String _squareToAlgebraic(int square) {
    // The chess package uses 0x88 board representation
    // Extract file (0-7) and rank (0-7) from the square
    final file = square & 7;  // Get lower 3 bits for file
    final rank = square >> 4; // Get upper bits for rank
    final files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    return '${files[file]}${rank + 1}';
  }

  @override
  Widget build(BuildContext context) {
    final gameState = ref.watch(currentGameProvider);
    final game = gameState.game;

    if (game == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Game')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    // Update board state when game changes from backend
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && game.boardState != _boardController.getFen() && !_isProcessingMove) {
        _boardController.loadFen(game.boardState);
      }
    });

    return Scaffold(
      appBar: AppBar(
        title: Text(
          game.isAiGame
              ? 'vs AI (${game.aiDifficulty})'
              : 'vs ${game.blackPlayer?.username ?? "Opponent"}',
        ),
        actions: [
          if (game.isActive)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Chip(
                label: Text(
                  '${game.currentTurn == "white" ? "White" : "Black"} to move',
                ),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // Game Status
          if (game.isCompleted)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              color: Theme.of(context).colorScheme.primaryContainer,
              child: Text(
                _getGameResultText(game.result),
                style: Theme.of(context).textTheme.titleLarge,
                textAlign: TextAlign.center,
              ),
            ),

          // Chess Board with coordinates and hint arrows
          Expanded(
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: AspectRatio(
                  aspectRatio: 1.0,
                  child: Stack(
                    children: [
                      // The chess board
                      ChessBoard(
                        controller: _boardController,
                        boardColor: BoardColor.brown,
                        boardOrientation: PlayerColor.white,
                        enableUserMoves: game.isActive && !_isProcessingMove,
                      ),
                      
                      // Hint arrow overlay
                      if (_hintMove != null)
                        Positioned.fill(
                          child: CustomPaint(
                            painter: HintArrowPainter(
                              from: _hintMove!.substring(0, 2),
                              to: _hintMove!.substring(2, 4),
                            ),
                          ),
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

          // Move History
          Container(
            height: 120,
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Moves',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: game.moves != null && game.moves!.isNotEmpty
                      ? ListView.builder(
                          scrollDirection: Axis.horizontal,
                          itemCount: game.moves!.length,
                          itemBuilder: (context, index) {
                            final move = game.moves![index];
                            return Padding(
                              padding: const EdgeInsets.only(right: 8.0),
                              child: Chip(
                                label: Text(
                                  '${move.moveNumber}. ${move.sanNotation}',
                                ),
                              ),
                            );
                          },
                        )
                      : const Text('No moves yet'),
                ),
              ],
            ),
          ),

          // Make Move Button (for testing)
          if (game.isActive && !_isProcessingMove)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _onMove,
                      child: const Text('Confirm Move'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton.icon(
                    onPressed: _getHint,
                    icon: const Icon(Icons.lightbulb_outline),
                    label: const Text('Hint'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.amber.shade700,
                    ),
                  ),
                ],
              ),
            ),

          // Error Display
          if (gameState.error != null)
            Container(
              padding: const EdgeInsets.all(16),
              color: Theme.of(context).colorScheme.errorContainer,
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      gameState.error!,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onErrorContainer,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () {
                      // Clear error
                    },
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  String _getGameResultText(String result) {
    switch (result) {
      case 'white_wins':
        return '🏆 White Wins!';
      case 'black_wins':
        return '🏆 Black Wins!';
      case 'draw':
        return '🤝 Draw!';
      default:
        return 'Game in progress';
    }
  }
}

// Custom painter for hint arrows
class HintArrowPainter extends CustomPainter {
  final String from;
  final String to;

  HintArrowPainter({required this.from, required this.to});

  @override
  void paint(Canvas canvas, Size size) {
    final squareSize = size.width / 8;
    
    // Parse from/to squares (e.g., "e2" -> col=4, row=6)
    final fromCol = from.codeUnitAt(0) - 'a'.codeUnitAt(0);
    final fromRow = 7 - (int.parse(from[1]) - 1);
    final toCol = to.codeUnitAt(0) - 'a'.codeUnitAt(0);
    final toRow = 7 - (int.parse(to[1]) - 1);
    
    // Calculate center points
    final fromX = (fromCol + 0.5) * squareSize;
    final fromY = (fromRow + 0.5) * squareSize;
    final toX = (toCol + 0.5) * squareSize;
    final toY = (toRow + 0.5) * squareSize;
    
    // Draw arrow
    final paint = Paint()
      ..color = Colors.green.withOpacity(0.7)
      ..strokeWidth = 8
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    
    // Arrow line
    canvas.drawLine(
      Offset(fromX, fromY),
      Offset(toX, toY),
      paint,
    );
    
    // Arrow head
    final arrowPaint = Paint()
      ..color = Colors.green.withOpacity(0.7)
      ..style = PaintingStyle.fill;
    
    final angle = atan2(toY - fromY, toX - fromX);
    const arrowSize = 20.0;
    
    final path = Path();
    path.moveTo(toX, toY);
    path.lineTo(
      toX - arrowSize * cos(angle - 0.5),
      toY - arrowSize * sin(angle - 0.5),
    );
    path.lineTo(
      toX - arrowSize * cos(angle + 0.5),
      toY - arrowSize * sin(angle + 0.5),
    );
    path.close();
    
    canvas.drawPath(path, arrowPaint);
    
    // Highlight from square
    final highlightPaint = Paint()
      ..color = Colors.yellow.withOpacity(0.3)
      ..style = PaintingStyle.fill;
    
    canvas.drawCircle(
      Offset(fromX, fromY),
      squareSize * 0.4,
      highlightPaint,
    );
  }

  @override
  bool shouldRepaint(HintArrowPainter oldDelegate) => true;
}

// Board coordinates widget
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

