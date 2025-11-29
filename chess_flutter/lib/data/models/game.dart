import 'user.dart';

class Game {
  final int id;
  final User? whitePlayer;
  final User? blackPlayer;
  final String gameMode; // 'pvp' or 'ai'
  final String? aiDifficulty;
  final String? aiColor;
  final String boardState; // FEN notation
  final String currentTurn; // 'white' or 'black'
  final String status; // 'waiting', 'active', 'completed', 'abandoned'
  final String result; // 'white_wins', 'black_wins', 'draw', 'ongoing'
  final DateTime createdAt;
  final DateTime updatedAt;
  final DateTime? completedAt;
  final List<Move>? moves;

  Game({
    required this.id,
    this.whitePlayer,
    this.blackPlayer,
    required this.gameMode,
    this.aiDifficulty,
    this.aiColor,
    required this.boardState,
    required this.currentTurn,
    required this.status,
    required this.result,
    required this.createdAt,
    required this.updatedAt,
    this.completedAt,
    this.moves,
  });

  factory Game.fromJson(Map<String, dynamic> json) {
    return Game(
      id: json['id'],
      whitePlayer: json['white_player'] != null ? User.fromJson(json['white_player']) : null,
      blackPlayer: json['black_player'] != null ? User.fromJson(json['black_player']) : null,
      gameMode: json['game_mode'],
      aiDifficulty: json['ai_difficulty'],
      aiColor: json['ai_color'],
      boardState: json['board_state'],
      currentTurn: json['current_turn'],
      status: json['status'],
      result: json['result'],
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
      completedAt: json['completed_at'] != null ? DateTime.parse(json['completed_at']) : null,
      moves: json['moves'] != null
          ? (json['moves'] as List).map((m) => Move.fromJson(m)).toList()
          : null,
    );
  }

  bool get isAiGame => gameMode == 'ai';
  bool get isActive => status == 'active';
  bool get isCompleted => status == 'completed';
}

class Move {
  final int id;
  final User? player;
  final bool isAiMove;
  final String moveNotation; // UCI format
  final String sanNotation; // Standard Algebraic Notation
  final String fenAfterMove;
  final int moveNumber;
  final DateTime timestamp;

  Move({
    required this.id,
    this.player,
    required this.isAiMove,
    required this.moveNotation,
    required this.sanNotation,
    required this.fenAfterMove,
    required this.moveNumber,
    required this.timestamp,
  });

  factory Move.fromJson(Map<String, dynamic> json) {
    return Move(
      id: json['id'],
      player: json['player'] != null ? User.fromJson(json['player']) : null,
      isAiMove: json['is_ai_move'],
      moveNotation: json['move_notation'],
      sanNotation: json['san_notation'],
      fenAfterMove: json['fen_after_move'],
      moveNumber: json['move_number'],
      timestamp: DateTime.parse(json['timestamp']),
    );
  }
}
