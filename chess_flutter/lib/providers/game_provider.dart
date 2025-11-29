import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../data/models/game.dart';
import '../data/repositories/game_repository.dart';

// Game Repository Provider
final gameRepositoryProvider = Provider((ref) => GameRepository());

// Current Game Provider
final currentGameProvider = StateNotifierProvider<CurrentGameNotifier, GameState>((ref) {
  return CurrentGameNotifier(ref.read(gameRepositoryProvider));
});

// My Games List Provider
final myGamesProvider = FutureProvider<List<Game>>((ref) async {
  final repository = ref.read(gameRepositoryProvider);
  return await repository.getMyGames();
});

// Available Games Provider
final availableGamesProvider = FutureProvider<List<Game>>((ref) async {
  final repository = ref.read(gameRepositoryProvider);
  return await repository.getAvailableGames();
});

class GameState {
  final Game? game;
  final bool isLoading;
  final String? error;
  final bool isMyTurn;

  GameState({
    this.game,
    this.isLoading = false,
    this.error,
    this.isMyTurn = false,
  });

  GameState copyWith({
    Game? game,
    bool? isLoading,
    String? error,
    bool? isMyTurn,
  }) {
    return GameState(
      game: game ?? this.game,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      isMyTurn: isMyTurn ?? this.isMyTurn,
    );
  }
}

class CurrentGameNotifier extends StateNotifier<GameState> {
  final GameRepository _gameRepository;

  CurrentGameNotifier(this._gameRepository) : super(GameState());

  Future<void> createGame({
    required String gameMode,
    String? aiDifficulty,
    String? aiColor,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final game = await _gameRepository.createGame(
        gameMode: gameMode,
        aiDifficulty: aiDifficulty,
        aiColor: aiColor,
      );
      state = state.copyWith(
        game: game,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  Future<void> loadGame(int gameId) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final game = await _gameRepository.getGame(gameId);
      state = state.copyWith(
        game: game,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  Future<void> joinGame(int gameId) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final game = await _gameRepository.joinGame(gameId);
      state = state.copyWith(
        game: game,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  Future<void> makeMove(String move) async {
    if (state.game == null) return;
    
    try {
      final response = await _gameRepository.makeMove(state.game!.id, move);
      final updatedGame = Game.fromJson(response['game']);
      state = state.copyWith(game: updatedGame);
      
      // If it's an AI game and now it's AI's turn, get AI move
      if (updatedGame.isAiGame && updatedGame.isActive) {
        await _getAiMove();
      }
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  Future<void> requestAiMove() async {
    await _getAiMove();
  }

  Future<void> _getAiMove() async {
    if (state.game == null) return;
    
    try {
      final response = await _gameRepository.getAiMove(state.game!.id);
      // Update game state with AI move
      final updatedGame = state.game!;
      state = state.copyWith(game: updatedGame);
      
      // Reload game to get updated state
      await loadGame(state.game!.id);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  void updateGame(Game game) {
    state = state.copyWith(game: game);
  }

  void clearGame() {
    state = GameState();
  }
}
