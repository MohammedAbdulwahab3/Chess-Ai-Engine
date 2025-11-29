import '../models/game.dart';
import '../../core/api/api_client.dart';
import '../../core/constants/api_constants.dart';

class GameRepository {
  final ApiClient _apiClient = ApiClient();

  Future<List<Game>> getMyGames() async {
    final response = await _apiClient.get(ApiConstants.games);
    return (response as List).map((json) => Game.fromJson(json)).toList();
  }

  Future<List<Game>> getAvailableGames() async {
    final response = await _apiClient.get(ApiConstants.availableGames);
    return (response as List).map((json) => Game.fromJson(json)).toList();
  }

  Future<Game> getGame(int gameId) async {
    final response = await _apiClient.get(ApiConstants.gameDetail(gameId));
    return Game.fromJson(response);
  }

  Future<Game> createGame({
    required String gameMode,
    String? aiDifficulty,
    String? aiColor,
  }) async {
    final data = {
      'game_mode': gameMode,
      if (aiDifficulty != null) 'ai_difficulty': aiDifficulty,
      if (aiColor != null) 'ai_color': aiColor,
    };
    
    final response = await _apiClient.post(ApiConstants.games, data);
    return Game.fromJson(response);
  }

  Future<Game> joinGame(int gameId) async {
    final response = await _apiClient.post(ApiConstants.joinGame(gameId), {});
    return Game.fromJson(response);
  }

  Future<Map<String, dynamic>> makeMove(int gameId, String move) async {
    final response = await _apiClient.post(
      ApiConstants.makeMove(gameId),
      {'move': move},
    );
    return response;
  }

  Future<Map<String, dynamic>> getAiMove(int gameId) async {
    final response = await _apiClient.post(
      ApiConstants.aiMove(gameId),
      {},
    );
    return response;
  }

  Future<Map<String, dynamic>> getGameStatus(int gameId) async {
    final response = await _apiClient.get(ApiConstants.gameStatus(gameId));
    return response;
  }

  Future<Map<String, dynamic>> getHint(int gameId, {String difficulty = 'medium'}) async {
    final response = await _apiClient.post(
      ApiConstants.getHint(gameId),
      {'difficulty': difficulty},
    );
    return response;
  }
}
