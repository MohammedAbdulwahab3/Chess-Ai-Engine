class ApiConstants {
  // Base URL - Change this to your backend URL
  static const String baseUrl = 'http://localhost:8000';
  static const String wsUrl = 'ws://localhost:8000';
  
  // Auth endpoints
  static const String register = '/api/auth/register/';
  static const String login = '/api/auth/login/';
  static const String profile = '/api/auth/profile/';
  static const String tokenRefresh = '/api/auth/token/refresh/';
  
  // Game endpoints
  static const String games = '/api/games/';
  static const String availableGames = '/api/games/available/';
  
  // AI endpoints
  static String aiMove(int gameId) => '/api/ai/games/$gameId/ai-move/';
  static String getHint(int gameId) => '/api/ai/games/$gameId/hint/';
  
  // Game specific endpoints
  static String gameDetail(int gameId) => '/api/games/$gameId/';
  static String joinGame(int gameId) => '/api/games/$gameId/join/';
  static String makeMove(int gameId) => '/api/games/$gameId/move/';
  static String gameStatus(int gameId) => '/api/games/$gameId/status/';
  
  // WebSocket
  static String gameWebSocket(int gameId) => '$wsUrl/ws/game/$gameId/';
}
