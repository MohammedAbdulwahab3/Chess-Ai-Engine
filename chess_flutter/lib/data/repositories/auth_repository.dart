import '../models/user.dart';
import '../../core/api/api_client.dart';
import '../../core/constants/api_constants.dart';

class AuthRepository {
  final ApiClient _apiClient = ApiClient();

  Future<Map<String, dynamic>> register({
    required String username,
    required String email,
    required String password,
  }) async {
    final response = await _apiClient.post(
      ApiConstants.register,
      {
        'username': username,
        'email': email,
        'password': password,
        'password_confirm': password,
      },
      auth: false,
    );
    return response;
  }

  Future<Map<String, dynamic>> login({
    required String username,
    required String password,
  }) async {
    final response = await _apiClient.post(
      ApiConstants.login,
      {
        'username': username,
        'password': password,
      },
      auth: false,
    );
    
    // Save tokens
    if (response['access'] != null && response['refresh'] != null) {
      await _apiClient.saveTokens(response['access'], response['refresh']);
    }
    
    return response;
  }

  Future<void> logout() async {
    await _apiClient.clearTokens();
  }

  Future<User> getProfile() async {
    final response = await _apiClient.get(ApiConstants.profile);
    return User.fromJson(response);
  }

  Future<void> loadTokens() async {
    await _apiClient.loadTokens();
  }

  bool get isAuthenticated => _apiClient.isAuthenticated;
}
