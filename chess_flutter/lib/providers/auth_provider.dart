import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../data/models/user.dart';
import '../data/repositories/auth_repository.dart';

// Auth Repository Provider
final authRepositoryProvider = Provider((ref) => AuthRepository());

// Auth State Provider
final authStateProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier(ref.read(authRepositoryProvider));
});

class AuthState {
  final User? user;
  final bool isLoading;
  final String? error;
  final bool isAuthenticated;

  AuthState({
    this.user,
    this.isLoading = false,
    this.error,
    this.isAuthenticated = false,
  });

  AuthState copyWith({
    User? user,
    bool? isLoading,
    String? error,
    bool? isAuthenticated,
  }) {
    return AuthState(
      user: user ?? this.user,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
    );
  }
}

class AuthNotifier extends StateNotifier<AuthState> {
  final AuthRepository _authRepository;

  AuthNotifier(this._authRepository) : super(AuthState()) {
    _checkAuth();
  }

  Future<void> _checkAuth() async {
    await _authRepository.loadTokens();
    if (_authRepository.isAuthenticated) {
      try {
        final user = await _authRepository.getProfile();
        state = state.copyWith(
          user: user,
          isAuthenticated: true,
        );
      } catch (e) {
        // Token might be expired
        await _authRepository.logout();
      }
    }
  }

  Future<void> register({
    required String username,
    required String email,
    required String password,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final response = await _authRepository.register(
        username: username,
        email: email,
        password: password,
      );
      
      // Auto-login after registration
      await login(username: username, password: password);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  Future<void> login({
    required String username,
    required String password,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final response = await _authRepository.login(
        username: username,
        password: password,
      );
      
      final user = User.fromJson(response['user']);
      state = state.copyWith(
        user: user,
        isAuthenticated: true,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  Future<void> logout() async {
    await _authRepository.logout();
    state = AuthState();
  }

  Future<void> refreshProfile() async {
    try {
      final user = await _authRepository.getProfile();
      state = state.copyWith(user: user);
    } catch (e) {
      // Handle error
    }
  }
}
