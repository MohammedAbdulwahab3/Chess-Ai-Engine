import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/api_constants.dart';

class ApiClient {
  static final ApiClient _instance = ApiClient._internal();
  factory ApiClient() => _instance;
  ApiClient._internal();

  final http.Client _client = http.Client();
  String? _accessToken;
  String? _refreshToken;

  Future<void> loadTokens() async {
    final prefs = await SharedPreferences.getInstance();
    _accessToken = prefs.getString('access_token');
    _refreshToken = prefs.getString('refresh_token');
  }

  Future<void> saveTokens(String access, String refresh) async {
    _accessToken = access;
    _refreshToken = refresh;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('access_token', access);
    await prefs.setString('refresh_token', refresh);
  }

  Future<void> clearTokens() async {
    _accessToken = null;
    _refreshToken = null;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('access_token');
    await prefs.remove('refresh_token');
  }

  Map<String, String> _getHeaders({bool includeAuth = true}) {
    final headers = {
      'Content-Type': 'application/json',
    };
    if (includeAuth && _accessToken != null) {
      headers['Authorization'] = 'Bearer $_accessToken';
    }
    return headers;
  }

  Future<Map<String, dynamic>> get(String endpoint, {bool auth = true}) async {
    final url = Uri.parse('${ApiConstants.baseUrl}$endpoint');
    final response = await _client.get(url, headers: _getHeaders(includeAuth: auth));
    return _handleResponse(response);
  }

  Future<Map<String, dynamic>> post(String endpoint, Map<String, dynamic> data, {bool auth = true}) async {
    final url = Uri.parse('${ApiConstants.baseUrl}$endpoint');
    final response = await _client.post(
      url,
      headers: _getHeaders(includeAuth: auth),
      body: json.encode(data),
    );
    return _handleResponse(response);
  }

  Future<Map<String, dynamic>> put(String endpoint, Map<String, dynamic> data, {bool auth = true}) async {
    final url = Uri.parse('${ApiConstants.baseUrl}$endpoint');
    final response = await _client.put(
      url,
      headers: _getHeaders(includeAuth: auth),
      body: json.encode(data),
    );
    return _handleResponse(response);
  }

  Future<Map<String, dynamic>> delete(String endpoint, {bool auth = true}) async {
    final url = Uri.parse('${ApiConstants.baseUrl}$endpoint');
    final response = await _client.delete(url, headers: _getHeaders(includeAuth: auth));
    return _handleResponse(response);
  }

  Map<String, dynamic> _handleResponse(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) return {};
      return json.decode(response.body);
    } else {
      throw ApiException(
        statusCode: response.statusCode,
        message: response.body.isNotEmpty ? json.decode(response.body) : 'Unknown error',
      );
    }
  }

  bool get isAuthenticated => _accessToken != null;
}

class ApiException implements Exception {
  final int statusCode;
  final dynamic message;

  ApiException({required this.statusCode, required this.message});

  @override
  String toString() => 'ApiException: $statusCode - $message';
}
