class User {
  final int id;
  final String username;
  final String email;
  final int rating;
  final int gamesPlayed;
  final int gamesWon;
  final int gamesLost;
  final int gamesDrawn;
  final double winRate;
  final DateTime createdAt;

  User({
    required this.id,
    required this.username,
    required this.email,
    required this.rating,
    required this.gamesPlayed,
    required this.gamesWon,
    required this.gamesLost,
    required this.gamesDrawn,
    required this.winRate,
    required this.createdAt,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      username: json['username'],
      email: json['email'],
      rating: json['rating'],
      gamesPlayed: json['games_played'],
      gamesWon: json['games_won'],
      gamesLost: json['games_lost'],
      gamesDrawn: json['games_drawn'],
      winRate: (json['win_rate'] as num).toDouble(),
      createdAt: DateTime.parse(json['created_at']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'username': username,
      'email': email,
      'rating': rating,
      'games_played': gamesPlayed,
      'games_won': gamesWon,
      'games_lost': gamesLost,
      'games_drawn': gamesDrawn,
      'win_rate': winRate,
      'created_at': createdAt.toIso8601String(),
    };
  }
}
