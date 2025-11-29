import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/auth_provider.dart';
import '../../providers/game_provider.dart';
import '../game/game_screen.dart';
import '../watch_model/watch_model_screen.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authStateProvider);
    final user = authState.user;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Chess Game'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              ref.read(authStateProvider.notifier).logout();
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // User Profile Card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    CircleAvatar(
                      radius: 40,
                      child: Text(
                        user?.username.substring(0, 1).toUpperCase() ?? 'U',
                        style: const TextStyle(fontSize: 32),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      user?.username ?? 'User',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        _StatItem(
                          label: 'Rating',
                          value: user?.rating.toString() ?? '0',
                        ),
                        _StatItem(
                          label: 'Games',
                          value: user?.gamesPlayed.toString() ?? '0',
                        ),
                        _StatItem(
                          label: 'Win Rate',
                          value: '${user?.winRate.toStringAsFixed(1) ?? '0'}%',
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            
            // New Game Section
            Text(
              'New Game',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            
            // Play vs AI
            Card(
              child: InkWell(
                onTap: () => _showAiGameDialog(context, ref),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Row(
                    children: [
                      Icon(
                        Icons.smart_toy,
                        size: 48,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Play vs AI',
                              style: Theme.of(context).textTheme.titleLarge,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'Challenge the computer',
                              style: Theme.of(context).textTheme.bodyMedium,
                            ),
                          ],
                        ),
                      ),
                      const Icon(Icons.arrow_forward_ios),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            
            // Play Online
            Card(
              child: InkWell(
                onTap: () => _createOnlineGame(context, ref),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Row(
                    children: [
                      Icon(
                        Icons.people,
                        size: 48,
                        color: Theme.of(context).colorScheme.secondary,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Play Online',
                              style: Theme.of(context).textTheme.titleLarge,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'Find an opponent',
                              style: Theme.of(context).textTheme.bodyMedium,
                            ),
                          ],
                        ),
                      ),
                      const Icon(Icons.arrow_forward_ios),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            
            // Watch Model Play
            Card(
              child: InkWell(
                onTap: () => _watchModelPlay(context),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Row(
                    children: [
                      Icon(
                        Icons.visibility,
                        size: 48,
                        color: Colors.purple.shade600,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Watch Model Play',
                              style: Theme.of(context).textTheme.titleLarge,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'See your trained AI vs Stockfish!',
                              style: Theme.of(context).textTheme.bodyMedium,
                            ),
                          ],
                        ),
                      ),
                      const Icon(Icons.arrow_forward_ios),
                    ],
                  ),
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // My Games
            Text(
              'My Games',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            
            _MyGamesList(),
          ],
        ),
      ),
    );
  }

  void _showAiGameDialog(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (dialogContext) {
        String difficulty = 'medium';
        String color = 'white';

        return StatefulBuilder(
          builder: (context, setState) => AlertDialog(
            title: const Text('Play vs AI'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                DropdownButtonFormField<String>(
                  value: difficulty,
                  decoration: const InputDecoration(labelText: 'Difficulty'),
                  items: const [
                    DropdownMenuItem(value: 'easy', child: Text('Easy')),
                    DropdownMenuItem(value: 'medium', child: Text('Medium')),
                    DropdownMenuItem(value: 'hard', child: Text('Hard')),
                  ],
                  onChanged: (value) => setState(() => difficulty = value!),
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<String>(
                  value: color,
                  decoration: const InputDecoration(labelText: 'Your Color'),
                  items: const [
                    DropdownMenuItem(value: 'white', child: Text('White')),
                    DropdownMenuItem(value: 'black', child: Text('Black')),
                  ],
                  onChanged: (value) => setState(() => color = value!),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(dialogContext),
                child: const Text('Cancel'),
              ),
              ElevatedButton(
                onPressed: () async {
                  Navigator.pop(dialogContext);
                  await ref.read(currentGameProvider.notifier).createGame(
                        gameMode: 'ai',
                        aiDifficulty: difficulty,
                        aiColor: color == 'white' ? 'black' : 'white',
                      );
                  
                  if (context.mounted) {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const GameScreen(),
                      ),
                    );
                  }
                },
                child: const Text('Start Game'),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _createOnlineGame(BuildContext context, WidgetRef ref) async {
    await ref.read(currentGameProvider.notifier).createGame(gameMode: 'pvp');
    
    if (context.mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => const GameScreen(),
        ),
      );
    }
  }

  void _watchModelPlay(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const WatchModelPlayScreen(),
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;

  const _StatItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

class _MyGamesList extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final gamesAsync = ref.watch(myGamesProvider);

    return gamesAsync.when(
      data: (games) {
        if (games.isEmpty) {
          return const Card(
            child: Padding(
              padding: EdgeInsets.all(32.0),
              child: Center(
                child: Text('No games yet. Start a new game!'),
              ),
            ),
          );
        }

        return Column(
          children: games.take(5).map((game) {
            return Card(
              child: ListTile(
                leading: Icon(
                  game.isAiGame ? Icons.smart_toy : Icons.people,
                  size: 32,
                ),
                title: Text(
                  game.isAiGame
                      ? 'vs AI (${game.aiDifficulty})'
                      : 'vs ${game.blackPlayer?.username ?? "Waiting..."}',
                ),
                subtitle: Text(game.status),
                trailing: game.isActive
                    ? const Icon(Icons.play_arrow, color: Colors.green)
                    : null,
                onTap: () {
                  ref.read(currentGameProvider.notifier).loadGame(game.id);
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const GameScreen(),
                    ),
                  );
                },
              ),
            );
          }).toList(),
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (error, stack) => Text('Error: $error'),
    );
  }
}
