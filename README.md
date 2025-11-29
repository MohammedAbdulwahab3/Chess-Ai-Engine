# Fullstack Chess Game - Flutter & Django

## 🎮 Project Overview

A real-time multiplayer chess game with AI opponent support, built with Flutter (frontend) and Django (backend).

### Features
- ✅ User authentication (register, login, JWT)
- ✅ Play against AI (Stockfish engine with 3 difficulty levels)
- ✅ Play online against other players
- ✅ Real-time game updates via WebSockets
- ✅ Game history and statistics
- ✅ ELO rating system
- ✅ Beautiful, modern UI

## 🚀 Quick Start

### Backend Setup (Django)

```bash
cd /home/maw/Desktop/trying

# Activate virtual environment
source venv/bin/activate

# Run migrations (already done)
python manage.py migrate

# Create a superuser (optional)
python manage.py createsuperuser

# Run the development server
python manage.py runserver
```

The backend will be available at `http://localhost:8000`

### Frontend Setup (Flutter)

```bash
cd /home/maw/Desktop/trying/chess_flutter

# Get dependencies (already done)
flutter pub get

# Update API URL in lib/core/constants/api_constants.dart if needed

# Run the app
flutter run
```

## 📁 Project Structure

### Backend (`/home/maw/Desktop/trying/`)
```
├── accounts/           # User authentication
├── game/              # Chess game logic
├── ai_engine/         # Stockfish AI integration
├── chess_backend/     # Django settings
├── requirements.txt   # Python dependencies
├── render.yaml        # Render deployment config
└── build.sh          # Build script for deployment
```

### Frontend (`/home/maw/Desktop/trying/chess_flutter/`)
```
├── lib/
│   ├── core/          # API client, constants
│   ├── data/          # Models, repositories
│   ├── providers/     # Riverpod state management
│   ├── screens/       # UI screens
│   └── main.dart      # App entry point
└── pubspec.yaml       # Flutter dependencies
```

## 🎯 API Endpoints

### Authentication
- `POST /api/auth/register/` - Register new user
- `POST /api/auth/login/` - Login
- `GET /api/auth/profile/` - Get user profile

### Games
- `GET /api/games/` - List user's games
- `POST /api/games/` - Create new game
- `GET /api/games/{id}/` - Get game details
- `POST /api/games/{id}/join/` - Join a game
- `POST /api/games/{id}/move/` - Make a move
- `GET /api/games/available/` - List available games

### AI
- `POST /api/ai/games/{id}/ai-move/` - Get AI move

### WebSocket
- `ws://localhost:8000/ws/game/{id}/` - Real-time game updates

## 🎮 How to Play

1. **Register/Login**: Create an account or login
2. **Choose Game Mode**:
   - **vs AI**: Select difficulty (easy/medium/hard) and your color
   - **vs Player**: Create a game and wait for opponent, or join an existing game
3. **Play**: Make moves by dragging pieces on the board
4. **View History**: See your past games and statistics

## 🛠️ Technologies Used

### Backend
- Django 4.2
- Django REST Framework
- Django Channels (WebSockets)
- PostgreSQL
- Stockfish (Chess AI)
- JWT Authentication

### Frontend
- Flutter 3.x
- Riverpod (State Management)
- flutter_chess_board
- HTTP & WebSocket clients

## 🚢 Deployment

### Render (Backend)
The project includes `render.yaml` for easy deployment to Render:

1. Push code to GitHub
2. Connect repository to Render
3. Render will automatically:
   - Install dependencies
   - Install Stockfish
   - Run migrations
   - Deploy the app

### Flutter (Frontend)
Build for your target platform:

```bash
# Android
flutter build apk

# iOS
flutter build ios

# Web
flutter build web
```

## 📝 Notes

- Backend uses SQLite for development (change to PostgreSQL for production)
- Update `ApiConstants.baseUrl` in Flutter app to point to your backend
- Stockfish binary must be installed on the server
- Redis is required for WebSocket support in production

## 🎨 Screenshots

The app features:
- Modern dark theme
- Gradient backgrounds
- Smooth animations
- Responsive design
- Real-time updates

## 🤝 Contributing

This is a complete fullstack chess game implementation with all core features working!

## 📄 License

MIT License
