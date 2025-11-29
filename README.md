# ♟️ Chess AI Engine & Fullstack Application

![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?style=for-the-badge&logo=Flutter&logoColor=white)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Dart](https://img.shields.io/badge/dart-%230175C2.svg?style=for-the-badge&logo=dart&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

A powerful, real-time multiplayer chess platform featuring a **Hybrid AI Engine** that combines traditional Stockfish analysis with a custom Deep Reinforcement Learning model. Built with a robust Django backend and a sleek Flutter frontend.

## ✨ Key Features

### 🧠 Advanced AI Engine
- **Hybrid Architecture**: Seamlessly switches between Stockfish and custom RL models.
- **Deep Reinforcement Learning**: Custom-trained neural network (`ai_engine/rl_model`) capable of self-play and robust training.
- **Watch Mode**: Visualize AI decision-making and training progress in real-time.
- **Difficulty Levels**: Adaptive difficulty settings for players of all skill levels.

### 🎮 Immersive Gameplay
- **Real-time Multiplayer**: Low-latency WebSocket connections for seamless online play.
- **Cross-Platform**: Fully responsive Flutter app for Mobile (iOS/Android), Web, and Desktop.
- **Interactive Board**: Smooth drag-and-drop mechanics with legal move highlighting.
- **Game History**: Detailed move logs and historical game analysis.

### 🛡️ Secure & Scalable
- **Authentication**: Secure JWT-based user registration and login.
- **Profile Management**: Track ELO ratings, win/loss stats, and match history.
- **Production Ready**: Configured for deployment on Render with Docker support.

---

## 🚀 Quick Start Guide

### 1️⃣ Backend Setup (Django)

```bash
# Clone the repository
git clone https://github.com/MohammedAbdulwahab3/Chess-Ai-Engine.git
cd Chess-Ai-Engine

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start the server
python manage.py runserver
```
*Backend runs at `http://localhost:8000`*

### 2️⃣ Frontend Setup (Flutter)

```bash
cd chess_flutter

# Install dependencies
flutter pub get

# Run the application
flutter run
```

---

## � Project Architecture

### Backend Structure (`/`)
| Directory | Description |
|-----------|-------------|
| `ai_engine/` | **Core AI Logic**: Contains `rl_model` (Neural Networks), `stockfish_engine.py`, and training scripts. |
| `game/` | **Game Logic**: Handles move validation, board state, and WebSocket consumers. |
| `accounts/` | **User Management**: Auth views, models, and JWT configuration. |
| `chess_backend/` | **Project Settings**: URL routing, ASGI/WSGI config. |

### Frontend Structure (`chess_flutter/lib/`)
| Directory | Description |
|-----------|-------------|
| `screens/` | UI Screens: `game`, `auth`, `home`, `watch_model`. |
| `providers/` | State Management (Riverpod). |
| `data/` | API repositories and data models. |
| `core/` | Constants, API clients, and utilities. |

---

## 🛠️ Tech Stack

- **Backend**: Django 4.2, Django REST Framework, Django Channels (WebSockets), Redis (optional for prod).
- **AI/ML**: PyTorch, NumPy, Stockfish Binary.
- **Frontend**: Flutter 3.x, Riverpod, flutter_chess_board.
- **Database**: SQLite (Dev), PostgreSQL (Prod).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
