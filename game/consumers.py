import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import Game, Move
from .chess_logic import ChessLogic


class GameConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time chess game updates"""
    
    async def connect(self):
        self.game_id = self.scope['url_route']['kwargs']['game_id']
        self.game_group_name = f'game_{self.game_id}'
        
        # Join game group
        await self.channel_layer.group_add(
            self.game_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send current game state
        game_data = await self.get_game_data()
        await self.send(text_data=json.dumps({
            'type': 'game_state',
            'data': game_data
        }))
    
    async def disconnect(self, close_code):
        # Leave game group
        await self.channel_layer.group_discard(
            self.game_group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Receive message from WebSocket"""
        data = json.loads(text_data)
        message_type = data.get('type')
        
        if message_type == 'make_move':
            # Handle move
            move_uci = data.get('move')
            user_id = self.scope['user'].id
            
            result = await self.make_move(user_id, move_uci)
            
            if result['success']:
                # Broadcast move to all players in the game
                await self.channel_layer.group_send(
                    self.game_group_name,
                    {
                        'type': 'move_made',
                        'move': result['move'],
                        'game_state': result['game_state']
                    }
                )
            else:
                # Send error to this connection only
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': result['error']
                }))
        
        elif message_type == 'request_game_state':
            game_data = await self.get_game_data()
            await self.send(text_data=json.dumps({
                'type': 'game_state',
                'data': game_data
            }))
    
    async def move_made(self, event):
        """Send move update to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'move_made',
            'move': event['move'],
            'game_state': event['game_state']
        }))
    
    async def player_joined(self, event):
        """Send player joined notification"""
        await self.send(text_data=json.dumps({
            'type': 'player_joined',
            'player': event['player']
        }))
    
    @database_sync_to_async
    def get_game_data(self):
        """Get current game data"""
        from .serializers import GameSerializer
        try:
            game = Game.objects.get(id=self.game_id)
            return GameSerializer(game).data
        except Game.DoesNotExist:
            return None
    
    @database_sync_to_async
    def make_move(self, user_id, move_uci):
        """Make a move in the game"""
        from django.contrib.auth import get_user_model
        from .serializers import MoveSerializer, GameSerializer
        
        User = get_user_model()
        
        try:
            game = Game.objects.get(id=self.game_id)
            user = User.objects.get(id=user_id)
            
            success, message, move_instance = ChessLogic.make_move(game, user, move_uci)
            
            if success:
                return {
                    'success': True,
                    'move': MoveSerializer(move_instance).data,
                    'game_state': GameSerializer(game).data
                }
            else:
                return {
                    'success': False,
                    'error': message
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
