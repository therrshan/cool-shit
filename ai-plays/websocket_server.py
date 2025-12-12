import asyncio
import websockets
import json
import numpy as np

class GodotEnvironment:
    def __init__(self):
        self.websocket = None
        self.current_observation = None
        self.current_reward = 0
        self.done = False
        
    async def connect(self, websocket):
        """Store the websocket connection"""
        self.websocket = websocket
        print("Godot client connected!")
        
    async def send_action(self, action):
        """Send action to Godot"""
        action_data = {
            "action": int(action)
        }
        await self.websocket.send(json.dumps(action_data))
        
    async def receive_observation(self):
        """Receive observation from Godot"""
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(message)
            
            self.current_observation = np.array([
                data['player_x'] / 1300.0,  # Normalize
                data['player_y'] / 720.0,
                data['player_vx'] / 300.0,
                data['player_vy'] / 850.0,
                data['on_floor'],
                data['melon_x'] / 1300.0,
                data['melon_y'] / 720.0,
                data['melon_dist'] / 1500.0,  # Max diagonal distance
                data['snail_x'] / 1300.0,
                data['snail_y'] / 720.0,
                data['snail_dist'] / 1500.0,
                data['snail_dir'],
                data['score'] / 100.0  # Normalize score
            ], dtype=np.float32)
            
            self.current_reward = data['reward']
            self.done = data['done']
            
            return self.current_observation, self.current_reward, self.done
            
        except asyncio.TimeoutError:
            print("Timeout waiting for observation")
            return None, 0, True
            
    async def reset(self):
        """Request environment reset"""
        reset_msg = {"reset": True}
        await self.websocket.send(json.dumps(reset_msg))
        # Wait for new observation after reset
        obs, _, _ = await self.receive_observation()
        return obs

async def handle_client(websocket, path, trainer):
    """Handle incoming Godot connections"""
    env = GodotEnvironment()
    await env.connect(websocket)
    
    try:
        # Training loop
        episode = 0
        while True:
            episode += 1
            print(f"\n=== Episode {episode} ===")
            
            # Reset environment
            obs = await env.reset()
            if obs is None:
                break
                
            episode_reward = 0
            step = 0
            
            while not env.done:
                # Get action from policy
                action = trainer.select_action(obs)
                
                # Send action to Godot
                await env.send_action(action)
                
                # Receive next observation
                next_obs, reward, done = await env.receive_observation()
                if next_obs is None:
                    break
                
                # Store transition
                trainer.store_transition(obs, action, reward, next_obs, done)
                
                obs = next_obs
                episode_reward += reward
                step += 1
                
                # Update policy periodically
                if step % 64 == 0:
                    trainer.update()
                    
            print(f"Episode {episode} finished - Steps: {step}, Reward: {episode_reward:.2f}")
            
            # Update at end of episode
            trainer.update()
            
            # Save model periodically
            if episode % 10 == 0:
                trainer.save_model(f"model_episode_{episode}.pth")
                
    except websockets.exceptions.ConnectionClosed:
        print("Godot disconnected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def start_server(trainer, host="localhost", port=8765):
    """Start WebSocket server"""
    print(f"Starting WebSocket server on {host}:{port}")
    
    async def handler(websocket):
        await handle_client(websocket, None, trainer)
    
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Import trainer (we'll create this next)
    from ppo_trainer import PPOTrainer
    
    # Initialize trainer
    trainer = PPOTrainer(
        state_dim=13,  # Number of observation features
        action_dim=6   # Number of possible actions
    )
    
    # Start server
    asyncio.run(start_server(trainer))