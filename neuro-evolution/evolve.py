import gymnasium as gym
import numpy as np
from typing import Tuple
import time

class NeuroEvolution:
    """Main class for running neuroevolution on Gym environments."""
    
    def __init__(
        self,
        env_name: str = 'CartPole-v1',
        population_size: int = 100,
        hidden_layers: list = [8, 8],
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        tournament_size: int = 5,
        evolve_architecture: bool = False,
        structural_mutation_rate: float = 0.03,
        save_gif: bool = False,
        gif_filename: str = 'evolution.gif'
    ):
        """
        Initialize neuroevolution system.
        
        Args:
            env_name: Name of the Gymnasium environment
            population_size: Number of networks in population
            hidden_layers: List of hidden layer sizes
            elite_ratio: Fraction of top performers to keep
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of mutation noise
            tournament_size: Number of candidates in tournament selection
            evolve_architecture: Whether to allow structural mutations (add/remove neurons)
            structural_mutation_rate: Probability of structural mutations
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.evolve_architecture = evolve_architecture
        
        # Get environment specs
        obs_size = self.env.observation_space.shape[0]
        
        # Handle different action space types
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_size = self.env.action_space.n
            self.action_type = 'discrete'
        else:
            action_size = self.env.action_space.shape[0]
            self.action_type = 'continuous'
        
        # Build network architecture
        self.layer_sizes = [obs_size] + hidden_layers + [action_size]
        
        print(f"Environment: {env_name}")
        print(f"Observation space: {obs_size}")
        print(f"Action space: {action_size} ({self.action_type})")
        print(f"Network architecture: {self.layer_sizes}")
        print(f"Population size: {population_size}")
        print(f"Architecture evolution: {'ENABLED' if evolve_architecture else 'DISABLED'}")
        if evolve_architecture:
            print(f"Structural mutation rate: {structural_mutation_rate}")
        print("-" * 50)
        
        # Create network factory
        def network_factory():
            from network import NeuralNetwork
            return NeuralNetwork(self.layer_sizes.copy())  # Copy to avoid shared reference
        
        # Initialize genetic algorithm
        from genetic import GeneticAlgorithm
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            network_factory=network_factory,
            elite_ratio=elite_ratio,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            tournament_size=tournament_size,
            evolve_architecture=evolve_architecture,
            structural_mutation_rate=structural_mutation_rate
        )
    
    def evaluate_network(self, network, n_episodes: int = 3, render: bool = False) -> float:
        """
        Evaluate a network's fitness by running it in the environment.
        
        Args:
            network: Neural network to evaluate
            n_episodes: Number of episodes to average over
            render: Whether to render the environment
            
        Returns:
            Average reward across episodes
        """
        total_reward = 0.0
        
        for episode in range(n_episodes):
            observation, info = self.env.reset()
            episode_reward = 0.0
            done = False
            truncated = False
            
            while not (done or truncated):
                if render:
                    self.env.render()
                
                # Get action from network
                action = network.get_action(observation)
                
                # Take action in environment
                observation, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def train(
        self,
        n_generations: int = 100,
        n_eval_episodes: int = 3,
        print_every: int = 5,
        save_best: bool = True
    ) -> dict:
        """
        Train the population through evolution.
        
        Args:
            n_generations: Number of generations to evolve
            n_eval_episodes: Number of episodes to evaluate each network
            print_every: Print statistics every N generations
            save_best: Whether to save the best network
            
        Returns:
            Dictionary with training statistics
        """
        print(f"\nStarting evolution for {n_generations} generations...")
        print("=" * 50)
        
        start_time = time.time()
        
        for gen in range(n_generations):
            gen_start = time.time()
            
            # Evaluate entire population
            self.ga.evaluate_population(
                lambda net: self.evaluate_network(net, n_episodes=n_eval_episodes)
            )
            
            # Get statistics
            stats = self.ga.get_statistics()
            
            # Print progress
            if gen % print_every == 0 or gen == n_generations - 1:
                gen_time = time.time() - gen_start
                elapsed = time.time() - start_time
                
                base_info = (f"Gen {gen:3d} | "
                            f"Best: {stats['best_fitness']:7.2f} | "
                            f"Avg: {stats['avg_fitness']:7.2f} | "
                            f"Worst: {stats['worst_fitness']:7.2f} | "
                            f"Diversity: {stats['diversity']:6.2f} | "
                            f"Time: {gen_time:.2f}s | "
                            f"Elapsed: {elapsed:.0f}s")
                
                # Add architecture info if evolving
                if self.evolve_architecture and 'unique_architectures' in stats:
                    arch_info = (f" | Archs: {stats['unique_architectures']} | "
                               f"Best: {stats['best_architecture']}")
                    print(base_info + arch_info)
                else:
                    print(base_info)
            
            # Evolve to next generation
            if gen < n_generations - 1:
                self.ga.evolve_generation()
        
        total_time = time.time() - start_time
        
        print("=" * 50)
        print(f"Evolution complete! Total time: {total_time:.2f}s")
        print(f"Best fitness achieved: {self.ga.get_best_network().fitness:.2f}")
        
        # Save best network
        if save_best:
            self.ga.save_checkpoint('best_network.npz', env_name=self.env_name)
            print("Best network saved to 'best_network.npz'")
        
        return {
            'best_fitness_history': self.ga.best_fitness_history,
            'avg_fitness_history': self.ga.avg_fitness_history,
            'diversity_history': self.ga.diversity_history,
            'best_network': self.ga.get_best_network(),
            'total_time': total_time
        }
    
    def demonstrate(self, network=None, n_episodes: int = 5):
        """
        Demonstrate the best network's performance with rendering.
        
        Args:
            network: Network to demonstrate (uses best if None)
            n_episodes: Number of episodes to show
        """
        if network is None:
            network = self.ga.get_best_network()
        
        print(f"\nDemonstrating best network (fitness: {network.fitness:.2f})")
        print("-" * 50)
        
        # Create a new environment with rendering
        render_env = gym.make(self.env_name, render_mode='human')
        
        for episode in range(n_episodes):
            observation, info = render_env.reset()
            episode_reward = 0.0
            done = False
            truncated = False
            steps = 0
            
            while not (done or truncated):
                render_env.render()
                
                action = network.get_action(observation)
                observation, reward, done, truncated, info = render_env.step(action)
                
                episode_reward += reward
                steps += 1
                
                time.sleep(0.01)  # Slow down for viewing
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        render_env.close()
        print("Demonstration complete!")
    
    def demonstrate_and_save_gif(self, network=None, n_episodes: int = 1, gif_filename: str = 'landing.gif'):
        """
        Demonstrate the network and save as GIF.
        
        Args:
            network: Network to demonstrate (uses best if None)
            n_episodes: Number of episodes to record
            gif_filename: Output filename for GIF
        """
        if network is None:
            network = self.ga.get_best_network()
        
        print(f"\nRecording demonstration to '{gif_filename}'...")
        print("-" * 50)
        
        # Create environment with rgb_array mode for recording
        render_env = gym.make(self.env_name, render_mode='rgb_array')
        
        frames = []
        
        for episode in range(n_episodes):
            observation, info = render_env.reset()
            episode_reward = 0.0
            done = False
            truncated = False
            steps = 0
            
            while not (done or truncated):
                # Capture frame
                frame = render_env.render()
                frames.append(frame)
                
                action = network.get_action(observation)
                observation, reward, done, truncated, info = render_env.step(action)
                
                episode_reward += reward
                steps += 1
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        render_env.close()
        
        # Save as GIF
        print(f"Saving GIF with {len(frames)} frames...")
        save_frames_as_gif(frames, gif_filename)
        print(f"GIF saved to '{gif_filename}'!")
    
    def close(self):
        """Clean up environment."""
        self.env.close()


def save_frames_as_gif(frames, filename, fps=30):
    """Save frames as animated GIF."""
    from PIL import Image
    
    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save as GIF
    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000//fps,  # Duration in milliseconds
        loop=0
    )