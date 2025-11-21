#!/usr/bin/env python3
"""
Neuroevolution - Train neural networks using genetic algorithms
A complete implementation with live visualization.

Usage:
    python main.py                          # Default CartPole training
    python main.py --env LunarLander-v2    # Different environment
    python main.py --no-viz                # Train without visualization
    python main.py --demo-only             # Just demonstrate saved network
    python main.py --evolve-architecture   # Enable architecture evolution
    python main.py --generations 100       # Custom number of generations
"""

import argparse
import sys
from network import NeuralNetwork
from genetic import GeneticAlgorithm
from evolve import NeuroEvolution
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Neuroevolution Training System')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Gym environment name (default: CartPole-v1)')
    
    # Training settings
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations to train (default: 50)')
    parser.add_argument('--population', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Episodes per evaluation (default: 5)')
    
    # Network architecture
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[16, 16],
                       help='Hidden layer sizes (default: 16 16)')
    
    # GA hyperparameters
    parser.add_argument('--elite-ratio', type=float, default=0.2,
                       help='Elite ratio (default: 0.2)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                       help='Mutation rate (default: 0.1)')
    parser.add_argument('--mutation-strength', type=float, default=0.3,
                       help='Mutation strength (default: 0.3)')
    parser.add_argument('--tournament-size', type=int, default=5,
                       help='Tournament size (default: 5)')
    
    # Architecture evolution
    parser.add_argument('--evolve-architecture', action='store_true',
                       help='Enable architecture evolution (add/remove neurons)')
    parser.add_argument('--structural-mutation-rate', type=float, default=0.03,
                       help='Structural mutation rate (default: 0.03)')
    
    # Visualization and demo
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable live visualization')
    parser.add_argument('--demo-only', action='store_true',
                       help='Only demonstrate saved network')
    parser.add_argument('--demo-episodes', type=int, default=3,
                       help='Number of episodes to demonstrate (default: 3)')
    
    # Saving
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save best network')

    parser.add_argument('--save-gif', action='store_true',
                       help='Save demonstration as GIF')

    parser.add_argument('--gif-filename', type=str, default='landing.gif',
                       help='Output filename for GIF (default: landing.gif)')
    
    return parser.parse_args()


def demonstrate_network(args):
    """Load and demonstrate a saved network."""
    print("Loading saved network...")
    
    try:
        # Load checkpoint to get environment name
        data = np.load('best_network.npz', allow_pickle=True)
        
        # Check if environment was saved
        if 'env_name' in data:
            saved_env = str(data['env_name'])
            print(f"Checkpoint trained on: {saved_env}")
            
            # Use saved environment unless user explicitly specified one
            if args.env == 'CartPole-v1':  # Default value
                env_to_use = saved_env
                print(f"Using saved environment: {env_to_use}")
            else:
                env_to_use = args.env
                print(f"Warning: Checkpoint trained on {saved_env} but demonstrating on {env_to_use}")
        else:
            env_to_use = args.env
            print(f"Warning: Checkpoint doesn't specify environment, using: {env_to_use}")
        
        nevo = NeuroEvolution(
            env_name=env_to_use,
            population_size=1,  # Only need 1 for demo
            hidden_layers=args.hidden_layers
        )
        
        best_network = nevo.ga.load_checkpoint('best_network.npz')
        print(f"Loaded network with fitness: {best_network.fitness:.2f}")

        if args.save_gif:
            print(f"Recording demonstration to '{args.gif_filename}'...")
            nevo.demonstrate_and_save_gif(
                network=best_network,
                n_episodes=args.demo_episodes,
                gif_filename=args.gif_filename
            )
        else:
            nevo.demonstrate(network=best_network, n_episodes=args.demo_episodes)
        
    except FileNotFoundError:
        print("Error: No saved network found at 'best_network.npz'")
        print("Train a network first before using --demo-only")
        sys.exit(1)
    finally:
        if 'nevo' in locals():
            nevo.close()


def train_without_visualization(args):
    """Train without visualization (faster), create summary at the end."""
    print("=" * 60)
    print("NEUROEVOLUTION TRAINING (NO VISUALIZATION)")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Architecture evolution: {'ENABLED' if args.evolve_architecture else 'DISABLED'}")
    print("=" * 60)
    
    # Create neuroevolution system
    nevo = NeuroEvolution(
        env_name=args.env,
        population_size=args.population,
        hidden_layers=args.hidden_layers,
        elite_ratio=args.elite_ratio,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        tournament_size=args.tournament_size,
        evolve_architecture=args.evolve_architecture,
        structural_mutation_rate=args.structural_mutation_rate
    )
    
    # Train
    results = nevo.train(
        n_generations=args.generations,
        n_eval_episodes=args.eval_episodes,
        print_every=5,
        save_best=not args.no_save
    )
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    
    # Create final summary visualization
    print("\nGenerating final summary visualization...")
    try:
        from visualize import EvolutionVisualizer
        import matplotlib.pyplot as plt
        
        # Create visualizer
        viz = EvolutionVisualizer()
        
        # Populate with training data
        for gen in range(len(results['best_fitness_history'])):
            stats = {
                'generation': gen,
                'best_fitness': results['best_fitness_history'][gen],
                'avg_fitness': results['avg_fitness_history'][gen],
                'worst_fitness': min(results['avg_fitness_history'][gen] - 50, 0),  # Approximate
                'std_fitness': 0,  # Not tracked during no-viz training
                'diversity': results['diversity_history'][gen]
            }
            
            # For final generation, use actual population fitness
            if gen == len(results['best_fitness_history']) - 1:
                population_fitness = [net.fitness for net in nevo.ga.population]
            else:
                population_fitness = []  # Don't have historical data
            
            viz.update(stats, population_fitness)
        
        # Visualize best network architecture
        best_net = results['best_network']
        viz.visualize_network(best_net)
        
        # Save the figure
        filename = f'summary_{args.env}_{args.generations}gen.png'
        viz.save_figure(filename)
        print(f"Summary visualization saved to '{filename}'")
        
        # Show the plot
        print("\nClose the plot window to continue...")
        plt.show()
        
        viz.close()
        
    except ImportError as e:
        print(f"Could not create visualization: {e}")
        print("Continuing without final plot...")
    
    if args.save_gif:
        print("Recording best network as GIF...")
        nevo.demonstrate_and_save_gif(
            n_episodes=1, 
            gif_filename=args.gif_filename
        )
    
    # Ask to demonstrate
    print("\n" + "=" * 60)
    response = input("Watch the best network perform? (y/n): ").strip().lower()
    
    if response == 'y':
        nevo.demonstrate(n_episodes=args.demo_episodes)
    
    nevo.close()

def train_with_visualization(args):
    """Train with live visualization."""
    from visualize import LiveEvolutionVisualizer
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("NEUROEVOLUTION TRAINING WITH LIVE VISUALIZATION")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Hidden layers: {args.hidden_layers}")
    print(f"Architecture evolution: {'ENABLED' if args.evolve_architecture else 'DISABLED'}")
    print("=" * 60)
    
    # Create neuroevolution system
    nevo = NeuroEvolution(
        env_name=args.env,
        population_size=args.population,
        hidden_layers=args.hidden_layers,
        elite_ratio=args.elite_ratio,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        tournament_size=args.tournament_size,
        evolve_architecture=args.evolve_architecture,
        structural_mutation_rate=args.structural_mutation_rate
    )
    
    # Create visualizer
    live_viz = LiveEvolutionVisualizer(nevo)
    
    # Train with visualization
    results = live_viz.train_with_visualization(
        n_generations=args.generations,
        n_eval_episodes=args.eval_episodes,
        update_every=1,
        save_best=not args.no_save
    )
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 60}")

    if args.save_gif:
        print("Recording best network as GIF...")
        nevo.demonstrate_and_save_gif(
            n_episodes=1, 
            gif_filename=args.gif_filename
        )
    
    # Ask to demonstrate
    response = input("Watch the best network perform? (y/n): ").strip().lower()
    
    if response == 'y':
        nevo.demonstrate(n_episodes=args.demo_episodes)
    
    # Keep visualization open
    print("\nClose the plot window to exit...")
    plt.show(block=True)
    
    live_viz.close()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.demo_only:
            demonstrate_network(args)
        elif args.no_viz:
            train_without_visualization(args)
        else:
            # Try visualization, fall back to no-viz if matplotlib issues
            try:
                from visualize import LiveEvolutionVisualizer
                import matplotlib.pyplot as plt
                train_with_visualization(args)
            except ImportError as e:
                print(f"Visualization unavailable: {e}")
                print("Falling back to training without visualization...")
                train_without_visualization(args)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()