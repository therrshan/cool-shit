import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import networkx as nx
from typing import List, Optional
import time

import matplotlib
matplotlib.use('Qt5Agg')

class EvolutionVisualizer:
    """Real-time visualization dashboard for neuroevolution."""
    
    def __init__(self, figsize=(16, 10)):
        """
        Initialize the visualization dashboard.
        
        Args:
            figsize: Figure size as (width, height)
        """
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('Neuroevolution Dashboard', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax_fitness = self.fig.add_subplot(gs[0, :2])  # Fitness over time
        self.ax_diversity = self.fig.add_subplot(gs[1, :2])  # Diversity over time
        self.ax_network = self.fig.add_subplot(gs[0:2, 2])  # Network architecture
        self.ax_population = self.fig.add_subplot(gs[2, :2])  # Population distribution
        self.ax_stats = self.fig.add_subplot(gs[2, 2])  # Statistics text
        
        # Initialize data storage
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        self.worst_fitness = []
        self.diversity = []
        self.current_population_fitness = []
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up the initial plot styles and labels."""
        
        # Fitness plot
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.set_title('Fitness Evolution Over Generations')
        self.ax_fitness.grid(True, alpha=0.3)
        self.ax_fitness.legend(['Best', 'Average', 'Worst'], loc='lower right')
        
        # Diversity plot
        self.ax_diversity.set_xlabel('Generation')
        self.ax_diversity.set_ylabel('Genetic Diversity')
        self.ax_diversity.set_title('Population Diversity Over Time')
        self.ax_diversity.grid(True, alpha=0.3)
        
        # Population distribution
        self.ax_population.set_xlabel('Fitness')
        self.ax_population.set_ylabel('Number of Networks')
        self.ax_population.set_title('Current Population Fitness Distribution')
        self.ax_population.grid(True, alpha=0.3)
        
        # Network architecture
        self.ax_network.set_title('Best Network Architecture')
        self.ax_network.axis('off')
        
        # Statistics panel
        self.ax_stats.axis('off')
        
        plt.ion()  # Interactive mode
        plt.show()
    
    def update(self, ga_stats: dict, population_fitness: List[float]):
        """
        Update all plots with new data.
        
        Args:
            ga_stats: Statistics dictionary from GeneticAlgorithm
            population_fitness: List of fitness values for current population
        """
        # Store data
        self.generations.append(ga_stats['generation'])
        self.best_fitness.append(ga_stats['best_fitness'])
        self.avg_fitness.append(ga_stats['avg_fitness'])
        self.worst_fitness.append(ga_stats['worst_fitness'])
        self.diversity.append(ga_stats['diversity'])
        self.current_population_fitness = population_fitness
        
        # Update plots
        self._update_fitness_plot()
        self._update_diversity_plot()
        self._update_population_distribution()
        self._update_stats_panel(ga_stats)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _update_fitness_plot(self):
        """Update the fitness over time plot."""
        self.ax_fitness.clear()
        
        self.ax_fitness.plot(self.generations, self.best_fitness, 
                            'g-', linewidth=2, label='Best')
        self.ax_fitness.plot(self.generations, self.avg_fitness, 
                            'b-', linewidth=2, label='Average')
        self.ax_fitness.plot(self.generations, self.worst_fitness, 
                            'r-', linewidth=1, alpha=0.5, label='Worst')
        
        self.ax_fitness.fill_between(self.generations, 
                                     self.worst_fitness, 
                                     self.best_fitness, 
                                     alpha=0.2, color='blue')
        
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.set_title('Fitness Evolution Over Generations')
        self.ax_fitness.grid(True, alpha=0.3)
        self.ax_fitness.legend(loc='lower right')
    
    def _update_diversity_plot(self):
        """Update the diversity over time plot."""
        self.ax_diversity.clear()
        
        self.ax_diversity.plot(self.generations, self.diversity, 
                              'purple', linewidth=2)
        self.ax_diversity.fill_between(self.generations, 0, self.diversity, 
                                       alpha=0.3, color='purple')
        
        self.ax_diversity.set_xlabel('Generation')
        self.ax_diversity.set_ylabel('Genetic Diversity')
        self.ax_diversity.set_title('Population Diversity Over Time')
        self.ax_diversity.grid(True, alpha=0.3)
    
    def _update_population_distribution(self):
        """Update the population fitness distribution histogram."""
        self.ax_population.clear()
        
        if len(self.current_population_fitness) > 0:
            self.ax_population.hist(self.current_population_fitness, 
                                   bins=20, 
                                   color='skyblue', 
                                   edgecolor='black', 
                                   alpha=0.7)
            
            # Add vertical lines for mean and best
            mean_fitness = np.mean(self.current_population_fitness)
            best_fitness = max(self.current_population_fitness)
            
            self.ax_population.axvline(mean_fitness, 
                                      color='blue', 
                                      linestyle='--', 
                                      linewidth=2, 
                                      label=f'Mean: {mean_fitness:.2f}')
            self.ax_population.axvline(best_fitness, 
                                      color='green', 
                                      linestyle='--', 
                                      linewidth=2, 
                                      label=f'Best: {best_fitness:.2f}')
        
        self.ax_population.set_xlabel('Fitness')
        self.ax_population.set_ylabel('Number of Networks')
        self.ax_population.set_title('Current Population Fitness Distribution')
        self.ax_population.grid(True, alpha=0.3)
        self.ax_population.legend()
    
    def _update_stats_panel(self, ga_stats: dict):
        """Update the statistics text panel."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats_text = f"""
        CURRENT STATISTICS
        ══════════════════════
        
        Generation: {ga_stats['generation']}
        
        Best Fitness:    {ga_stats['best_fitness']:.2f}
        Average Fitness: {ga_stats['avg_fitness']:.2f}
        Worst Fitness:   {ga_stats['worst_fitness']:.2f}
        Std Dev:         {ga_stats['std_fitness']:.2f}
        
        Diversity:       {ga_stats['diversity']:.2f}
        
        Improvement: {self._calculate_improvement():.2f}%
        """
        
        self.ax_stats.text(0.1, 0.5, stats_text, 
                          fontsize=10, 
                          verticalalignment='center',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', 
                                   facecolor='wheat', 
                                   alpha=0.5))
    
    def _calculate_improvement(self) -> float:
        """Calculate percentage improvement from first to current generation."""
        if len(self.best_fitness) < 2:
            return 0.0
        
        initial = self.best_fitness[0]
        current = self.best_fitness[-1]
        
        if initial == 0:
            return 0.0
        
        return ((current - initial) / abs(initial)) * 100
    
    def visualize_network(self, network):
        """
        Visualize the network architecture using networkx.
        
        Args:
            network: NeuralNetwork instance to visualize
        """
        self.ax_network.clear()
        self.ax_network.axis('off')
        self.ax_network.set_title('Best Network Architecture')
        
        G = nx.DiGraph()
        pos = {}
        
        layer_sizes = network.layer_sizes
        max_neurons = max(layer_sizes)
        
        # Create nodes and positions
        node_id = 0
        layer_nodes = []
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_node_ids = []
            
            # Calculate vertical spacing for this layer
            vertical_spacing = 1.0 / (layer_size + 1)
            vertical_offset = (max_neurons - layer_size) * vertical_spacing / 2
            
            for neuron_idx in range(layer_size):
                G.add_node(node_id)
                x = layer_idx / (len(layer_sizes) - 1)
                y = vertical_offset + (neuron_idx + 1) * vertical_spacing
                pos[node_id] = (x, y)
                layer_node_ids.append(node_id)
                node_id += 1
            
            layer_nodes.append(layer_node_ids)
        
        # Add edges between layers
        for i in range(len(layer_nodes) - 1):
            for src in layer_nodes[i]:
                for dst in layer_nodes[i + 1]:
                    G.add_edge(src, dst)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue', 
                              node_size=500, 
                              ax=self.ax_network)
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray', 
                              alpha=0.3, 
                              arrows=False, 
                              ax=self.ax_network)
        
        # Add layer labels
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes) - 2)] + ['Output']
        for i, (name, size) in enumerate(zip(layer_names, layer_sizes)):
            x = i / (len(layer_sizes) - 1)
            self.ax_network.text(x, -0.1, f'{name}\n({size})', 
                               ha='center', 
                               va='top', 
                               fontsize=9,
                               fontweight='bold')
        
        # Add parameter count
        param_count = network.get_genome_size()
        self.ax_network.text(0.5, 1.05, f'Total Parameters: {param_count:,}', 
                            ha='center', 
                            va='bottom', 
                            fontsize=10,
                            fontweight='bold')
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)
    
    def save_figure(self, filename: str = 'evolution_dashboard.png'):
        """Save the current dashboard as an image."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to {filename}")


class LiveEvolutionVisualizer:
    """Wrapper to integrate visualization with evolution training."""
    
    def __init__(self, neuroevolution):
        """
        Initialize live visualizer.
        
        Args:
            neuroevolution: NeuroEvolution instance
        """
        self.nevo = neuroevolution
        self.visualizer = EvolutionVisualizer()
        
    def train_with_visualization(
        self,
        n_generations: int = 100,
        n_eval_episodes: int = 3,
        update_every: int = 1,
        save_best: bool = True
    ):
        """
        Train with live visualization updates.
        
        Args:
            n_generations: Number of generations to evolve
            n_eval_episodes: Number of episodes to evaluate each network
            update_every: Update visualization every N generations
            save_best: Whether to save the best network
        """
        print(f"Starting evolution with live visualization...")
        print("=" * 50)
        
        start_time = time.time()
        
        for gen in range(n_generations):
            # Evaluate population
            self.nevo.ga.evaluate_population(
                lambda net: self.nevo.evaluate_network(net, n_episodes=n_eval_episodes)
            )
            
            # Get statistics
            stats = self.nevo.ga.get_statistics()
            population_fitness = [net.fitness for net in self.nevo.ga.population]
            
            # Update visualization
            if gen % update_every == 0:
                self.visualizer.update(stats, population_fitness)
                
                # Update network visualization periodically
                if gen % 5 == 0:
                    best_net = self.nevo.ga.get_best_network()
                    self.visualizer.visualize_network(best_net)
            
            # Evolve to next generation
            if gen < n_generations - 1:
                self.nevo.ga.evolve_generation()
        
        total_time = time.time() - start_time
        
        print("=" * 50)
        print(f"Evolution complete! Total time: {total_time:.2f}s")
        
        # Final visualization update
        stats = self.nevo.ga.get_statistics()
        population_fitness = [net.fitness for net in self.nevo.ga.population]
        self.visualizer.update(stats, population_fitness)
        best_net = self.nevo.ga.get_best_network()
        self.visualizer.visualize_network(best_net)
        
        # Save dashboard
        self.visualizer.save_figure('final_dashboard.png')
        
        if save_best:
            self.nevo.ga.save_checkpoint('best_network.npz')
            print("Best network saved to 'best_network.npz'")
        
        return {
            'best_fitness_history': self.nevo.ga.best_fitness_history,
            'avg_fitness_history': self.nevo.ga.avg_fitness_history,
            'diversity_history': self.nevo.ga.diversity_history,
            'best_network': best_net,
            'total_time': total_time
        }
    
    def close(self):
        """Clean up resources."""
        self.visualizer.close()
        self.nevo.close()


def main():
    """Example usage with live visualization."""
    from evolve import NeuroEvolution
    
    # Create neuroevolution system
    nevo = NeuroEvolution(
        env_name='CartPole-v1',
        population_size=100,
        hidden_layers=[16, 16],
        elite_ratio=0.2,
        mutation_rate=0.1,
        mutation_strength=0.3,
        tournament_size=5
    )
    
    # Create visualizer
    live_viz = LiveEvolutionVisualizer(nevo)
    
    # Train with visualization
    results = live_viz.train_with_visualization(
        n_generations=50,
        n_eval_episodes=5,
        update_every=1,
        save_best=True
    )
    
    print("\nPress Enter to see demonstration of best network...")
    input()
    
    # Demonstrate best network
    nevo.demonstrate(n_episodes=3)
    
    # Keep visualization open
    print("\nClose the plot window to exit...")
    plt.show(block=True)
    
    live_viz.close()


if __name__ == "__main__":
    main()