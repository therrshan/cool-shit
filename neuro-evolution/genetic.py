import numpy as np
from typing import List, Callable, Tuple
import copy

class GeneticAlgorithm:
    """Genetic algorithm for evolving neural networks."""
    
    def __init__(
        self,
        population_size: int,
        network_factory: Callable,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        tournament_size: int = 5,
        evolve_architecture: bool = False,
        structural_mutation_rate: float = 0.03
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Number of networks in the population
            network_factory: Function that creates a new network (e.g., lambda: NeuralNetwork([4, 8, 2]))
            elite_ratio: Fraction of top performers to keep unchanged (elitism)
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of mutation noise
            tournament_size: Number of candidates in tournament selection
            evolve_architecture: Whether to allow structural mutations
            structural_mutation_rate: Probability of adding/removing neurons
        """
        self.population_size = population_size
        self.network_factory = network_factory
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.evolve_architecture = evolve_architecture
        self.structural_mutation_rate = structural_mutation_rate
        
        # Initialize population
        self.population = [network_factory() for _ in range(population_size)]
        self.generation = 0
        
        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
    
    def evaluate_population(self, fitness_function: Callable) -> None:
        """
        Evaluate fitness of all networks in the population.
        
        Args:
            fitness_function: Function that takes a network and returns its fitness score
        """
        for network in self.population:
            network.fitness = fitness_function(network)
    
    def get_statistics(self) -> dict:
        """Get current population statistics."""
        fitnesses = [net.fitness for net in self.population]
        
        # Track architecture diversity
        architectures = [tuple(net.layer_sizes) for net in self.population]
        unique_architectures = len(set(architectures))
        
        # Get architecture stats
        total_params = [net.get_genome_size() for net in self.population]
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'diversity': self._calculate_diversity(),
            'unique_architectures': unique_architectures,
            'avg_params': np.mean(total_params),
            'best_architecture': self.get_best_network().layer_sizes
        }
    
    def _calculate_diversity(self) -> float:
        """
        Calculate genetic diversity in population.
        Simple metric: average pairwise weight difference.
        """
        if len(self.population) < 2:
            return 0.0
        
        # Sample a few pairs to estimate diversity (expensive for large populations)
        sample_size = min(10, len(self.population))
        sample = np.random.choice(self.population, sample_size, replace=False)
        
        total_diff = 0.0
        count = 0
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                diff = self._network_distance(sample[i], sample[j])
                total_diff += diff
                count += 1
        
        return total_diff / count if count > 0 else 0.0
    
    def _network_distance(self, net1, net2) -> float:
        """Calculate L2 distance between two networks' weights."""
        # Skip if architectures don't match
        if net1.layer_sizes != net2.layer_sizes:
            return 100.0  # Return large distance for different architectures
        
        total_diff = 0.0
        for w1, w2 in zip(net1.weights, net2.weights):
            total_diff += np.sum((w1 - w2) ** 2)
        return np.sqrt(total_diff)
    
    def select_parent(self) -> 'NeuralNetwork':
        """
        Select a parent using tournament selection.
        
        Returns:
            Selected network
        """
        # Randomly select tournament_size candidates
        candidates = np.random.choice(self.population, self.tournament_size, replace=False)
        
        # Return the best one
        return max(candidates, key=lambda net: net.fitness)
    
    def evolve_generation(self) -> None:
        """Evolve the population by one generation."""
        # Sort population by fitness (descending)
        self.population.sort(key=lambda net: net.fitness, reverse=True)
        
        # Track statistics
        stats = self.get_statistics()
        self.best_fitness_history.append(stats['best_fitness'])
        self.avg_fitness_history.append(stats['avg_fitness'])
        self.diversity_history.append(stats['diversity'])
        
        # Calculate how many elites to keep
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        
        # Keep the elite unchanged
        new_population = [net.clone() for net in self.population[:n_elite]]
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            # Create offspring through crossover
            child = parent1.crossover(parent1, parent2)
            
            # Mutate the offspring
            child.mutate(
                self.mutation_rate, 
                self.mutation_strength,
                structural=self.evolve_architecture,
                structural_rate=self.structural_mutation_rate
            )
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def get_best_network(self) -> 'NeuralNetwork':
        """Get the best performing network in current population."""
        return max(self.population, key=lambda net: net.fitness)
    
    def get_top_networks(self, n: int = 5) -> List['NeuralNetwork']:
        """Get the top n networks by fitness."""
        sorted_pop = sorted(self.population, key=lambda net: net.fitness, reverse=True)
        return sorted_pop[:n]
    
    def save_checkpoint(self, filepath: str, env_name: str = None) -> None:
        """Save the current best network to a file."""
        best_net = self.get_best_network()
        save_dict = {
            'layer_sizes': np.array(best_net.layer_sizes),
            'fitness': best_net.fitness,
            'generation': self.generation
        }
        
        # Save environment name if provided
        if env_name:
            save_dict['env_name'] = env_name
        
        # Save weights and biases separately
        for i, (w, b) in enumerate(zip(best_net.weights, best_net.biases)):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}'] = b
        
        np.savez(filepath, **save_dict)
    
    def load_checkpoint(self, filepath: str) -> 'NeuralNetwork':
        """Load a network from a saved checkpoint."""
        data = np.load(filepath, allow_pickle=True)
        
        layer_sizes = data['layer_sizes'].tolist()
        network = self.network_factory()
        
        # Load weights and biases
        network.weights = []
        network.biases = []
        
        i = 0
        while f'weight_{i}' in data:
            network.weights.append(data[f'weight_{i}'])
            network.biases.append(data[f'bias_{i}'])
            i += 1
        
        network.fitness = float(data['fitness'])
        
        return network
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"GeneticAlgorithm(gen={self.generation}, "
                f"pop_size={self.population_size}, "
                f"best_fitness={stats['best_fitness']:.2f}, "
                f"avg_fitness={stats['avg_fitness']:.2f})")