import numpy as np
from typing import List, Tuple
import copy

class NeuralNetwork:
    """Simple feedforward neural network for neuroevolution."""
    
    def __init__(self, layer_sizes: List[int]):
        """
        Initialize network with given architecture.
        
        Args:
            layer_sizes: List of integers defining layer sizes
                        e.g., [4, 8, 8, 2] for 4 inputs, two hidden layers of 8, 2 outputs
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases with Xavier initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization for better initial performance
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        self.fitness = 0.0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input array of shape (n_samples, n_features) or (n_features,)
        
        Returns:
            Output of the network
        """
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        activation = x
        
        # Pass through all layers
        for i in range(self.num_layers - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            # Use tanh for hidden layers, no activation for output layer
            if i < self.num_layers - 2:
                activation = np.tanh(z)
            else:
                activation = z
        
        return activation
    
    def get_action(self, observation: np.ndarray) -> int:
        """
        Get action from observation (for discrete action spaces).
        
        Args:
            observation: Environment observation
            
        Returns:
            Action index (argmax of output)
        """
        output = self.forward(observation)
        return np.argmax(output)
    
    def clone(self) -> 'NeuralNetwork':
        """Create a deep copy of this network."""
        new_network = NeuralNetwork(self.layer_sizes.copy())
        new_network.weights = [w.copy() for w in self.weights]
        new_network.biases = [b.copy() for b in self.biases]
        new_network.fitness = self.fitness
        return new_network
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5, 
               structural: bool = False, structural_rate: float = 0.03):
        """
        Mutate network weights and biases, optionally with structural mutations.
        
        Args:
            mutation_rate: Probability of mutating each weight/bias
            mutation_strength: Standard deviation of Gaussian noise to add
            structural: Whether to allow structural mutations (add/remove neurons)
            structural_rate: Probability of structural mutation
        """
        # Weight mutations (always happen)
        for i in range(len(self.weights)):
            # Mutate weights
            mask = np.random.random(self.weights[i].shape) < mutation_rate
            self.weights[i] += mask * np.random.normal(0, mutation_strength, self.weights[i].shape)
            
            # Mutate biases
            mask = np.random.random(self.biases[i].shape) < mutation_rate
            self.biases[i] += mask * np.random.normal(0, mutation_strength, self.biases[i].shape)
        
        # Structural mutations (optional)
        if structural and len(self.layer_sizes) > 2:  # Only if we have hidden layers
            # Add neuron to a hidden layer
            if np.random.random() < structural_rate:
                self._add_neuron()
            
            # Remove neuron from a hidden layer (only if layer has >2 neurons)
            if np.random.random() < structural_rate:
                self._remove_neuron()
    
    def _add_neuron(self):
        """Add a neuron to a random hidden layer."""
        # Pick a random hidden layer (not input or output)
        if len(self.layer_sizes) <= 2:
            return
        
        hidden_layer_idx = np.random.randint(1, len(self.layer_sizes) - 1)
        old_size = self.layer_sizes[hidden_layer_idx]
        
        # Increase layer size
        self.layer_sizes[hidden_layer_idx] += 1
        new_size = self.layer_sizes[hidden_layer_idx]
        
        # Rebuild weights for connections TO this layer (from previous layer)
        if hidden_layer_idx > 0:
            prev_size = self.layer_sizes[hidden_layer_idx - 1]
            old_weights = self.weights[hidden_layer_idx - 1]
            old_biases = self.biases[hidden_layer_idx - 1]
            
            # Add new column to weights (random initialization)
            new_col = np.random.randn(prev_size, 1) * 0.1
            self.weights[hidden_layer_idx - 1] = np.hstack([old_weights, new_col])
            
            # Add new bias
            new_bias = np.zeros((1, 1))
            self.biases[hidden_layer_idx - 1] = np.hstack([old_biases, new_bias])
        
        # Rebuild weights for connections FROM this layer (to next layer)
        if hidden_layer_idx < len(self.layer_sizes) - 1:
            next_size = self.layer_sizes[hidden_layer_idx + 1]
            old_weights = self.weights[hidden_layer_idx]
            
            # Add new row to weights (random initialization)
            new_row = np.random.randn(1, next_size) * 0.1
            self.weights[hidden_layer_idx] = np.vstack([old_weights, new_row])
    
    def _remove_neuron(self):
        """Remove a neuron from a random hidden layer."""
        if len(self.layer_sizes) <= 2:
            return
        
        # Find hidden layers with more than 2 neurons
        removable_layers = []
        for i in range(1, len(self.layer_sizes) - 1):
            if self.layer_sizes[i] > 2:  # Keep at least 2 neurons
                removable_layers.append(i)
        
        if not removable_layers:
            return
        
        hidden_layer_idx = np.random.choice(removable_layers)
        neuron_idx = np.random.randint(0, self.layer_sizes[hidden_layer_idx])
        old_size = self.layer_sizes[hidden_layer_idx]
        
        # Decrease layer size
        self.layer_sizes[hidden_layer_idx] -= 1
        new_size = self.layer_sizes[hidden_layer_idx]
        
        # Remove column from weights TO this layer
        if hidden_layer_idx > 0:
            self.weights[hidden_layer_idx - 1] = np.delete(
                self.weights[hidden_layer_idx - 1], neuron_idx, axis=1
            )
            self.biases[hidden_layer_idx - 1] = np.delete(
                self.biases[hidden_layer_idx - 1], neuron_idx, axis=1
            )
        
        # Remove row from weights FROM this layer
        if hidden_layer_idx < len(self.layer_sizes) - 1:
            self.weights[hidden_layer_idx] = np.delete(
                self.weights[hidden_layer_idx], neuron_idx, axis=0
            )
    
    @staticmethod
    def crossover(parent1: 'NeuralNetwork', parent2: 'NeuralNetwork') -> 'NeuralNetwork':
        """
        Create offspring from two parent networks using uniform crossover.
        If architectures differ, return clone of fitter parent.
        
        Args:
            parent1: First parent network
            parent2: Second parent network
            
        Returns:
            Child network with mixed genes from both parents
        """
        # If architectures don't match, clone the fitter parent
        if parent1.layer_sizes != parent2.layer_sizes:
            return parent1.clone() if parent1.fitness >= parent2.fitness else parent2.clone()
        
        child = parent1.clone()
        
        # Uniform crossover: randomly pick genes from either parent
        for i in range(len(child.weights)):
            # Weight crossover
            mask = np.random.random(child.weights[i].shape) < 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
            
            # Bias crossover
            mask = np.random.random(child.biases[i].shape) < 0.5
            child.biases[i] = np.where(mask, parent1.biases[i], parent2.biases[i])
        
        child.fitness = 0.0
        return child
    
    def get_genome_size(self) -> int:
        """Get total number of parameters in the network."""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total
    
    def __repr__(self) -> str:
        return f"NeuralNetwork({self.layer_sizes}, fitness={self.fitness:.2f}, params={self.get_genome_size()})"