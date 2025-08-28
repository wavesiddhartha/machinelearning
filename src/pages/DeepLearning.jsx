function DeepLearning() {
  return (
    <div className="page">
      <div className="content">
        <h1>🧠 Deep Learning & Neural Networks</h1>
        
        <section className="dl-intro">
          <h2>Introduction to Deep Learning</h2>
          <p>Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It's inspired by how the human brain processes information.</p>
          
          <div className="dl-concepts">
            <div className="concept-card">
              <h3>🔗 Neural Networks</h3>
              <p>Networks of interconnected nodes (neurons) that process information</p>
            </div>
            <div className="concept-card">
              <h3>⚡ Activation Functions</h3>
              <p>Functions that determine neuron output based on input</p>
            </div>
            <div className="concept-card">
              <h3>🎯 Backpropagation</h3>
              <p>Algorithm for training networks by propagating errors backward</p>
            </div>
            <div className="concept-card">
              <h3>🔄 Gradient Descent</h3>
              <p>Optimization algorithm for updating network weights</p>
            </div>
          </div>
        </section>

        <section className="neural-network">
          <h2>📊 Neural Network from Scratch</h2>
          
          <div className="code-example">
            <h4>Complete Neural Network Implementation</h4>
            <pre><code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    """Complete Neural Network implementation with mathematical details"""
    
    def __init__(self, layers, learning_rate=0.01, epochs=1000):
        self.layers = layers  # List of layer sizes [input, hidden1, hidden2, ..., output]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.costs = []
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize weights and biases using Xavier/Glorot initialization"""
        for i in range(len(self.layers) - 1):
            # Xavier initialization: weights ~ N(0, sqrt(2/(n_in + n_out)))
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / (self.layers[i] + self.layers[i + 1]))
            bias = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, z):
        """Sigmoid activation: σ(z) = 1/(1 + e^(-z))"""
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
        return z * (1 - z)
    
    def relu(self, z):
        """ReLU activation: ReLU(z) = max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def tanh(self, z):
        """Tanh activation: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))"""
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        """Derivative of tanh: tanh'(z) = 1 - tanh²(z)"""
        return 1 - np.power(z, 2)
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            # Linear transformation: z = XW + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer
                activation = self.sigmoid(z)  # Use sigmoid for binary classification
            else:  # Hidden layers
                activation = self.relu(z)  # Use ReLU for hidden layers
            
            activations.append(activation)
        
        return activations, z_values
    
    def compute_cost(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        m = y_true.shape[0]
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def backward_propagation(self, X, y, activations, z_values):
        """Backward pass - compute gradients using chain rule"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Start with output layer error
        # δ^L = (a^L - y) ⊙ σ'(z^L)  [for sigmoid output]
        delta = activations[-1] - y  # Derivative of cross-entropy + sigmoid
        
        # Compute gradients for each layer (backward)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients for weights and biases
            dW = (1/m) * np.dot(activations[i].T, delta)
            db = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                # δ^(l-1) = (δ^l * W^l) ⊙ f'(z^(l-1))
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def fit(self, X, y):
        """Train the neural network"""
        for epoch in range(self.epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(y, activations[-1])
            self.costs.append(cost)
            
            # Backward propagation
            gradients_w, gradients_b = self.backward_propagation(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(gradients_w, gradients_b)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Cost: {cost:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        predictions = activations[-1]
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

# Generate sample data
print("Creating sample dataset for neural network...")
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                          n_redundant=2, n_clusters_per_class=1, random_state=42)

# Reshape y for neural network
y = y.reshape(-1, 1)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Create and train neural network
print("\\nCreating neural network...")
# Architecture: 10 input -> 16 hidden -> 8 hidden -> 1 output
nn = NeuralNetwork(layers=[10, 16, 8, 1], learning_rate=0.01, epochs=1000)

print(f"Network architecture: {nn.layers}")
print(f"Total parameters: {sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)}")

print("\\nTraining neural network...")
nn.fit(X_train_scaled, y_train)

# Make predictions
train_predictions = nn.predict(X_train_scaled)
test_predictions = nn.predict(X_test_scaled)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f"\\nTraining Results:")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualizations
print("\\nCreating visualizations...")

# 1. Training curve
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(nn.costs)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True)

# 2. Prediction probabilities
plt.subplot(1, 3, 2)
train_probs = nn.predict_proba(X_train_scaled).flatten()
test_probs = nn.predict_proba(X_test_scaled).flatten()

plt.hist(train_probs[y_train.flatten() == 0], alpha=0.7, bins=30, label='Class 0', density=True)
plt.hist(train_probs[y_train.flatten() == 1], alpha=0.7, bins=30, label='Class 1', density=True)
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# 3. Weight distributions for first layer
plt.subplot(1, 3, 3)
first_layer_weights = nn.weights[0].flatten()
plt.hist(first_layer_weights, bins=30, alpha=0.7, edgecolor='black')
plt.title('First Layer Weight Distribution')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
# plt.show()

print("Neural network training completed successfully!")

# Mathematical explanation
print("\\n" + "=" * 60)
print("NEURAL NETWORK MATHEMATICS")
print("=" * 60)

math_explanation = '''
Neural Network Mathematical Foundation:

1. Forward Propagation:
   For each layer l:
   z^[l] = W^[l] * a^[l-1] + b^[l]    (linear transformation)
   a^[l] = g^[l](z^[l])               (activation function)

2. Activation Functions:
   - Sigmoid: σ(z) = 1/(1 + e^(-z))
   - ReLU: ReLU(z) = max(0, z)
   - Tanh: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))

3. Cost Function (Binary Cross-Entropy):
   J = -(1/m) Σ [y*log(a) + (1-y)*log(1-a)]

4. Backward Propagation (Chain Rule):
   δ^[L] = (a^[L] - y) ⊙ g'^[L](z^[L])     (output layer error)
   δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'^[l](z^[l])  (hidden layer error)

5. Gradient Computation:
   ∂J/∂W^[l] = (1/m) * δ^[l] * (a^[l-1])^T
   ∂J/∂b^[l] = (1/m) * Σ δ^[l]

6. Parameter Update:
   W^[l] := W^[l] - α * ∂J/∂W^[l]
   b^[l] := b^[l] - α * ∂J/∂b^[l]

Key Concepts:
- Universal Approximation Theorem: Neural networks can approximate any continuous function
- Backpropagation: Efficient algorithm for computing gradients using chain rule
- Activation Functions: Introduce non-linearity, enabling complex pattern learning
- Overfitting: Model memorizes training data, use regularization/dropout
'''

print(math_explanation)`}</code></pre>
          </div>
        </section>

        <section className="advanced-architectures">
          <h2>🏗️ Advanced Neural Network Architectures</h2>
          
          <div className="architecture-grid">
            <div className="arch-card">
              <h3>🖼️ Convolutional Neural Networks (CNNs)</h3>
              <p><strong>Best for:</strong> Image processing, computer vision</p>
              <p><strong>Key Components:</strong></p>
              <ul>
                <li>Convolutional layers (feature detection)</li>
                <li>Pooling layers (dimensionality reduction)</li>
                <li>Fully connected layers (classification)</li>
              </ul>
              <p><strong>Applications:</strong> Image recognition, medical imaging, autonomous vehicles</p>
            </div>

            <div className="arch-card">
              <h3>🔄 Recurrent Neural Networks (RNNs)</h3>
              <p><strong>Best for:</strong> Sequential data, time series</p>
              <p><strong>Variants:</strong></p>
              <ul>
                <li>LSTM (Long Short-Term Memory)</li>
                <li>GRU (Gated Recurrent Unit)</li>
                <li>Bidirectional RNNs</li>
              </ul>
              <p><strong>Applications:</strong> Natural language processing, speech recognition, stock prediction</p>
            </div>

            <div className="arch-card">
              <h3>🤖 Transformers</h3>
              <p><strong>Best for:</strong> Natural language understanding</p>
              <p><strong>Key Innovation:</strong></p>
              <ul>
                <li>Self-attention mechanism</li>
                <li>Parallel processing</li>
                <li>Position encoding</li>
              </ul>
              <p><strong>Applications:</strong> GPT, BERT, machine translation, ChatGPT</p>
            </div>

            <div className="arch-card">
              <h3>🎭 Generative Adversarial Networks (GANs)</h3>
              <p><strong>Best for:</strong> Generating new data</p>
              <p><strong>Components:</strong></p>
              <ul>
                <li>Generator (creates fake data)</li>
                <li>Discriminator (detects fake data)</li>
                <li>Adversarial training</li>
              </ul>
              <p><strong>Applications:</strong> Image generation, art creation, data augmentation</p>
            </div>
          </div>
        </section>

        <section className="practical-tips">
          <h2>💡 Practical Deep Learning Tips</h2>
          
          <div className="tips-grid">
            <div className="tip-category">
              <h3>🎯 Model Design</h3>
              <ul>
                <li>Start simple, then add complexity</li>
                <li>Use appropriate architecture for your data type</li>
                <li>Consider transfer learning for small datasets</li>
                <li>Balance model capacity with available data</li>
              </ul>
            </div>

            <div className="tip-category">
              <h3>🔧 Training Optimization</h3>
              <ul>
                <li>Use proper weight initialization (Xavier/He)</li>
                <li>Apply batch normalization for stable training</li>
                <li>Use dropout to prevent overfitting</li>
                <li>Monitor validation loss for early stopping</li>
              </ul>
            </div>

            <div className="tip-category">
              <h3>📊 Hyperparameter Tuning</h3>
              <ul>
                <li>Learning rate: Start with 0.001-0.01</li>
                <li>Batch size: Powers of 2 (32, 64, 128)</li>
                <li>Network depth: Add layers gradually</li>
                <li>Use learning rate scheduling</li>
              </ul>
            </div>

            <div className="tip-category">
              <h3>🚀 Performance Optimization</h3>
              <ul>
                <li>Use GPU acceleration (CUDA/OpenCL)</li>
                <li>Implement efficient data pipelines</li>
                <li>Use mixed precision training</li>
                <li>Optimize inference for production</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="frameworks">
          <h2>🛠️ Deep Learning Frameworks</h2>
          
          <div className="framework-comparison">
            <div className="framework-card">
              <h3>🔥 PyTorch</h3>
              <p><strong>Pros:</strong> Dynamic graphs, Pythonic, great for research</p>
              <p><strong>Cons:</strong> Smaller ecosystem than TensorFlow</p>
              <p><strong>Best for:</strong> Research, prototyping, custom architectures</p>
            </div>

            <div className="framework-card">
              <h3>🧠 TensorFlow/Keras</h3>
              <p><strong>Pros:</strong> Large ecosystem, production ready, TensorBoard</p>
              <p><strong>Cons:</strong> Steeper learning curve</p>
              <p><strong>Best for:</strong> Production deployment, large scale</p>
            </div>

            <div className="framework-card">
              <h3>⚡ JAX</h3>
              <p><strong>Pros:</strong> Fast compilation, functional programming</p>
              <p><strong>Cons:</strong> Smaller community</p>
              <p><strong>Best for:</strong> High-performance computing, research</p>
            </div>
          </div>
        </section>

        <div className="next-steps">
          <h2>🎯 Next Steps in Your Deep Learning Journey</h2>
          <ol>
            <li><strong>Master the fundamentals:</strong> Linear algebra, calculus, probability</li>
            <li><strong>Practice with datasets:</strong> MNIST, CIFAR-10, ImageNet</li>
            <li><strong>Build projects:</strong> Image classifier, chatbot, recommender system</li>
            <li><strong>Learn a framework:</strong> Start with PyTorch or TensorFlow</li>
            <li><strong>Stay updated:</strong> Follow papers, conferences (NeurIPS, ICML, ICLR)</li>
            <li><strong>Join communities:</strong> Reddit ML, Twitter, Discord servers</li>
          </ol>
        </div>
      </div>
    </div>
  )
}

export default DeepLearning