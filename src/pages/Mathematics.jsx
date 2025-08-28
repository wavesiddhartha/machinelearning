function Mathematics() {
  return (
    <div className="page">
      <div className="content">
        <h1>📐 Mathematics for Machine Learning</h1>
        
        <section className="math-intro">
          <h2>Why Mathematics is Crucial for ML</h2>
          <p>Mathematics is the foundation of machine learning. Understanding the mathematical concepts helps you:</p>
          <ul>
            <li>Choose the right algorithms for your problem</li>
            <li>Debug and improve model performance</li>
            <li>Understand why algorithms work</li>
            <li>Create custom solutions</li>
            <li>Optimize hyperparameters effectively</li>
          </ul>
        </section>

        <section className="linear-algebra">
          <h2>🔢 Linear Algebra</h2>
          <p>Linear algebra is the backbone of machine learning, dealing with vectors, matrices, and their operations.</p>
          
          <h3>Vectors</h3>
          <div className="math-concept">
            <p><strong>Definition:</strong> A vector is an ordered list of numbers representing magnitude and direction.</p>
            
            <div className="code-example">
              <h4>Vectors in Python</h4>
              <pre><code>{`import numpy as np

# Create vectors
v1 = np.array([1, 2, 3])      # 3D vector
v2 = np.array([4, 5, 6])      # Another 3D vector

# Vector operations
addition = v1 + v2             # [5, 7, 9]
subtraction = v1 - v2          # [-3, -3, -3]
scalar_mult = 3 * v1           # [3, 6, 9]

# Dot product (very important in ML)
dot_product = np.dot(v1, v2)   # 1*4 + 2*5 + 3*6 = 32

# Magnitude (Euclidean norm)
magnitude_v1 = np.linalg.norm(v1)  # √(1² + 2² + 3²) = √14

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"Dot product: {dot_product}")
print(f"Magnitude of v1: {magnitude_v1:.3f}")

# Cosine similarity (used in recommendation systems)
cosine_sim = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"Cosine similarity: {cosine_sim:.3f}")`}</code></pre>
            </div>
            
            <div className="math-explanation">
              <h4>Mathematical Notation:</h4>
              <p><strong>Dot Product:</strong> v₁ · v₂ = Σᵢ(v₁ᵢ × v₂ᵢ)</p>
              <p><strong>Magnitude:</strong> ||v|| = √(Σᵢ vᵢ²)</p>
              <p><strong>Cosine Similarity:</strong> cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)</p>
            </div>
          </div>

          <h3>Matrices</h3>
          <div className="math-concept">
            <p><strong>Definition:</strong> A matrix is a 2D array of numbers. Essential for representing datasets and transformations.</p>
            
            <div className="code-example">
              <h4>Matrix Operations</h4>
              <pre><code>{`import numpy as np

# Create matrices
A = np.array([[1, 2], 
              [3, 4]])      # 2x2 matrix

B = np.array([[5, 6],
              [7, 8]])      # Another 2x2 matrix

# Matrix operations
addition = A + B              # Element-wise addition
multiplication = A @ B        # Matrix multiplication (not element-wise!)
transpose = A.T               # Transpose
inverse = np.linalg.inv(A)    # Inverse (if exists)

print("Matrix A:")
print(A)
print("\\nMatrix B:")
print(B)
print("\\nA @ B (matrix multiplication):")
print(multiplication)
print("\\nTranspose of A:")
print(transpose)

# Eigenvalues and eigenvectors (used in PCA)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\\nEigenvalues: {eigenvalues}")
print("Eigenvectors:")
print(eigenvectors)

# Dataset as matrix (very common in ML)
# Rows = samples, Columns = features
dataset = np.array([
    [1.2, 0.8, 2.1],    # Sample 1: feature1, feature2, feature3
    [0.9, 1.5, 1.8],    # Sample 2
    [2.1, 0.3, 2.9],    # Sample 3
    [1.7, 1.1, 2.2]     # Sample 4
])

print(f"\\nDataset shape: {dataset.shape}")  # (4 samples, 3 features)
print(f"Feature means: {np.mean(dataset, axis=0)}")  # Mean of each feature`}</code></pre>
            </div>
            
            <div className="math-explanation">
              <h4>Key Matrix Properties:</h4>
              <p><strong>Matrix Multiplication:</strong> (A × B)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ</p>
              <p><strong>Transpose:</strong> (Aᵀ)ᵢⱼ = Aⱼᵢ</p>
              <p><strong>Identity Matrix:</strong> AI = IA = A</p>
              <p><strong>Inverse:</strong> AA⁻¹ = A⁻¹A = I (if A is invertible)</p>
            </div>
          </div>
        </section>

        <section className="calculus">
          <h2>📊 Calculus</h2>
          <p>Calculus helps us understand how functions change, which is crucial for optimization in machine learning.</p>
          
          <h3>Derivatives - The Heart of Optimization</h3>
          <div className="math-concept">
            <p><strong>Derivative:</strong> Measures how a function changes as its input changes. Essential for gradient descent.</p>
            
            <div className="code-example">
              <h4>Derivatives in Practice</h4>
              <pre><code>{`import numpy as np
import matplotlib.pyplot as plt

# Define a simple function: f(x) = x²
def f(x):
    return x**2

# Its derivative: f'(x) = 2x
def df_dx(x):
    return 2*x

# Numerical derivative approximation
def numerical_derivative(func, x, h=1e-8):
    return (func(x + h) - func(x - h)) / (2 * h)

# Test points
x_vals = np.linspace(-3, 3, 100)
y_vals = f(x_vals)
dy_vals = df_dx(x_vals)

# Example: Find minimum using gradient descent
def gradient_descent(start_x, learning_rate=0.1, iterations=20):
    x = start_x
    path = [x]
    
    for i in range(iterations):
        gradient = df_dx(x)
        x = x - learning_rate * gradient  # Move opposite to gradient
        path.append(x)
        print(f"Iteration {i+1}: x={x:.4f}, f(x)={f(x):.4f}, gradient={gradient:.4f}")
    
    return path

print("Gradient Descent to find minimum of x²:")
path = gradient_descent(start_x=2.5, learning_rate=0.3, iterations=10)

# Loss function example (Mean Squared Error)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Derivative of MSE with respect to predictions
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Example data
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.8, 4.2, 4.9])

loss = mse_loss(y_true, y_pred)
grad = mse_derivative(y_true, y_pred)

print(f"\\nMSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")`}</code></pre>
            </div>
            
            <div className="math-explanation">
              <h4>Key Derivative Rules:</h4>
              <p><strong>Power Rule:</strong> d/dx(xⁿ) = n·xⁿ⁻¹</p>
              <p><strong>Chain Rule:</strong> d/dx[f(g(x))] = f'(g(x)) · g'(x)</p>
              <p><strong>Product Rule:</strong> d/dx[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)</p>
            </div>
          </div>

          <h3>Partial Derivatives - Multivariable Functions</h3>
          <div className="math-concept">
            <p>Most ML functions depend on multiple variables. Partial derivatives tell us how the function changes with respect to each variable.</p>
            
            <div className="code-example">
              <h4>Partial Derivatives Example</h4>
              <pre><code>{`# Function of two variables: f(x,y) = x² + y² + 2xy
def f(x, y):
    return x**2 + y**2 + 2*x*y

# Partial derivatives
def df_dx(x, y):
    return 2*x + 2*y  # ∂f/∂x

def df_dy(x, y):  
    return 2*y + 2*x  # ∂f/∂y

# Gradient vector [∂f/∂x, ∂f/∂y]
def gradient(x, y):
    return np.array([df_dx(x, y), df_dy(x, y)])

# Example: Linear regression cost function
# J(w, b) = (1/2m) * Σ(h(x) - y)²
# where h(x) = wx + b

def linear_regression_cost(w, b, X, y):
    m = len(y)
    predictions = w * X + b
    cost = np.sum((predictions - y)**2) / (2 * m)
    return cost

def cost_gradients(w, b, X, y):
    m = len(y)
    predictions = w * X + b
    dw = np.sum((predictions - y) * X) / m      # ∂J/∂w
    db = np.sum(predictions - y) / m           # ∂J/∂b
    return dw, db

# Example with sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
w, b = 0.5, 0.5  # Initial parameters

cost = linear_regression_cost(w, b, X, y)
dw, db = cost_gradients(w, b, X, y)

print(f"Initial cost: {cost:.4f}")
print(f"Gradient w.r.t w: {dw:.4f}")  
print(f"Gradient w.r.t b: {db:.4f}")

# Gradient descent update
learning_rate = 0.01
w_new = w - learning_rate * dw
b_new = b - learning_rate * db
print(f"Updated parameters: w={w_new:.4f}, b={b_new:.4f}")`}</code></pre>
            </div>
          </div>
        </section>

        <section className="statistics">
          <h2>📈 Statistics and Probability</h2>
          <p>Statistics and probability provide the theoretical foundation for understanding data and model behavior.</p>
          
          <h3>Descriptive Statistics</h3>
          <div className="math-concept">
            <div className="code-example">
              <h4>Statistical Measures</h4>
              <pre><code>{`import numpy as np
from scipy import stats

# Sample dataset
data = np.array([85, 90, 78, 92, 88, 76, 95, 89, 84, 91])

# Central tendency
mean = np.mean(data)
median = np.median(data)  
mode_result = stats.mode(data)

# Variability
variance = np.var(data, ddof=1)  # Sample variance
std_dev = np.std(data, ddof=1)   # Sample standard deviation
range_val = np.max(data) - np.min(data)

# Quartiles
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1  # Interquartile range

print(f"Dataset: {data}")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Range: {range_val}")
print(f"IQR: {iqr}")

# Z-scores (standardization)
z_scores = (data - mean) / std_dev
print(f"\\nZ-scores: {z_scores}")

# Detect outliers (|z-score| > 2)
outliers = data[np.abs(z_scores) > 2]
print(f"Outliers: {outliers}")

# Correlation between two variables
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
correlation = np.corrcoef(x, y)[0, 1]
print(f"\\nCorrelation between x and y: {correlation:.3f}")

# Covariance
covariance = np.cov(x, y)[0, 1]
print(f"Covariance: {covariance:.3f}")`}</code></pre>
            </div>
            
            <div className="math-explanation">
              <h4>Key Formulas:</h4>
              <p><strong>Mean:</strong> μ = (Σxᵢ) / n</p>
              <p><strong>Variance:</strong> σ² = Σ(xᵢ - μ)² / n</p>
              <p><strong>Standard Deviation:</strong> σ = √σ²</p>
              <p><strong>Z-score:</strong> z = (x - μ) / σ</p>
              <p><strong>Correlation:</strong> r = Cov(X,Y) / (σₓ × σᵧ)</p>
            </div>
          </div>

          <h3>Probability Distributions</h3>
          <div className="math-concept">
            <div className="code-example">
              <h4>Common Distributions in ML</h4>
              <pre><code>{`import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Normal Distribution (Gaussian)
# Very common in ML - many phenomena follow this distribution
mu, sigma = 0, 1  # Standard normal distribution
normal_dist = stats.norm(mu, sigma)

# Generate random samples
samples = normal_dist.rvs(1000)
print(f"Normal distribution samples (first 10): {samples[:10]}")

# Probability density function
x = np.linspace(-3, 3, 100)
pdf = normal_dist.pdf(x)
print(f"PDF at x=0: {normal_dist.pdf(0):.4f}")

# Cumulative distribution function  
cdf_at_0 = normal_dist.cdf(0)
print(f"CDF at x=0: {cdf_at_0:.4f}")  # Should be 0.5

# Bernoulli Distribution (for binary classification)
p = 0.7  # Probability of success
bernoulli_dist = stats.bernoulli(p)
binary_samples = bernoulli_dist.rvs(20)
print(f"\\nBinary samples (0s and 1s): {binary_samples}")

# Binomial Distribution (number of successes in n trials)
n, p = 10, 0.3
binomial_dist = stats.binom(n, p)
binom_samples = binomial_dist.rvs(10)
print(f"Binomial samples: {binom_samples}")

# Expected value and variance
print(f"Binomial expected value: {binomial_dist.mean()}")
print(f"Binomial variance: {binomial_dist.var()}")

# Uniform Distribution (random initialization in neural networks)
uniform_samples = np.random.uniform(-1, 1, 10)  # Between -1 and 1
print(f"\\nUniform samples: {uniform_samples}")

# Central Limit Theorem demonstration
def clt_demo(sample_size, num_experiments):
    # Draw samples from any distribution (e.g., exponential)
    sample_means = []
    
    for _ in range(num_experiments):
        sample = np.random.exponential(2, sample_size)  # Not normal!
        sample_means.append(np.mean(sample))
    
    return np.array(sample_means)

# The distribution of sample means will be approximately normal!
means = clt_demo(30, 1000)
print(f"\\nSample means statistics:")
print(f"Mean of means: {np.mean(means):.3f}")
print(f"Std of means: {np.std(means):.3f}")

# Test for normality
statistic, p_value = stats.shapiro(means)
print(f"Normality test p-value: {p_value:.6f}")
if p_value > 0.05:
    print("Sample means follow normal distribution! (CLT confirmed)")`}</code></pre>
            </div>
          </div>
        </section>

        <section className="optimization">
          <h2>🎯 Optimization</h2>
          <p>Optimization is how we train machine learning models - finding the best parameters that minimize error.</p>
          
          <h3>Gradient Descent</h3>
          <div className="math-concept">
            <p>The fundamental optimization algorithm in machine learning.</p>
            
            <div className="code-example">
              <h4>Implementing Gradient Descent</h4>
              <pre><code>{`import numpy as np
import matplotlib.pyplot as plt

# Complete gradient descent implementation
class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        
    def compute_cost(self, X, y, theta):
        """Compute cost function (MSE)"""
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def compute_gradients(self, X, y, theta):
        """Compute gradients"""
        m = len(y)
        predictions = X.dot(theta)
        gradients = (1/m) * X.T.dot(predictions - y)
        return gradients
    
    def fit(self, X, y):
        """Fit the model using gradient descent"""
        # Add bias term (intercept)
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])
        
        # Initialize parameters
        theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        for i in range(self.max_iterations):
            # Compute cost and gradients
            cost = self.compute_cost(X_with_bias, y, theta)
            gradients = self.compute_gradients(X_with_bias, y, theta)
            
            # Update parameters
            theta = theta - self.learning_rate * gradients
            
            # Store cost
            self.cost_history.append(cost)
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
        
        self.theta = theta
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_bias.dot(self.theta)

# Example: Linear regression with gradient descent
np.random.seed(42)
m = 100  # Number of samples

# Generate synthetic data
X = np.random.randn(m, 1)  # Features
y = 4 + 3 * X.flatten() + np.random.randn(m) * 0.5  # y = 4 + 3x + noise

# Create and train model
model = GradientDescent(learning_rate=0.1, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Results
print(f"\\nFinal parameters:")
print(f"Bias (intercept): {model.theta[0]:.4f} (true: 4.0)")
print(f"Weight: {model.theta[1]:.4f} (true: 3.0)")
print(f"Final cost: {model.cost_history[-1]:.6f}")

# Different gradient descent variants
class AdvancedOptimizers:
    @staticmethod
    def sgd_with_momentum(X, y, learning_rate=0.01, momentum=0.9, epochs=100):
        """Stochastic Gradient Descent with Momentum"""
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])
        theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
        velocity = np.zeros_like(theta)
        
        for epoch in range(epochs):
            # Shuffle data for SGD
            indices = np.random.permutation(m)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                # Single sample gradient
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                prediction = xi.dot(theta)
                gradient = xi.T.dot(prediction - yi)
                
                # Momentum update
                velocity = momentum * velocity + learning_rate * gradient.flatten()
                theta = theta - velocity
        
        return theta
    
    @staticmethod  
    def adam_optimizer(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epochs=100):
        """Adam optimizer"""
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])
        theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        # Initialize moments
        m_t = np.zeros_like(theta)  # First moment
        v_t = np.zeros_like(theta)  # Second moment
        epsilon = 1e-8
        
        for t in range(1, epochs + 1):
            # Compute gradients (using full batch for simplicity)
            predictions = X_with_bias.dot(theta)
            gradients = (1/m) * X_with_bias.T.dot(predictions - y)
            
            # Update biased first moment estimate
            m_t = beta1 * m_t + (1 - beta1) * gradients
            
            # Update biased second moment estimate  
            v_t = beta2 * v_t + (1 - beta2) * (gradients ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m_t / (1 - beta1 ** t)
            
            # Compute bias-corrected second moment estimate
            v_hat = v_t / (1 - beta2 ** t)
            
            # Update parameters
            theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return theta

# Compare optimizers
theta_momentum = AdvancedOptimizers.sgd_with_momentum(X, y)
theta_adam = AdvancedOptimizers.adam_optimizer(X, y)

print(f"\\nComparison of optimizers:")
print(f"Standard GD - Bias: {model.theta[0]:.4f}, Weight: {model.theta[1]:.4f}")
print(f"SGD+Momentum - Bias: {theta_momentum[0]:.4f}, Weight: {theta_momentum[1]:.4f}")
print(f"Adam - Bias: {theta_adam[0]:.4f}, Weight: {theta_adam[1]:.4f}")`}</code></pre>
            </div>
            
            <div className="math-explanation">
              <h4>Gradient Descent Update Rule:</h4>
              <p><strong>θ := θ - α∇J(θ)</strong></p>
              <p>Where: θ = parameters, α = learning rate, ∇J(θ) = gradient of cost function</p>
              
              <h4>Momentum Update:</h4>
              <p><strong>v := βv + α∇J(θ)</strong></p>
              <p><strong>θ := θ - v</strong></p>
              
              <h4>Adam Update:</h4>
              <p><strong>m := β₁m + (1-β₁)∇J(θ)</strong> (first moment)</p>
              <p><strong>v := β₂v + (1-β₂)(∇J(θ))²</strong> (second moment)</p>
            </div>
          </div>
        </section>

        <section className="information-theory">
          <h2>📡 Information Theory</h2>
          <p>Information theory provides the mathematical foundation for understanding entropy, information gain, and uncertainty - crucial for decision trees and neural networks.</p>
          
          <div className="code-example">
            <h4>Entropy and Information Gain</h4>
            <pre><code>{`import numpy as np
from math import log2

def entropy(labels):
    """Calculate entropy of a set of labels"""
    if len(labels) == 0:
        return 0
    
    # Count occurrences of each class
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    # Calculate entropy: H(S) = -Σ p(x) * log₂(p(x))
    entropy_val = -np.sum([p * log2(p) if p > 0 else 0 for p in probabilities])
    return entropy_val

def information_gain(parent_labels, left_labels, right_labels):
    """Calculate information gain from a split"""
    parent_entropy = entropy(parent_labels)
    
    # Weighted average of child entropies
    n_parent = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    
    weighted_child_entropy = (n_left/n_parent) * entropy(left_labels) + \\
                           (n_right/n_parent) * entropy(right_labels)
    
    return parent_entropy - weighted_child_entropy

# Example: Binary classification dataset
labels = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])  # 6 ones, 4 zeros

print(f"Dataset: {labels}")
print(f"Entropy: {entropy(labels):.4f}")

# Perfect split example
left_split = np.array([1, 1, 1, 1, 1])   # All ones
right_split = np.array([0, 0, 0, 0, 0])  # All zeros

ig = information_gain(labels, left_split, right_split)
print(f"\\nPerfect split:")
print(f"Left entropy: {entropy(left_split):.4f}")
print(f"Right entropy: {entropy(right_split):.4f}")
print(f"Information gain: {ig:.4f}")

# Cross-entropy loss (used in neural networks)
def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """Binary cross-entropy loss"""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Model predictions

loss = cross_entropy_loss(y_true, y_pred)
print(f"\\nCross-entropy loss: {loss:.4f}")

# KL Divergence (measures difference between probability distributions)
def kl_divergence(p, q, epsilon=1e-15):
    """Kullback-Leibler divergence"""
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))

# Compare two probability distributions
p = np.array([0.7, 0.2, 0.1])  # True distribution
q = np.array([0.6, 0.3, 0.1])  # Approximate distribution

kl_div = kl_divergence(p, q)
print(f"\\nKL divergence: {kl_div:.4f}")
print("(Lower values mean distributions are more similar)")`}</code></pre>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Mathematics