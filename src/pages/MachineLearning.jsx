function MachineLearning() {
  return (
    <div className="page">
      <div className="content">
        <h1>🤖 Machine Learning Fundamentals</h1>
        
        <section className="ml-intro">
          <h2>What is Machine Learning?</h2>
          <p>Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It's about finding patterns in data and making predictions.</p>
          
          <div className="ml-types-grid">
            <div className="ml-type-card">
              <h3>🎯 Supervised Learning</h3>
              <p>Learn from labeled examples</p>
              <ul>
                <li>Classification (categories)</li>
                <li>Regression (continuous values)</li>
              </ul>
            </div>
            <div className="ml-type-card">
              <h3>🔍 Unsupervised Learning</h3>
              <p>Find hidden patterns in unlabeled data</p>
              <ul>
                <li>Clustering</li>
                <li>Dimensionality Reduction</li>
              </ul>
            </div>
            <div className="ml-type-card">
              <h3>🎮 Reinforcement Learning</h3>
              <p>Learn through trial and error</p>
              <ul>
                <li>Agent-Environment Interaction</li>
                <li>Reward-based Learning</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="supervised-learning">
          <h2>📊 Supervised Learning</h2>
          
          <h3>🎯 Classification</h3>
          <p>Predict categories or classes. Examples: spam detection, image recognition, medical diagnosis.</p>
          
          <div className="code-example">
            <h4>Logistic Regression - Mathematical Foundation</h4>
            <pre><code>{`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LogisticRegressionFromScratch:
    """Implement Logistic Regression from scratch to understand the math"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.costs = []
    
    def sigmoid(self, z):
        """Sigmoid activation function: σ(z) = 1/(1 + e^(-z))"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = X.dot(self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Cost function (Cross-entropy loss)
            # J = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
            epsilon = 1e-15  # Prevent log(0)
            predictions = np.clip(predictions, epsilon, 1-epsilon)
            cost = -(1/n_samples) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
            self.costs.append(cost)
            
            # Backward pass (compute gradients)
            dw = (1/n_samples) * X.T.dot(predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions"""
        linear_pred = X.dot(self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        return (predictions >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        linear_pred = X.dot(self.weights) + self.bias
        return self.sigmoid(linear_pred)

# Generate sample data
print("Creating sample classification dataset...")
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Train our custom logistic regression
print("\\nTraining custom logistic regression...")
custom_lr = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
custom_lr.fit(X_train_scaled, y_train)

# Make predictions
train_predictions = custom_lr.predict(X_train_scaled)
test_predictions = custom_lr.predict(X_test_scaled)

print(f"Custom LR - Train Accuracy: {accuracy_score(y_train, train_predictions):.4f}")
print(f"Custom LR - Test Accuracy: {accuracy_score(y_test, test_predictions):.4f}")

# Compare with sklearn
print("\\nComparing with scikit-learn...")
sklearn_lr = LogisticRegression()
sklearn_lr.fit(X_train_scaled, y_train)

sklearn_train_pred = sklearn_lr.predict(X_train_scaled)
sklearn_test_pred = sklearn_lr.predict(X_test_scaled)

print(f"Sklearn LR - Train Accuracy: {accuracy_score(y_train, sklearn_train_pred):.4f}")
print(f"Sklearn LR - Test Accuracy: {accuracy_score(y_test, sklearn_test_pred):.4f}")

# Mathematical explanation
print("\\n" + "=" * 60)
print("MATHEMATICAL FOUNDATION")
print("=" * 60)

explanation = '''
Logistic Regression Mathematics:

1. Linear Combination:
   z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w^T x + b

2. Sigmoid Function:
   σ(z) = 1 / (1 + e^(-z))
   - Maps any real number to (0, 1)
   - Represents probability

3. Prediction:
   ŷ = 1 if σ(z) ≥ 0.5 else 0

4. Cost Function (Cross-entropy):
   J(w) = -(1/m) Σ [y⁽ⁱ⁾ log(h⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-h⁽ⁱ⁾)]
   
   Where h⁽ⁱ⁾ = σ(w^T x⁽ⁱ⁾ + b)

5. Gradient Descent Update:
   w := w - α * ∇J(w)
   
   ∇J(w) = (1/m) X^T (h - y)

Why Cross-entropy Loss?
- Convex function (guaranteed global minimum)
- Heavily penalizes confident wrong predictions
- Derivative has nice form: (prediction - actual)
'''

print(explanation)

# Visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    """Plot decision boundary for 2D data"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(mesh_points)
    else:
        Z = model.predict(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdGy')
    plt.colorbar(label='Prediction Probability')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdGy', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    # plt.show()

# Plot decision boundaries
print("\\nCreating decision boundary visualizations...")
plot_decision_boundary(X_test_scaled, y_test, custom_lr, 'Custom Logistic Regression Decision Boundary')

# Cost function visualization
plt.figure(figsize=(10, 6))
plt.plot(custom_lr.costs)
plt.title('Cost Function During Training')
plt.xlabel('Iteration')
plt.ylabel('Cost (Cross-entropy Loss)')
plt.grid(True)
# plt.show()

print("Visualization plots created successfully!")

# Detailed evaluation
print("\\n" + "=" * 60)
print("DETAILED MODEL EVALUATION")
print("=" * 60)

print("\\nConfusion Matrix:")
cm = confusion_matrix(y_test, test_predictions)
print(cm)

print("\\nClassification Report:")
print(classification_report(y_test, test_predictions))

# Feature importance (weights)
print(f"\\nModel Weights:")
print(f"Feature 1 weight: {custom_lr.weights[0]:.4f}")
print(f"Feature 2 weight: {custom_lr.weights[1]:.4f}")
print(f"Bias term: {custom_lr.bias:.4f}")

print("\\nInterpretation:")
print("- Positive weights increase probability of class 1")
print("- Negative weights decrease probability of class 1")
print("- Magnitude indicates feature importance")`}</code></pre>
          </div>

          <h3>📈 Regression</h3>
          <p>Predict continuous values. Examples: house prices, stock prices, temperature.</p>
          
          <div className="code-example">
            <h4>Linear Regression - Complete Implementation</h4>
            <pre><code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionFromScratch:
    """Complete Linear Regression implementation with mathematical details"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization  # None, 'l1', 'l2'
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.costs = []
    
    def add_regularization(self, cost, weights):
        """Add regularization to cost function"""
        if self.regularization == 'l1':
            # L1 regularization (Lasso): λ Σ|wᵢ|
            reg_term = self.lambda_reg * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            # L2 regularization (Ridge): λ Σwᵢ²
            reg_term = self.lambda_reg * np.sum(weights ** 2)
        else:
            reg_term = 0
        
        return cost + reg_term
    
    def compute_regularization_gradient(self, weights):
        """Compute regularization gradient"""
        if self.regularization == 'l1':
            return self.lambda_reg * np.sign(weights)
        elif self.regularization == 'l2':
            return 2 * self.lambda_reg * weights
        else:
            return np.zeros_like(weights)
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass: y_pred = X * w + b
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute cost (Mean Squared Error)
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            cost = self.add_regularization(cost, self.weights)
            self.costs.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Add regularization to weight gradients
            dw += self.compute_regularization_gradient(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping if cost increases (sign of overfitting)
            if i > 10 and self.costs[-1] > self.costs[-2]:
                break
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.weights) + self.bias
    
    def analytical_solution(self, X, y):
        """Compute analytical solution: w = (X^T X)^(-1) X^T y"""
        try:
            # Add bias column to X
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Normal equation
            theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
            
            self.bias = theta[0]
            self.weights = theta[1:]
            
            return True
        except np.linalg.LinAlgError:
            print("Matrix is singular, using gradient descent instead")
            return False

# Generate sample data
print("Creating sample regression dataset...")
X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)

# Add some polynomial features for complexity
X_poly = np.column_stack([X, X[:, 0]**2, X[:, 1]**2])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Features: {X_train_scaled.shape[1]}")

# Train different models
models = {
    'Linear (Gradient Descent)': LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000),
    'Ridge Regression': LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000, 
                                                   regularization='l2', lambda_reg=0.1),
    'Lasso Regression': LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000,
                                                   regularization='l1', lambda_reg=0.1)
}

results = {}

print("\\n" + "=" * 60)
print("TRAINING DIFFERENT REGRESSION MODELS")
print("=" * 60)

for name, model in models.items():
    print(f"\\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    train_pred_scaled = model.predict(X_train_scaled)
    test_pred_scaled = model.predict(X_test_scaled)
    
    # Convert back to original scale
    train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'weights': model.weights.copy(),
        'costs': model.costs.copy()
    }
    
    print(f"  Train MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
    print(f"  Test MSE: {test_mse:.2f}, R²: {test_r2:.4f}")

# Compare with sklearn
print("\\nComparing with scikit-learn...")
sklearn_lr = LinearRegression()
sklearn_lr.fit(X_train_scaled, y_train_scaled)
sklearn_pred = scaler_y.inverse_transform(sklearn_lr.predict(X_test_scaled).reshape(-1, 1)).flatten()
sklearn_mse = mean_squared_error(y_test, sklearn_pred)
sklearn_r2 = r2_score(y_test, sklearn_pred)
print(f"Sklearn Linear Regression - Test MSE: {sklearn_mse:.2f}, R²: {sklearn_r2:.4f}")

# Visualizations
print("\\nCreating visualizations...")

# 1. Cost function evolution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for name, result in results.items():
    plt.plot(result['costs'][:100], label=name)  # First 100 iterations
plt.title('Cost Function Evolution')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.legend()
plt.grid(True)

# 2. Predictions vs Actual
plt.subplot(1, 3, 2)
model_name = 'Linear (Gradient Descent)'
model = models[model_name]
test_pred_scaled = model.predict(X_test_scaled)
test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

plt.scatter(y_test, test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual')
plt.grid(True)

# 3. Residuals plot
plt.subplot(1, 3, 3)
residuals = y_test - test_pred
plt.scatter(test_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

plt.tight_layout()
# plt.show()

# Mathematical explanation
print("\\n" + "=" * 60)
print("MATHEMATICAL FOUNDATION")
print("=" * 60)

math_explanation = '''
Linear Regression Mathematics:

1. Model Equation:
   ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w^T x + b

2. Cost Function (Mean Squared Error):
   J(w) = (1/2m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²

3. Gradient Descent Updates:
   w := w - α * ∇J(w)
   b := b - α * ∂J/∂b
   
   Where:
   ∇J(w) = (1/m) X^T (ŷ - y)
   ∂J/∂b = (1/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)

4. Normal Equation (Analytical Solution):
   w = (X^T X)⁻¹ X^T y
   
   Pros: No learning rate, exact solution
   Cons: Slow for large datasets (O(n³))

5. Regularization:
   - Ridge (L2): J(w) = MSE + λ Σwᵢ²
   - Lasso (L1): J(w) = MSE + λ Σ|wᵢ|
   - ElasticNet: J(w) = MSE + λ₁ Σ|wᵢ| + λ₂ Σwᵢ²

6. Model Evaluation Metrics:
   - MSE: Mean Squared Error
   - RMSE: Root Mean Squared Error = √MSE
   - MAE: Mean Absolute Error
   - R²: Coefficient of Determination (1 = perfect fit)
'''

print(math_explanation)

# Feature importance analysis
print("\\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

for name, result in results.items():
    print(f"\\n{name} - Feature Weights:")
    weights = result['weights']
    for i, weight in enumerate(weights):
        print(f"  Feature {i+1}: {weight:.4f}")

print("\\nWeight Interpretation:")
print("- Positive weights: increase in feature increases prediction")
print("- Negative weights: increase in feature decreases prediction")  
print("- Magnitude indicates feature importance")
print("- Regularization shrinks weights toward zero")`}</code></pre>
          </div>
        </section>

        <section className="unsupervised-learning">
          <h2>🔍 Unsupervised Learning</h2>
          
          <h3>🎯 Clustering</h3>
          <p>Group similar data points together without labeled examples.</p>
          
          <div className="code-example">
            <h4>K-Means Clustering Implementation</h4>
            <pre><code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

class KMeansFromScratch:
    """K-Means clustering implementation with mathematical details"""
    
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4, random_state=None):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iterations_ = 0
    
    def initialize_centroids(self, X):
        """Initialize centroids using random method or k-means++"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Simple random initialization
        centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), (self.k, n_features))
        
        return centroids
    
    def kmeans_plus_plus_init(self, X):
        """K-means++ initialization for better centroid selection"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centroids
        for c_id in range(1, self.k):
            # Calculate squared distances to nearest centroid
            distances = np.array([min([np.sum((x - c)**2) for c in centroids[:c_id]]) for x in X])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    i = j
                    break
            
            centroids[c_id] = X[i]
        
        return centroids
    
    def assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid"""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)
        
        for i, point in enumerate(X):
            # Calculate Euclidean distance to each centroid
            distances = [np.sqrt(np.sum((point - centroid)**2)) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        return labels.astype(int)
    
    def update_centroids(self, X, labels):
        """Update centroids as the mean of assigned points"""
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        
        for k in range(self.k):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
            else:
                # Handle empty cluster
                centroids[k] = X[np.random.randint(X.shape[0])]
        
        return centroids
    
    def calculate_inertia(self, X, labels, centroids):
        """Calculate within-cluster sum of squared errors (WCSS)"""
        inertia = 0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k])**2)
        return inertia
    
    def fit(self, X):
        """Fit K-means clustering"""
        # Initialize centroids
        centroids = self.kmeans_plus_plus_init(X)
        
        prev_centroids = None
        
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            labels = self.assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self.update_centroids(X, labels)
            
            # Check for convergence
            if prev_centroids is not None:
                centroid_shift = np.max(np.abs(new_centroids - prev_centroids))
                if centroid_shift < self.tolerance:
                    break
            
            prev_centroids = centroids.copy()
            centroids = new_centroids
            self.n_iterations_ = iteration + 1
        
        self.centroids = centroids
        self.labels = labels
        self.inertia_ = self.calculate_inertia(X, labels, centroids)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)

# Generate sample data
print("Creating sample clustering dataset...")
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, 
                       random_state=42, center_box=(-10, 10))

print(f"Dataset shape: {X.shape}")
print(f"True number of clusters: {len(np.unique(y_true))}")

# Apply K-means
print("\\nApplying K-means clustering...")

# Try different values of k
k_values = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeansFromScratch(k=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    if k > 1:
        score = silhouette_score(X, kmeans.labels)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# Find optimal k using elbow method
def find_elbow(inertias):
    """Find elbow point in inertia curve"""
    n_points = len(inertias)
    all_coord = np.vstack((range(n_points), inertias)).T
    
    # First and last points
    first_point = all_coord[0]
    last_point = all_coord[-1]
    
    # Create line between first and last point
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # Find distance from each point to the line
    distances = []
    for coord in all_coord:
        vec_from_first = coord - first_point
        scalar_product = np.dot(vec_from_first, line_vec_norm)
        vec_from_first_parallel = scalar_product * line_vec_norm
        vec_to_line = vec_from_first - vec_from_first_parallel
        distance = np.sqrt(np.sum(vec_to_line**2))
        distances.append(distance)
    
    return np.argmax(distances) + 1

optimal_k_elbow = find_elbow(inertias)
optimal_k_silhouette = np.argmax(silhouette_scores) + 1

print(f"Optimal k (Elbow method): {optimal_k_elbow}")
print(f"Optimal k (Silhouette): {optimal_k_silhouette}")

# Train final model with optimal k
final_k = 4  # We know the true number of clusters
kmeans_final = KMeansFromScratch(k=final_k, random_state=42)
kmeans_final.fit(X)

print(f"\\nFinal model with k={final_k}:")
print(f"Iterations to converge: {kmeans_final.n_iterations_}")
print(f"Final inertia: {kmeans_final.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X, kmeans_final.labels):.3f}")

# Compare with sklearn
sklearn_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
sklearn_labels = sklearn_kmeans.fit_predict(X)

print(f"\\nComparison with scikit-learn:")
print(f"Custom K-means inertia: {kmeans_final.inertia_:.2f}")
print(f"Sklearn K-means inertia: {sklearn_kmeans.inertia_:.2f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(kmeans_final.labels, sklearn_labels):.3f}")

# Visualizations
print("\\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Original data
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
axes[0, 0].set_title('True Clusters')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# 2. K-means results
axes[0, 1].scatter(X[:, 0], X[:, 1], c=kmeans_final.labels, cmap='viridis', alpha=0.7)
axes[0, 1].scatter(kmeans_final.centroids[:, 0], kmeans_final.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
axes[0, 1].set_title('K-means Clustering Results')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')
axes[0, 1].legend()

# 3. Elbow curve
axes[0, 2].plot(k_values, inertias, 'bo-')
axes[0, 2].axvline(x=optimal_k_elbow, color='red', linestyle='--', 
                   label=f'Elbow at k={optimal_k_elbow}')
axes[0, 2].set_title('Elbow Method for Optimal k')
axes[0, 2].set_xlabel('Number of Clusters (k)')
axes[0, 2].set_ylabel('Inertia (WCSS)')
axes[0, 2].legend()
axes[0, 2].grid(True)

# 4. Silhouette scores
axes[1, 0].plot(range(2, 11), silhouette_scores[1:], 'ro-')
axes[1, 0].axvline(x=optimal_k_silhouette, color='blue', linestyle='--',
                   label=f'Max at k={optimal_k_silhouette}')
axes[1, 0].set_title('Silhouette Score vs k')
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('Silhouette Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 5. Distance from points to their centroids
cluster_distances = []
for i in range(len(X)):
    cluster = kmeans_final.labels[i]
    centroid = kmeans_final.centroids[cluster]
    distance = np.sqrt(np.sum((X[i] - centroid)**2))
    cluster_distances.append(distance)

axes[1, 1].hist(cluster_distances, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribution of Distances to Centroids')
axes[1, 1].set_xlabel('Distance to Centroid')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

# 6. Cluster sizes
cluster_sizes = [np.sum(kmeans_final.labels == i) for i in range(final_k)]
axes[1, 2].bar(range(final_k), cluster_sizes, alpha=0.7)
axes[1, 2].set_title('Cluster Sizes')
axes[1, 2].set_xlabel('Cluster')
axes[1, 2].set_ylabel('Number of Points')
axes[1, 2].grid(True)

plt.tight_layout()
# plt.show()

# Mathematical explanation
print("\\n" + "=" * 60)
print("K-MEANS MATHEMATICAL FOUNDATION")
print("=" * 60)

math_explanation = '''
K-Means Algorithm:

1. Objective Function (minimize within-cluster sum of squares):
   J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
   
   Where:
   - k = number of clusters
   - Cᵢ = set of points in cluster i
   - μᵢ = centroid of cluster i

2. Algorithm Steps:
   a) Initialize k centroids randomly (or using k-means++)
   b) Assign each point to nearest centroid:
      cluster(x) = argmin||x - μⱼ||²
   c) Update centroids as mean of assigned points:
      μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x
   d) Repeat until convergence

3. Distance Metric (Euclidean):
   d(x, μ) = √Σⱼ₌₁ⁿ (xⱼ - μⱼ)²

4. Convergence Criteria:
   - Centroids don't move significantly
   - Assignment doesn't change
   - Maximum iterations reached

5. Evaluation Metrics:
   - Inertia (WCSS): Lower is better
   - Silhouette Score: [-1, 1], higher is better
   - Calinski-Harabasz Index: Higher is better

6. Choosing Optimal k:
   - Elbow Method: Find "elbow" in inertia curve
   - Silhouette Analysis: Maximize average silhouette score
   - Gap Statistic: Compare with random data
'''

print(math_explanation)

print("\\n" + "=" * 60)
print("CLUSTERING COMPLETE!")
print("=" * 60)`}</code></pre>
          </div>
        </section>
      </div>
    </div>
  )
}

export default MachineLearning