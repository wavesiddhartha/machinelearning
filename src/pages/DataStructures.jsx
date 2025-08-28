function DataStructures() {
  return (
    <div className="page">
      <div className="content">
        <h1>📊 Data Structures & Algorithms</h1>
        
        <section className="ds-overview">
          <h2>Why Data Structures Matter for AI/ML</h2>
          <p>Efficient data structures are crucial for machine learning applications. They determine how fast your algorithms run and how much memory they use.</p>
          
          <div className="complexity-chart">
            <h3>Big O Complexity Cheat Sheet</h3>
            <table className="complexity-table">
              <thead>
                <tr>
                  <th>Operation</th>
                  <th>List</th>
                  <th>Dict</th>
                  <th>Set</th>
                  <th>Tuple</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Access</td>
                  <td>O(1)</td>
                  <td>O(1)</td>
                  <td>N/A</td>
                  <td>O(1)</td>
                </tr>
                <tr>
                  <td>Search</td>
                  <td>O(n)</td>
                  <td>O(1)</td>
                  <td>O(1)</td>
                  <td>O(n)</td>
                </tr>
                <tr>
                  <td>Insert</td>
                  <td>O(1)*</td>
                  <td>O(1)</td>
                  <td>O(1)</td>
                  <td>N/A</td>
                </tr>
                <tr>
                  <td>Delete</td>
                  <td>O(n)</td>
                  <td>O(1)</td>
                  <td>O(1)</td>
                  <td>N/A</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="ds-section">
          <h2>📝 Lists - The Workhorse of ML</h2>
          <p>Lists are ordered, mutable collections. Perfect for storing datasets, features, and predictions.</p>
          
          <div className="code-example">
            <h4>List Operations for ML</h4>
            <pre><code>{`# Creating lists for ML data
features = [1.2, 0.8, -0.5, 2.1]
labels = [1, 0, 1, 1]
model_accuracies = [0.85, 0.92, 0.78, 0.95]

# List comprehensions (very Pythonic!)
squared_features = [x**2 for x in features]
print(squared_features)  # [1.44, 0.64, 0.25, 4.41]

# Filter positive features
positive_features = [x for x in features if x > 0]
print(positive_features)  # [1.2, 0.8, 2.1]

# Nested lists for 2D data (like images)
image_matrix = [
    [255, 128, 64],
    [192, 0, 32],
    [96, 224, 160]
]

# Accessing nested data
pixel_value = image_matrix[0][1]  # 128

# Useful list methods
features.append(1.5)      # Add element
features.extend([0.7, 1.8])  # Add multiple elements
features.insert(0, 3.2)   # Insert at position
removed = features.pop()   # Remove and return last
features.remove(0.8)      # Remove first occurrence
features.sort()           # Sort in place
features.reverse()        # Reverse in place

# Slicing (crucial for data preprocessing)
train_data = features[:80]     # First 80% for training
test_data = features[80:]      # Last 20% for testing
every_second = features[::2]   # Every second element`}</code></pre>
          </div>
        </section>

        <section className="ds-section">
          <h2>🗂️ Dictionaries - Key-Value Mapping</h2>
          <p>Dictionaries are perfect for storing model parameters, hyperparameters, and structured data.</p>
          
          <div className="code-example">
            <h4>Dictionary Applications in ML</h4>
            <pre><code>{`# Model configuration
model_config = {
    'algorithm': 'Random Forest',
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'learning_rate': 0.01
}

# Dataset information
dataset_info = {
    'name': 'Iris Dataset',
    'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target': 'species',
    'samples': 150,
    'classes': ['setosa', 'versicolor', 'virginica']
}

# Model performance metrics
metrics = {
    'accuracy': 0.95,
    'precision': 0.92,
    'recall': 0.94,
    'f1_score': 0.93
}

# Dictionary methods
print(model_config.keys())      # All keys
print(model_config.values())    # All values
print(model_config.items())     # Key-value pairs

# Safe access with get()
batch_size = model_config.get('batch_size', 32)  # Default to 32

# Dictionary comprehensions
squared_metrics = {k: v**2 for k, v in metrics.items()}

# Nested dictionaries for complex configurations
deep_config = {
    'model': {
        'type': 'neural_network',
        'layers': [
            {'type': 'dense', 'units': 64, 'activation': 'relu'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'dense', 'units': 32, 'activation': 'relu'},
            {'type': 'dense', 'units': 3, 'activation': 'softmax'}
        ]
    },
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'optimizer': 'adam'
    }
}`}</code></pre>
          </div>
        </section>

        <section className="ds-section">
          <h2>🎯 Sets - Unique Collections</h2>
          <p>Sets store unique elements and provide fast membership testing. Great for feature selection and data cleaning.</p>
          
          <div className="code-example">
            <h4>Sets in Data Processing</h4>
            <pre><code>{`# Remove duplicates from data
raw_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
unique_labels = set(raw_labels)
print(unique_labels)  # {0, 1}

# Find unique feature names
features_model1 = {'age', 'income', 'education', 'experience'}
features_model2 = {'age', 'income', 'location', 'skills'}

# Set operations
common_features = features_model1 & features_model2  # Intersection
all_features = features_model1 | features_model2     # Union
unique_to_model1 = features_model1 - features_model2 # Difference
symmetric_diff = features_model1 ^ features_model2   # Symmetric difference

print(f"Common features: {common_features}")
print(f"All features: {all_features}")

# Fast membership testing
if 'age' in features_model1:  # O(1) operation
    print("Age is a feature")

# Convert back to list if needed
feature_list = list(all_features)`}</code></pre>
          </div>
        </section>

        <section className="ds-section">
          <h2>📦 Tuples - Immutable Sequences</h2>
          <p>Tuples are immutable and perfect for coordinates, configurations that shouldn't change, and returning multiple values.</p>
          
          <div className="code-example">
            <h4>Tuples in ML Applications</h4>
            <pre><code>{`# Image dimensions (width, height, channels)
image_shape = (224, 224, 3)
width, height, channels = image_shape  # Tuple unpacking

# Model architecture definition
layer_config = (
    ('conv2d', 32, (3, 3)),
    ('relu',),
    ('maxpool', (2, 2)),
    ('conv2d', 64, (3, 3)),
    ('relu',),
    ('flatten',),
    ('dense', 128),
    ('dropout', 0.5),
    ('dense', 10)
)

# Coordinate pairs for plotting
data_points = [(1, 2), (3, 4), (5, 6), (7, 8)]

# Function returning multiple values
def train_test_split_info(data_size, test_ratio=0.2):
    test_size = int(data_size * test_ratio)
    train_size = data_size - test_size
    return train_size, test_size, test_ratio

# Unpack the returned tuple
train_samples, test_samples, ratio = train_test_split_info(1000)
print(f"Train: {train_samples}, Test: {test_samples}, Ratio: {ratio}")

# Named tuples for better readability
from collections import namedtuple

ModelResult = namedtuple('ModelResult', ['accuracy', 'loss', 'training_time'])
result = ModelResult(0.95, 0.05, 120.5)

print(f"Model achieved {result.accuracy} accuracy in {result.training_time}s")`}</code></pre>
          </div>
        </section>

        <section className="algorithms-section">
          <h2>🔍 Essential Algorithms for ML</h2>
          
          <h3>Searching Algorithms</h3>
          <div className="code-example">
            <pre><code>{`# Binary Search - O(log n)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Usage with sorted model accuracies
accuracies = [0.72, 0.81, 0.85, 0.89, 0.92, 0.95]
index = binary_search(accuracies, 0.89)
print(f"Found 0.89 at index {index}")`}</code></pre>
          </div>

          <h3>Sorting Algorithms</h3>
          <div className="code-example">
            <pre><code>{`# Quick Sort - Average O(n log n)
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Sort model scores
scores = [0.85, 0.92, 0.78, 0.95, 0.81]
sorted_scores = quicksort(scores)
print(f"Sorted scores: {sorted_scores}")

# Custom sorting with key function
models = [
    ('RandomForest', 0.85),
    ('SVM', 0.92), 
    ('NeuralNet', 0.78),
    ('XGBoost', 0.95)
]

# Sort by accuracy (second element)
models_by_accuracy = sorted(models, key=lambda x: x[1], reverse=True)
print(f"Best model: {models_by_accuracy[0]}")`}</code></pre>
          </div>
        </section>

        <section className="memory-section">
          <h2>💾 Memory Management Tips</h2>
          <div className="tips-grid">
            <div className="tip-card">
              <h4>🚀 Performance Tips</h4>
              <ul>
                <li>Use list comprehensions instead of loops when possible</li>
                <li>Choose the right data structure for the task</li>
                <li>Use generators for large datasets to save memory</li>
                <li>Prefer dictionaries for fast lookups</li>
              </ul>
            </div>
            
            <div className="tip-card">
              <h4>🧠 Memory Optimization</h4>
              <ul>
                <li>Use <code>__slots__</code> in classes to reduce memory</li>
                <li>Delete large objects when done: <code>del large_data</code></li>
                <li>Use numpy arrays instead of lists for numerical data</li>
                <li>Process data in chunks for very large datasets</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default DataStructures