import { useState } from 'react'

function PythonBasics() {
  const [activeSection, setActiveSection] = useState('introduction')
  const [runningCode, setRunningCode] = useState('')
  const [codeOutput, setCodeOutput] = useState('')

  const executeCode = (code) => {
    setRunningCode(code)
    // Simulate code execution (in real app, you'd use a Python interpreter)
    try {
      // Simple evaluation for demonstration
      if (code.includes('print(')) {
        const match = code.match(/print\(([^)]+)\)/)
        if (match) {
          setCodeOutput(match[1].replace(/['"]/g, ''))
        }
      } else if (code.includes('=')) {
        setCodeOutput('Variable assigned successfully')
      } else {
        setCodeOutput('Code executed')
      }
    } catch (error) {
      setCodeOutput('Error: ' + error.message)
    }
  }

  const sections = {
    introduction: {
      title: '🚀 Introduction to Python',
      content: (
        <div className="section-content">
          <h3>What is Python?</h3>
          <p>Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, Python emphasizes code readability and allows programmers to express concepts in fewer lines of code.</p>
          
          <h4>Why Python for AI/ML?</h4>
          <ul>
            <li><strong>Easy to Learn:</strong> Simple syntax similar to English</li>
            <li><strong>Extensive Libraries:</strong> NumPy, Pandas, TensorFlow, PyTorch</li>
            <li><strong>Large Community:</strong> Massive support and resources</li>
            <li><strong>Cross-platform:</strong> Works on Windows, Mac, Linux</li>
            <li><strong>Interactive:</strong> Great for experimentation and prototyping</li>
          </ul>

          <div className="code-example">
            <h4>Your First Python Program</h4>
            <pre><code>{`# This is a comment
print("Hello, AI World!")
print("Python is awesome for machine learning!")`}</code></pre>
            <button onClick={() => executeCode('print("Hello, AI World!")')}>Run Code</button>
            {codeOutput && <div className="output">Output: {codeOutput}</div>}
          </div>
        </div>
      )
    },
    variables: {
      title: '📦 Variables and Data Types',
      content: (
        <div className="section-content">
          <h3>Variables in Python</h3>
          <p>Variables are containers for storing data values. Python has dynamic typing, meaning you don't need to declare the type of a variable.</p>
          
          <h4>Basic Data Types</h4>
          <div className="data-types-grid">
            <div className="data-type">
              <h5>Numbers (int, float)</h5>
              <pre><code>{`# Integer
age = 25
count = 1000

# Float  
pi = 3.14159
temperature = 98.6

# Complex numbers
complex_num = 3 + 4j`}</code></pre>
            </div>
            
            <div className="data-type">
              <h5>Strings (str)</h5>
              <pre><code>{`# String creation
name = "Python"
message = 'Machine Learning'
multiline = """This is a
multiline string"""

# String operations
full_name = "John" + " " + "Doe"
repeated = "AI" * 3  # "AIAIAI"`}</code></pre>
            </div>
            
            <div className="data-type">
              <h5>Boolean (bool)</h5>
              <pre><code>{`# Boolean values
is_learning = True
is_difficult = False

# Boolean operations
result = True and False  # False
result = True or False   # True
result = not True        # False`}</code></pre>
            </div>
          </div>

          <h4>Variable Naming Rules</h4>
          <ul>
            <li>Must start with a letter or underscore</li>
            <li>Can contain letters, numbers, and underscores</li>
            <li>Case-sensitive (name ≠ Name)</li>
            <li>Cannot use Python keywords (def, class, if, etc.)</li>
          </ul>

          <div className="code-example">
            <h4>Try It Yourself</h4>
            <pre><code>{`# Create variables for a machine learning dataset
dataset_size = 10000
accuracy = 0.95
model_name = "Neural Network"
is_trained = True

print(f"Model: {model_name}")
print(f"Dataset Size: {dataset_size}")
print(f"Accuracy: {accuracy * 100}%")`}</code></pre>
            <button onClick={() => executeCode('print("Model: Neural Network")')}>Run Code</button>
            {codeOutput && <div className="output">Output: {codeOutput}</div>}
          </div>
        </div>
      )
    },
    operators: {
      title: '⚡ Operators and Expressions',
      content: (
        <div className="section-content">
          <h3>Python Operators</h3>
          
          <div className="operators-grid">
            <div className="operator-category">
              <h4>Arithmetic Operators</h4>
              <pre><code>{`# Basic arithmetic
a = 10
b = 3

addition = a + b      # 13
subtraction = a - b   # 7  
multiplication = a * b # 30
division = a / b      # 3.333...
floor_division = a // b # 3
modulus = a % b       # 1
exponent = a ** b     # 1000`}</code></pre>
            </div>

            <div className="operator-category">
              <h4>Comparison Operators</h4>
              <pre><code>{`# Comparisons return boolean
x = 5
y = 10

equal = x == y        # False
not_equal = x != y    # True
greater = x > y       # False  
less = x < y          # True
greater_equal = x >= y # False
less_equal = x <= y   # True`}</code></pre>
            </div>

            <div className="operator-category">
              <h4>Logical Operators</h4>
              <pre><code>{`# Logical operations
learning_python = True
has_experience = False

# AND operator
ready_for_ai = learning_python and has_experience  # False

# OR operator  
can_start = learning_python or has_experience  # True

# NOT operator
is_beginner = not has_experience  # True`}</code></pre>
            </div>
          </div>

          <h4>Order of Operations (PEMDAS)</h4>
          <p>Python follows mathematical precedence:</p>
          <ol>
            <li><strong>Parentheses</strong> ()</li>
            <li><strong>Exponents</strong> **</li>
            <li><strong>Multiplication/Division</strong> *, /, //, %</li>
            <li><strong>Addition/Subtraction</strong> +, -</li>
          </ol>

          <div className="code-example">
            <h4>Complex Expression Example</h4>
            <pre><code>{`# Calculate accuracy percentage
correct_predictions = 850
total_predictions = 1000

accuracy = (correct_predictions / total_predictions) * 100
print(f"Model Accuracy: {accuracy}%")

# Complex mathematical expression
result = (2 + 3) * 4 ** 2 - 10 / 2
print(f"Mathematical result: {result}")`}</code></pre>
          </div>
        </div>
      )
    },
    control_flow: {
      title: '🔀 Control Flow',
      content: (
        <div className="section-content">
          <h3>Control Flow Statements</h3>
          
          <h4>Conditional Statements (if/elif/else)</h4>
          <pre><code>{`# Basic if statement
accuracy = 0.95

if accuracy > 0.9:
    print("Excellent model!")
elif accuracy > 0.8:
    print("Good model")
elif accuracy > 0.7:
    print("Decent model")
else:
    print("Model needs improvement")

# Nested conditions
dataset_size = 50000
model_complexity = "high"

if dataset_size > 10000:
    if model_complexity == "high":
        print("Use deep learning")
    else:
        print("Use traditional ML")
else:
    print("Collect more data")`}</code></pre>

          <h4>Loops</h4>
          
          <h5>For Loops</h5>
          <pre><code>{`# Iterate over a sequence
algorithms = ["Linear Regression", "Random Forest", "Neural Network"]

for algorithm in algorithms:
    print(f"Training {algorithm}...")

# Range-based loop
for epoch in range(1, 11):  # 1 to 10
    print(f"Epoch {epoch}/10")

# Enumerate for index and value
for i, algorithm in enumerate(algorithms):
    print(f"{i+1}. {algorithm}")`}</code></pre>

          <h5>While Loops</h5>
          <pre><code>{`# While loop for training convergence
loss = 1.0
epoch = 0
max_epochs = 100

while loss > 0.01 and epoch < max_epochs:
    # Simulate training (loss decreases)
    loss *= 0.9
    epoch += 1
    print(f"Epoch {epoch}: Loss = {loss:.4f}")

print(f"Training completed after {epoch} epochs")`}</code></pre>

          <h4>Loop Control</h4>
          <pre><code>{`# break and continue
for i in range(10):
    if i == 3:
        continue  # Skip iteration when i=3
    if i == 7:
        break     # Stop loop when i=7
    print(i)

# Output: 0, 1, 2, 4, 5, 6`}</code></pre>
        </div>
      )
    },
    functions: {
      title: '🔧 Functions',
      content: (
        <div className="section-content">
          <h3>Functions in Python</h3>
          <p>Functions are reusable blocks of code that perform specific tasks. Essential for organizing ML code and avoiding repetition.</p>

          <h4>Basic Function Syntax</h4>
          <pre><code>{`def function_name(parameters):
    """Docstring describing the function"""
    # Function body
    return result

# Example: Calculate accuracy
def calculate_accuracy(correct, total):
    """Calculate accuracy percentage"""
    if total == 0:
        return 0
    accuracy = (correct / total) * 100
    return accuracy

# Call the function
acc = calculate_accuracy(85, 100)
print(f"Accuracy: {acc}%")`}</code></pre>

          <h4>Function Parameters</h4>
          <pre><code>{`# Default parameters
def train_model(algorithm="Random Forest", epochs=100, learning_rate=0.01):
    print(f"Training {algorithm}")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")

# Different ways to call
train_model()  # Uses all defaults
train_model("Neural Network")  # Override algorithm
train_model(epochs=50, algorithm="SVM")  # Keyword arguments

# Variable arguments (*args, **kwargs)
def evaluate_models(*models, **metrics):
    print(f"Evaluating {len(models)} models:")
    for model in models:
        print(f"  - {model}")
    
    print("Metrics to calculate:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value}")

evaluate_models("RF", "SVM", "NN", accuracy=True, precision=True)`}</code></pre>

          <h4>Lambda Functions (Anonymous Functions)</h4>
          <pre><code>{`# Lambda functions for simple operations
square = lambda x: x ** 2
print(square(5))  # 25

# Useful with map, filter, sort
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Filter even numbers
even = list(filter(lambda x: x % 2 == 0, numbers))
print(even)  # [2, 4]`}</code></pre>

          <div className="code-example">
            <h4>ML Function Example</h4>
            <pre><code>{`def sigmoid(x):
    """Sigmoid activation function"""
    import math
    return 1 / (1 + math.exp(-x))

def predict_binary(features, weights, bias):
    """Simple linear classifier"""
    z = sum(f * w for f, w in zip(features, weights)) + bias
    probability = sigmoid(z)
    return 1 if probability > 0.5 else 0

# Example usage
features = [1.2, 0.8, -0.5]
weights = [0.5, -1.2, 0.8]  
bias = 0.1

prediction = predict_binary(features, weights, bias)
print(f"Prediction: {prediction}")`}</code></pre>
          </div>
        </div>
      )
    }
  }

  const sectionKeys = Object.keys(sections)

  return (
    <div className="page">
      <div className="learning-container">
        <aside className="sidebar">
          <h3>📚 Python Basics</h3>
          <nav className="section-nav">
            {sectionKeys.map(key => (
              <button
                key={key}
                className={`section-link ${activeSection === key ? 'active' : ''}`}
                onClick={() => setActiveSection(key)}
              >
                {sections[key].title}
              </button>
            ))}
          </nav>
        </aside>

        <main className="content-area">
          <div className="section-header">
            <h1>{sections[activeSection].title}</h1>
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{width: `${((sectionKeys.indexOf(activeSection) + 1) / sectionKeys.length) * 100}%`}}
              ></div>
            </div>
            <p>{sectionKeys.indexOf(activeSection) + 1} of {sectionKeys.length} sections</p>
          </div>

          {sections[activeSection].content}

          <div className="section-navigation">
            {sectionKeys.indexOf(activeSection) > 0 && (
              <button 
                className="nav-btn prev"
                onClick={() => setActiveSection(sectionKeys[sectionKeys.indexOf(activeSection) - 1])}
              >
                ← Previous
              </button>
            )}
            {sectionKeys.indexOf(activeSection) < sectionKeys.length - 1 && (
              <button 
                className="nav-btn next"
                onClick={() => setActiveSection(sectionKeys[sectionKeys.indexOf(activeSection) + 1])}
              >
                Next →
              </button>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default PythonBasics