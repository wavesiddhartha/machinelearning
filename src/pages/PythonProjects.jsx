import { useState } from 'react'

function PythonProjects() {
  const [selectedCategory, setSelectedCategory] = useState('beginner')
  const [selectedProject, setSelectedProject] = useState(null)

  const projectCategories = [
    { id: 'beginner', title: 'Beginner Projects', icon: '🌱', color: '#10b981' },
    { id: 'intermediate', title: 'Intermediate Projects', icon: '🚀', color: '#3b82f6' },
    { id: 'advanced', title: 'Advanced Projects', icon: '🔥', color: '#ef4444' },
    { id: 'web-dev', title: 'Web Development', icon: '🌐', color: '#8b5cf6' },
    { id: 'data-science', title: 'Data Science & ML', icon: '📊', color: '#f59e0b' },
    { id: 'automation', title: 'Automation & Scripts', icon: '⚙️', color: '#06b6d4' }
  ]

  const projects = {
    beginner: [
      {
        id: 1,
        title: "Password Generator",
        description: "Create a secure password generator with customizable options",
        difficulty: "Easy",
        timeEstimate: "2-3 hours",
        skills: ["Functions", "Random", "String manipulation", "User input"],
        features: [
          "Generate passwords of custom length",
          "Include/exclude character types (uppercase, lowercase, numbers, symbols)",
          "Copy password to clipboard",
          "Save generated passwords to file"
        ],
        codePreview: `import random
import string

def generate_password(length=12, include_upper=True, include_lower=True, 
                     include_numbers=True, include_symbols=True):
    characters = ""
    
    if include_lower:
        characters += string.ascii_lowercase
    if include_upper:
        characters += string.ascii_uppercase
    if include_numbers:
        characters += string.digits
    if include_symbols:
        characters += "!@#$%^&*"
    
    if not characters:
        raise ValueError("At least one character type must be selected")
    
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

# Usage
password = generate_password(16, include_symbols=False)
print(f"Generated password: {password}")`,
        learningOutcomes: [
          "Working with Python's random and string modules",
          "Function parameters and default values",
          "String manipulation techniques",
          "Input validation and error handling"
        ]
      },
      {
        id: 2,
        title: "Todo List Application",
        description: "Build a command-line todo list manager with file persistence",
        difficulty: "Easy",
        timeEstimate: "3-4 hours",
        skills: ["Lists", "File I/O", "Functions", "Loops", "Exception handling"],
        features: [
          "Add, remove, and mark tasks as complete",
          "Save tasks to file (persistence)",
          "Display tasks with different statuses",
          "Search and filter tasks"
        ],
        codePreview: `import json
from datetime import datetime

class TodoList:
    def __init__(self, filename="todos.json"):
        self.filename = filename
        self.tasks = self.load_tasks()
    
    def add_task(self, description, priority="medium"):
        task = {
            "id": len(self.tasks) + 1,
            "description": description,
            "priority": priority,
            "completed": False,
            "created_at": datetime.now().isoformat()
        }
        self.tasks.append(task)
        self.save_tasks()
        return task["id"]
    
    def complete_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = datetime.now().isoformat()
                self.save_tasks()
                return True
        return False
    
    def save_tasks(self):
        with open(self.filename, 'w') as f:
            json.dump(self.tasks, f, indent=2)
    
    def load_tasks(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []`,
        learningOutcomes: [
          "Object-oriented programming basics",
          "JSON file handling for data persistence",
          "Working with dates and timestamps",
          "Command-line interface design"
        ]
      }
    ],
    intermediate: [
      {
        id: 3,
        title: "Web Scraper with BeautifulSoup",
        description: "Build a web scraper to collect data from websites and save to CSV",
        difficulty: "Medium",
        timeEstimate: "4-6 hours",
        skills: ["Requests", "BeautifulSoup", "CSV", "Error handling", "Data cleaning"],
        features: [
          "Scrape product information from e-commerce sites",
          "Handle different page structures",
          "Export data to CSV format",
          "Implement rate limiting and respectful scraping"
        ],
        codePreview: `import requests
from bs4 import BeautifulSoup
import csv
import time
import random

class WebScraper:
    def __init__(self, base_url, delay_range=(1, 3)):
        self.base_url = base_url
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_product_info(self, product_urls):
        products = []
        
        for url in product_urls:
            try:
                # Respectful delay
                time.sleep(random.uniform(*self.delay_range))
                
                response = self.session.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                product = {
                    'title': self.extract_title(soup),
                    'price': self.extract_price(soup),
                    'rating': self.extract_rating(soup),
                    'url': url
                }
                
                products.append(product)
                print(f"Scraped: {product['title']}")
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        return products
    
    def save_to_csv(self, products, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['title', 'price', 'rating', 'url'])
            writer.writeheader()
            writer.writerows(products)`,
        learningOutcomes: [
          "HTTP requests and response handling",
          "HTML parsing with BeautifulSoup",
          "CSV file operations",
          "Error handling and robust programming",
          "Ethical web scraping practices"
        ]
      },
      {
        id: 4,
        title: "Personal Finance Tracker",
        description: "Create a comprehensive personal finance management application",
        difficulty: "Medium",
        timeEstimate: "6-8 hours",
        skills: ["OOP", "Data visualization", "File handling", "Date manipulation"],
        features: [
          "Track income and expenses by category",
          "Generate spending reports and visualizations",
          "Budget planning and tracking",
          "Export financial data to various formats"
        ],
        codePreview: `import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, date
import json

class FinanceTracker:
    def __init__(self):
        self.transactions = []
        self.categories = {
            'income': ['Salary', 'Freelance', 'Investments', 'Other Income'],
            'expenses': ['Food', 'Transport', 'Entertainment', 'Bills', 'Shopping', 'Healthcare']
        }
    
    def add_transaction(self, amount, category, description, transaction_type='expense'):
        transaction = {
            'id': len(self.transactions) + 1,
            'date': date.today().isoformat(),
            'amount': abs(amount) if transaction_type == 'expense' else amount,
            'category': category,
            'description': description,
            'type': transaction_type
        }
        self.transactions.append(transaction)
        return transaction['id']
    
    def get_monthly_summary(self, year, month):
        monthly_transactions = [
            t for t in self.transactions 
            if datetime.fromisoformat(t['date']).year == year 
            and datetime.fromisoformat(t['date']).month == month
        ]
        
        total_income = sum(t['amount'] for t in monthly_transactions if t['type'] == 'income')
        total_expenses = sum(t['amount'] for t in monthly_transactions if t['type'] == 'expense')
        
        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_savings': total_income - total_expenses,
            'transactions': monthly_transactions
        }
    
    def create_spending_chart(self, year, month):
        summary = self.get_monthly_summary(year, month)
        expenses_by_category = {}
        
        for transaction in summary['transactions']:
            if transaction['type'] == 'expense':
                category = transaction['category']
                expenses_by_category[category] = expenses_by_category.get(category, 0) + transaction['amount']
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(expenses_by_category.values(), labels=expenses_by_category.keys(), autopct='%1.1f%%')
        plt.title(f'Spending by Category - {month}/{year}')
        plt.show()`,
        learningOutcomes: [
          "Data visualization with Matplotlib",
          "Working with Pandas for data analysis",
          "Date and time manipulation",
          "Creating comprehensive class structures",
          "Financial calculation logic"
        ]
      }
    ],
    advanced: [
      {
        id: 5,
        title: "Real-time Chat Application",
        description: "Build a real-time chat application using WebSockets and asyncio",
        difficulty: "Hard",
        timeEstimate: "10-15 hours",
        skills: ["Asyncio", "WebSockets", "Networking", "Concurrency", "Database"],
        features: [
          "Real-time messaging between multiple users",
          "User authentication and sessions",
          "Private messaging and group chats",
          "Message history and persistence"
        ],
        codePreview: `import asyncio
import websockets
import json
import sqlite3
from datetime import datetime
import hashlib

class ChatServer:
    def __init__(self):
        self.clients = {}
        self.rooms = {}
        self.init_database()
    
    def init_database(self):
        self.conn = sqlite3.connect('chat.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                room TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    async def register_client(self, websocket, path):
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def handle_message(self, websocket, data):
        message_type = data.get('type')
        
        if message_type == 'join':
            await self.join_room(websocket, data)
        elif message_type == 'message':
            await self.broadcast_message(websocket, data)
        elif message_type == 'private':
            await self.send_private_message(websocket, data)
    
    async def join_room(self, websocket, data):
        username = data['username']
        room = data['room']
        
        self.clients[websocket] = {'username': username, 'room': room}
        
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(websocket)
        
        # Send join notification
        notification = {
            'type': 'notification',
            'message': f'{username} joined the room',
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_to_room(room, notification, exclude=websocket)
    
    async def broadcast_message(self, websocket, data):
        if websocket not in self.clients:
            return
        
        client_info = self.clients[websocket]
        room = client_info['room']
        username = client_info['username']
        
        # Save to database
        self.conn.execute(
            'INSERT INTO messages (username, room, message) VALUES (?, ?, ?)',
            (username, room, data['message'])
        )
        self.conn.commit()
        
        # Broadcast to room
        message = {
            'type': 'message',
            'username': username,
            'message': data['message'],
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_to_room(room, message)
    
    async def broadcast_to_room(self, room, message, exclude=None):
        if room not in self.rooms:
            return
        
        disconnected = []
        for client in self.rooms[room]:
            if client == exclude:
                continue
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            self.rooms[room].discard(client)
    
    async def start_server(self, host='localhost', port=8765):
        print(f"Chat server starting on {host}:{port}")
        async with websockets.serve(self.register_client, host, port):
            await asyncio.Future()  # Run forever

# Start the server
if __name__ == "__main__":
    server = ChatServer()
    asyncio.run(server.start_server())`,
        learningOutcomes: [
          "Advanced asyncio programming patterns",
          "WebSocket protocol implementation",
          "Concurrent programming with multiple clients",
          "Database integration for persistence",
          "Real-time application architecture"
        ]
      },
      {
        id: 6,
        title: "Machine Learning Model Deployment API",
        description: "Create a REST API to serve machine learning models with Docker deployment",
        difficulty: "Hard",
        timeEstimate: "12-20 hours",
        skills: ["FastAPI", "Machine Learning", "Docker", "Model deployment", "API design"],
        features: [
          "Train and save ML models",
          "REST API for model predictions",
          "Model versioning and management",
          "Containerized deployment with Docker"
        ],
        codePreview: `from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import uvicorn
from typing import List
import os

app = FastAPI(title="ML Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: str = "default"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_dir = "models/"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_model(self, X, y, model_name="default"):
        """Train a new model and save it"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # Store model info
        self.models[model_name] = {
            "model": model,
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "version": "1.0.0"
        }
        
        return {
            "model_name": model_name,
            "train_accuracy": train_score,
            "test_accuracy": test_score
        }
    
    def load_model(self, model_name):
        """Load a saved model"""
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            self.models[model_name] = {
                "model": model,
                "version": "1.0.0"
            }
            return model
        return None
    
    def predict(self, features, model_name="default"):
        """Make prediction using specified model"""
        if model_name not in self.models:
            self.load_model(model_name)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]["model"]
        features_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_array)[0]
        confidence = max(model.predict_proba(features_array)[0])
        
        return prediction, confidence

# Initialize trainer
trainer = ModelTrainer()

@app.post("/train")
async def train_model(file: UploadFile = File(...), model_name: str = "default"):
    """Train a new model with uploaded CSV data"""
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Assume last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        result = trainer.train_model(X, y, model_name)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make prediction using trained model"""
    try:
        prediction, confidence = trainer.predict(request.features, request.model_name)
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            model_version=trainer.models[request.model_name]["version"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "name": name,
                "version": info.get("version", "unknown"),
                "train_accuracy": info.get("train_accuracy"),
                "test_accuracy": info.get("test_accuracy")
            }
            for name, info in trainer.models.items()
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "ML API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)`,
        learningOutcomes: [
          "FastAPI framework for building REST APIs",
          "Machine learning model training and deployment",
          "Model versioning and management strategies",
          "Docker containerization for deployment",
          "Production-ready API design patterns"
        ]
      }
    ],
    'web-dev': [
      {
        id: 7,
        title: "Django Blog Platform",
        description: "Build a full-featured blog platform with Django",
        difficulty: "Medium",
        timeEstimate: "8-12 hours",
        skills: ["Django", "ORM", "Templates", "Authentication", "Admin interface"],
        features: [
          "User registration and authentication",
          "Create, edit, and delete blog posts",
          "Comment system with moderation",
          "Admin interface for content management"
        ]
      },
      {
        id: 8,
        title: "Flask E-commerce API",
        description: "Create a RESTful API for an e-commerce platform using Flask",
        difficulty: "Hard",
        timeEstimate: "10-15 hours",
        skills: ["Flask", "SQLAlchemy", "JWT", "Payment integration", "Testing"],
        features: [
          "Product catalog with categories",
          "Shopping cart and order management",
          "Payment processing integration",
          "User authentication with JWT tokens"
        ]
      }
    ],
    'data-science': [
      {
        id: 9,
        title: "Stock Price Predictor",
        description: "Build a machine learning model to predict stock prices",
        difficulty: "Medium",
        timeEstimate: "6-10 hours",
        skills: ["Pandas", "NumPy", "Scikit-learn", "Data visualization", "Time series"],
        features: [
          "Fetch stock data from APIs",
          "Feature engineering for time series",
          "Multiple ML models comparison",
          "Interactive visualizations"
        ]
      },
      {
        id: 10,
        title: "Customer Sentiment Analysis",
        description: "Analyze customer reviews using NLP techniques",
        difficulty: "Hard",
        timeEstimate: "8-12 hours",
        skills: ["NLTK", "Transformers", "Text preprocessing", "Deep learning", "Visualization"],
        features: [
          "Text preprocessing and cleaning",
          "Sentiment classification models",
          "Topic modeling and analysis",
          "Interactive dashboard for results"
        ]
      }
    ],
    automation: [
      {
        id: 11,
        title: "Email Automation System",
        description: "Create an automated email system for marketing campaigns",
        difficulty: "Medium",
        timeEstimate: "4-6 hours",
        skills: ["Email libraries", "Scheduling", "Template engines", "Database integration"],
        features: [
          "Email template management",
          "Automated sending schedules",
          "Recipient list management",
          "Campaign analytics and tracking"
        ]
      },
      {
        id: 12,
        title: "System Monitoring Dashboard",
        description: "Build a system monitoring tool with alerts and dashboards",
        difficulty: "Hard",
        timeEstimate: "10-15 hours",
        skills: ["System APIs", "Real-time data", "Visualization", "Alert systems"],
        features: [
          "CPU, memory, and disk monitoring",
          "Real-time dashboards",
          "Customizable alerts",
          "Historical data analysis"
        ]
      }
    ]
  }

  const ProjectCard = ({ project, isSelected, onClick }) => (
    <div 
      className={`project-card ${isSelected ? 'selected' : ''}`}
      onClick={() => onClick(project)}
    >
      <div className="project-header">
        <h4 className="project-title">{project.title}</h4>
        <div className="project-meta">
          <span className={`difficulty ${project.difficulty.toLowerCase()}`}>
            {project.difficulty}
          </span>
          <span className="time-estimate">⏱️ {project.timeEstimate}</span>
        </div>
      </div>
      
      <p className="project-description">{project.description}</p>
      
      <div className="project-skills">
        <h5>Skills You'll Learn:</h5>
        <div className="skills-tags">
          {project.skills.map((skill, index) => (
            <span key={index} className="skill-tag">{skill}</span>
          ))}
        </div>
      </div>
      
      <div className="project-features">
        <h5>Key Features:</h5>
        <ul>
          {project.features.slice(0, 3).map((feature, index) => (
            <li key={index}>{feature}</li>
          ))}
          {project.features.length > 3 && <li>...and more</li>}
        </ul>
      </div>
    </div>
  )

  const ProjectDetails = ({ project, onClose }) => (
    <div className="project-details-overlay">
      <div className="project-details">
        <div className="project-details-header">
          <h3>{project.title}</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="project-details-content">
          <div className="project-overview">
            <div className="project-meta-detailed">
              <span className={`difficulty ${project.difficulty.toLowerCase()}`}>
                {project.difficulty}
              </span>
              <span className="time-estimate">⏱️ {project.timeEstimate}</span>
            </div>
            <p className="project-description-detailed">{project.description}</p>
          </div>

          <div className="project-section">
            <h4>🎯 Learning Outcomes</h4>
            <ul className="learning-outcomes">
              {project.learningOutcomes?.map((outcome, index) => (
                <li key={index}>{outcome}</li>
              ))}
            </ul>
          </div>

          <div className="project-section">
            <h4>✨ Features to Implement</h4>
            <ul className="feature-list">
              {project.features.map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </ul>
          </div>

          <div className="project-section">
            <h4>🧰 Technologies & Skills</h4>
            <div className="skills-tags-detailed">
              {project.skills.map((skill, index) => (
                <span key={index} className="skill-tag-detailed">{skill}</span>
              ))}
            </div>
          </div>

          {project.codePreview && (
            <div className="project-section">
              <h4>📝 Code Preview</h4>
              <div className="code-example">
                <pre>{project.codePreview}</pre>
              </div>
            </div>
          )}

          <div className="project-actions">
            <button className="start-project-btn">🚀 Start This Project</button>
            <button className="bookmark-btn">🔖 Bookmark for Later</button>
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <div className="page">
      <div className="python-projects-container">
        <div className="projects-header">
          <h1>🚀 Python Projects</h1>
          <p>Build real-world applications and strengthen your Python skills through hands-on projects</p>
        </div>

        <div className="category-tabs">
          {projectCategories.map(category => (
            <button
              key={category.id}
              className={`category-tab ${selectedCategory === category.id ? 'active' : ''}`}
              onClick={() => setSelectedCategory(category.id)}
              style={{ borderColor: selectedCategory === category.id ? category.color : '#e5e7eb' }}
            >
              <span className="category-icon">{category.icon}</span>
              <span className="category-title">{category.title}</span>
            </button>
          ))}
        </div>

        <div className="projects-section">
          <div className="section-header">
            <h2>
              {projectCategories.find(c => c.id === selectedCategory)?.icon} {' '}
              {projectCategories.find(c => c.id === selectedCategory)?.title}
            </h2>
            <p>
              {selectedCategory === 'beginner' && 'Perfect for those starting their Python journey. Focus on fundamentals and basic programming concepts.'}
              {selectedCategory === 'intermediate' && 'Ready to tackle more complex challenges? These projects combine multiple concepts and libraries.'}
              {selectedCategory === 'advanced' && 'Advanced projects that simulate real-world applications with complex architectures.'}
              {selectedCategory === 'web-dev' && 'Build modern web applications using Python\'s powerful web frameworks.'}
              {selectedCategory === 'data-science' && 'Dive into data analysis, machine learning, and AI with these data-focused projects.'}
              {selectedCategory === 'automation' && 'Automate repetitive tasks and build tools that make life easier.'}
            </p>
          </div>

          <div className="projects-grid">
            {projects[selectedCategory]?.map(project => (
              <ProjectCard
                key={project.id}
                project={project}
                isSelected={selectedProject?.id === project.id}
                onClick={setSelectedProject}
              />
            ))}
          </div>
        </div>

        {selectedProject && (
          <ProjectDetails
            project={selectedProject}
            onClose={() => setSelectedProject(null)}
          />
        )}
      </div>
    </div>
  )
}

export default PythonProjects