import { useState } from 'react'

function WebDevelopmentComplete() {
  const [activeSection, setActiveSection] = useState(0)
  const [expandedCode, setExpandedCode] = useState({})

  const toggleCode = (sectionId, codeId) => {
    const key = `${sectionId}-${codeId}`
    setExpandedCode(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  const sections = [
    {
      id: 'django-intro',
      title: 'Django Framework Fundamentals',
      icon: '🎸',
      description: 'Master Django - The web framework for perfectionists with deadlines',
      content: `
        Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design.
        It follows the Model-View-Template (MVT) architectural pattern and includes many built-in features.
      `,
      keyTopics: [
        'Django Architecture (MVT Pattern)',
        'Models and Database ORM',
        'Views and URL Routing',
        'Templates and Template Language',
        'Forms and Form Validation',
        'Admin Interface',
        'Authentication and Authorization',
        'Middleware and Security'
      ],
      codeExamples: [
        {
          title: 'Django Project Structure',
          description: 'Setting up a Django project and understanding the structure',
          code: `# Create a new Django project
django-admin startproject myproject
cd myproject

# Create a new app
python manage.py startapp myapp

# Project structure:
myproject/
├── manage.py
├── myproject/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── myapp/
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── models.py
    ├── tests.py
    └── views.py`
        },
        {
          title: 'Django Models - Database ORM',
          description: 'Creating models and interacting with the database',
          code: `# models.py
from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "categories"
    
    def __str__(self):
        return self.name

class Post(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
    ]
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    content = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title

# Database operations
# Create
post = Post.objects.create(
    title="My First Post",
    slug="my-first-post",
    author=user,
    category=category,
    content="This is my first blog post."
)

# Read
all_posts = Post.objects.all()
published_posts = Post.objects.filter(status='published')
post = Post.objects.get(slug='my-first-post')

# Update
post.title = "Updated Title"
post.save()

# Delete
post.delete()`
        },
        {
          title: 'Django Views and URLs',
          description: 'Creating views and URL patterns',
          code: `# views.py
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.generic import ListView, DetailView
from .models import Post, Category

# Function-based views
def post_list(request):
    posts = Post.objects.filter(status='published')
    categories = Category.objects.all()
    context = {
        'posts': posts,
        'categories': categories
    }
    return render(request, 'blog/post_list.html', context)

def post_detail(request, slug):
    post = get_object_or_404(Post, slug=slug, status='published')
    return render(request, 'blog/post_detail.html', {'post': post})

def api_posts(request):
    posts = list(Post.objects.filter(status='published').values(
        'title', 'slug', 'created_at'
    ))
    return JsonResponse({'posts': posts})

# Class-based views
class PostListView(ListView):
    model = Post
    template_name = 'blog/post_list.html'
    context_object_name = 'posts'
    paginate_by = 10
    
    def get_queryset(self):
        return Post.objects.filter(status='published')

class PostDetailView(DetailView):
    model = Post
    template_name = 'blog/post_detail.html'
    slug_field = 'slug'
    context_object_name = 'post'

# urls.py (app level)
from django.urls import path
from . import views

app_name = 'blog'
urlpatterns = [
    path('', views.PostListView.as_view(), name='post_list'),
    path('post/<slug:slug>/', views.PostDetailView.as_view(), name='post_detail'),
    path('api/posts/', views.api_posts, name='api_posts'),
    path('category/<int:category_id>/', views.category_posts, name='category_posts'),
]

# urls.py (project level)
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls')),
    path('', include('blog.urls')),
]`
        }
      ]
    },
    {
      id: 'flask-intro',
      title: 'Flask Microframework',
      icon: '🌶️',
      description: 'Build lightweight and flexible web applications with Flask',
      content: `
        Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy,
        with the ability to scale up to complex applications.
      `,
      keyTopics: [
        'Flask Application Structure',
        'Routing and URL Building',
        'Templates with Jinja2',
        'Request Handling',
        'Session Management',
        'Blueprints for Organization',
        'Database Integration with SQLAlchemy',
        'RESTful APIs'
      ],
      codeExamples: [
        {
          title: 'Flask Application Basics',
          description: 'Creating a basic Flask application',
          code: `# app.py
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Basic route
@app.route('/')
def home():
    return render_template('index.html', 
                         title='Welcome to Flask',
                         current_time=datetime.now())

# Route with parameters
@app.route('/user/<username>')
def user_profile(username):
    return f"<h1>Welcome, {username}!</h1>"

# Route with multiple HTTP methods
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Process form data (save to database, send email, etc.)
        
        return render_template('contact.html', 
                             success=True, 
                             message="Thank you for your message!")
    
    return render_template('contact.html')

# JSON API endpoint
@app.route('/api/users')
def api_users():
    users = [
        {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
        {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com'}
    ]
    return jsonify(users)

# Error handling
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)`
        },
        {
          title: 'Flask with SQLAlchemy',
          description: 'Database integration using SQLAlchemy ORM',
          code: `# models.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Post {self.title}>'

# app.py
from flask import Flask, render_template, request, redirect, url_for
from models import db, User, Post

@app.route('/posts')
def posts():
    posts = Post.query.order_by(Post.date_posted.desc()).all()
    return render_template('posts.html', posts=posts)

@app.route('/post/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        
        # Assume user is logged in (in real app, get from session)
        user = User.query.first()
        
        post = Post(title=title, content=content, author=user)
        db.session.add(post)
        db.session.commit()
        
        return redirect(url_for('posts'))
    
    return render_template('new_post.html')

# Database initialization
@app.before_first_request
def create_tables():
    db.create_all()
    
    # Create sample data if doesn't exist
    if not User.query.first():
        user = User(username='admin', email='admin@example.com')
        db.session.add(user)
        db.session.commit()`
        },
        {
          title: 'Flask Blueprints - Application Structure',
          description: 'Organizing Flask applications with blueprints',
          code: `# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    
    db.init_app(app)
    
    # Register blueprints
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

# app/main/__init__.py
from flask import Blueprint

bp = Blueprint('main', __name__)

from app.main import routes

# app/main/routes.py
from flask import render_template, request
from app.main import bp
from app.models import Post

@bp.route('/')
def index():
    posts = Post.query.order_by(Post.timestamp.desc()).limit(5).all()
    return render_template('index.html', posts=posts)

@bp.route('/about')
def about():
    return render_template('about.html')

# app/auth/__init__.py
from flask import Blueprint

bp = Blueprint('auth', __name__)

from app.auth import routes

# app/auth/routes.py
from flask import render_template, redirect, url_for, request, session
from app.auth import bp
from app.models import User

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            return redirect(url_for('main.index'))
    
    return render_template('auth/login.html')

@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('main.index'))

# app/api/__init__.py
from flask import Blueprint

bp = Blueprint('api', __name__)

from app.api import routes

# app/api/routes.py
from flask import jsonify, request
from app.api import bp
from app.models import Post, User

@bp.route('/posts')
def posts():
    posts = Post.query.all()
    return jsonify([{
        'id': p.id,
        'title': p.title,
        'content': p.content,
        'author': p.author.username,
        'timestamp': p.timestamp.isoformat()
    } for p in posts])

@bp.route('/posts/<int:id>')
def post_detail(id):
    post = Post.query.get_or_404(id)
    return jsonify({
        'id': post.id,
        'title': post.title,
        'content': post.content,
        'author': post.author.username,
        'timestamp': post.timestamp.isoformat()
    })`
        }
      ]
    },
    {
      id: 'fastapi-intro',
      title: 'FastAPI - Modern Python APIs',
      icon: '⚡',
      description: 'Build high-performance APIs with automatic documentation',
      content: `
        FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ 
        based on standard Python type hints. It provides automatic API documentation and validation.
      `,
      keyTopics: [
        'FastAPI Fundamentals',
        'Automatic Documentation',
        'Type Hints and Validation',
        'Dependency Injection',
        'Database Integration',
        'Authentication and Security',
        'Background Tasks',
        'Testing APIs'
      ],
      codeExamples: [
        {
          title: 'FastAPI Basics',
          description: 'Creating a basic FastAPI application with automatic documentation',
          code: `# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="My API",
    description="A sample API built with FastAPI",
    version="1.0.0"
)

# Pydantic models for request/response validation
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class PostBase(BaseModel):
    title: str
    content: str

class PostCreate(PostBase):
    pass

class Post(PostBase):
    id: int
    author_id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

# In-memory data (use database in production)
users_db = []
posts_db = []

# Basic routes
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/users", response_model=List[User])
async def get_users(skip: int = 0, limit: int = 100):
    return users_db[skip : skip + limit]

@app.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # In real app, hash the password
    user_dict = user.dict()
    user_dict["id"] = len(users_db) + 1
    user_dict["is_active"] = True
    user_dict["created_at"] = datetime.now()
    users_db.append(user_dict)
    return user_dict

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    for user in users_db:
        if user["id"] == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

# Query parameters with validation
@app.get("/posts")
async def get_posts(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    author_id: Optional[int] = None
):
    filtered_posts = posts_db
    
    if search:
        filtered_posts = [p for p in filtered_posts if search.lower() in p["title"].lower()]
    
    if author_id:
        filtered_posts = [p for p in filtered_posts if p["author_id"] == author_id]
    
    return filtered_posts[skip : skip + limit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`
        },
        {
          title: 'FastAPI with Database (SQLAlchemy)',
          description: 'Integrating FastAPI with SQLAlchemy ORM',
          code: `# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    posts = relationship("PostDB", back_populates="author")

class PostDB(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
    author_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    author = relationship("UserDB", back_populates="posts")

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# main.py (updated with database)
from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.orm import Session
from database import get_db, UserDB, PostDB
from passlib.context import CryptContext
import crud, schemas

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = hash_password(user.password)
    db_user = UserDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(UserDB).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/users/{user_id}/posts/", response_model=schemas.Post)
def create_post_for_user(
    user_id: int, 
    post: schemas.PostCreate, 
    db: Session = Depends(get_db)
):
    # Verify user exists
    db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_post = PostDB(**post.dict(), author_id=user_id)
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

@app.get("/posts/", response_model=List[schemas.Post])
def read_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    posts = db.query(PostDB).offset(skip).limit(limit).all()
    return posts`
        },
        {
          title: 'FastAPI Authentication & Security',
          description: 'Implementing JWT authentication and security',
          code: `# auth.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import get_db, UserDB

# Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token scheme
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserDB = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# main.py (with authentication)
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from auth import authenticate_user, create_access_token, get_current_active_user

class Token(BaseModel):
    access_token: str
    token_type: str

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/token", response_model=Token)
async def login_for_access_token(user_login: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: UserDB = Depends(get_current_active_user)):
    return current_user

@app.post("/users/me/posts", response_model=schemas.Post)
async def create_post_for_current_user(
    post: schemas.PostCreate,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    db_post = PostDB(**post.dict(), author_id=current_user.id)
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

# Protected endpoint
@app.get("/protected")
async def protected_route(current_user: UserDB = Depends(get_current_active_user)):
    return {"message": f"Hello {current_user.username}, this is a protected route!"}`
        }
      ]
    },
    {
      id: 'web-scraping',
      title: 'Web Scraping with Python',
      icon: '🕷️',
      description: 'Extract data from websites using Beautiful Soup, Scrapy, and Selenium',
      content: `
        Web scraping is the process of extracting data from websites. Python provides excellent libraries 
        for web scraping, from simple HTML parsing to complex JavaScript-heavy sites.
      `,
      keyTopics: [
        'Beautiful Soup for HTML Parsing',
        'Requests for HTTP Operations',
        'Scrapy Framework',
        'Selenium for JavaScript Sites',
        'Data Extraction Patterns',
        'Handling Forms and Sessions',
        'Rate Limiting and Ethics',
        'Data Storage and Processing'
      ],
      codeExamples: [
        {
          title: 'Beautiful Soup Basics',
          description: 'Web scraping with requests and Beautiful Soup',
          code: `import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin, urlparse

# Basic web scraping example
def scrape_quotes():
    url = "http://quotes.toscrape.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    quotes = []
    for quote in soup.find_all('div', class_='quote'):
        text = quote.find('span', class_='text').get_text()
        author = quote.find('small', class_='author').get_text()
        tags = [tag.get_text() for tag in quote.find_all('a', class_='tag')]
        
        quotes.append({
            'text': text,
            'author': author,
            'tags': ', '.join(tags)
        })
    
    return quotes

# Advanced scraping with session handling
class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_with_pagination(self, base_url):
        all_data = []
        page = 1
        
        while True:
            url = f"{base_url}/page/{page}/"
            print(f"Scraping page {page}")
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if we've reached the last page
            quotes = soup.find_all('div', class_='quote')
            if not quotes:
                break
            
            for quote in quotes:
                text = quote.find('span', class_='text').get_text()
                author = quote.find('small', class_='author').get_text()
                tags = [tag.get_text() for tag in quote.find_all('a', class_='tag')]
                
                all_data.append({
                    'text': text,
                    'author': author,
                    'tags': ', '.join(tags),
                    'page': page
                })
            
            # Be respectful - add delay between requests
            time.sleep(1)
            page += 1
        
        return all_data
    
    def scrape_news_articles(self, news_url):
        articles = []
        response = self.session.get(news_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links (this will vary by site)
        article_links = soup.find_all('a', href=True)
        
        for link in article_links[:5]:  # Limit to first 5 for demo
            article_url = urljoin(news_url, link['href'])
            
            try:
                article_response = self.session.get(article_url)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                # Extract article data (adjust selectors for specific sites)
                title = article_soup.find('h1')
                content = article_soup.find('div', class_='content') or article_soup.find('article')
                
                if title and content:
                    articles.append({
                        'url': article_url,
                        'title': title.get_text().strip(),
                        'content': content.get_text().strip()[:500] + '...',
                    })
                
                time.sleep(2)  # Be respectful
            except Exception as e:
                print(f"Error scraping {article_url}: {e}")
                continue
        
        return articles

# Usage examples
if __name__ == "__main__":
    # Basic scraping
    quotes = scrape_quotes()
    print(f"Scraped {len(quotes)} quotes")
    
    # Advanced scraping
    scraper = WebScraper()
    all_quotes = scraper.scrape_with_pagination("http://quotes.toscrape.com")
    
    # Save to CSV
    df = pd.DataFrame(all_quotes)
    df.to_csv('quotes.csv', index=False)
    print(f"Saved {len(all_quotes)} quotes to CSV")`
        },
        {
          title: 'Selenium for Dynamic Content',
          description: 'Scraping JavaScript-heavy websites with Selenium',
          code: `from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time

class SeleniumScraper:
    def __init__(self, headless=True):
        # Set up Chrome options
        self.options = Options()
        if headless:
            self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        
        # Initialize driver
        self.driver = None
        self.wait = None
    
    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def close_driver(self):
        if self.driver:
            self.driver.quit()
    
    def scrape_spa_website(self, url):
        """Scrape Single Page Application with dynamic content"""
        self.start_driver()
        
        try:
            self.driver.get(url)
            
            # Wait for content to load
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "quote"))
            )
            
            quotes = []
            
            # Simulate scrolling to load more content (for infinite scroll)
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while True:
                # Scroll down to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait for new content to load
                time.sleep(2)
                
                # Calculate new scroll height and compare to last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            # Extract all quotes after scrolling
            quote_elements = self.driver.find_elements(By.CLASS_NAME, "quote")
            
            for quote_elem in quote_elements:
                try:
                    text = quote_elem.find_element(By.CLASS_NAME, "text").text
                    author = quote_elem.find_element(By.CLASS_NAME, "author").text
                    tags = [tag.text for tag in quote_elem.find_elements(By.CLASS_NAME, "tag")]
                    
                    quotes.append({
                        'text': text,
                        'author': author,
                        'tags': ', '.join(tags)
                    })
                except NoSuchElementException as e:
                    print(f"Error extracting quote: {e}")
                    continue
            
            return quotes
            
        finally:
            self.close_driver()
    
    def scrape_with_form_interaction(self, url):
        """Scrape site requiring form interaction"""
        self.start_driver()
        
        try:
            self.driver.get(url)
            
            # Fill out search form
            search_input = self.wait.until(
                EC.presence_of_element_located((By.NAME, "search"))
            )
            search_input.clear()
            search_input.send_keys("python")
            
            # Submit form
            submit_button = self.driver.find_element(By.XPATH, "//input[@type='submit']")
            submit_button.click()
            
            # Wait for results to load
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "search-result"))
            )
            
            # Extract results
            results = []
            result_elements = self.driver.find_elements(By.CLASS_NAME, "search-result")
            
            for result in result_elements:
                title = result.find_element(By.TAG_NAME, "h3").text
                description = result.find_element(By.CLASS_NAME, "description").text
                link = result.find_element(By.TAG_NAME, "a").get_attribute("href")
                
                results.append({
                    'title': title,
                    'description': description,
                    'link': link
                })
            
            return results
            
        finally:
            self.close_driver()
    
    def scrape_table_data(self, url):
        """Scrape complex table data with pagination"""
        self.start_driver()
        
        try:
            self.driver.get(url)
            
            all_data = []
            page = 1
            
            while True:
                print(f"Scraping page {page}")
                
                # Wait for table to load
                table = self.wait.until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
                # Extract table data
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 3:  # Ensure we have enough columns
                        all_data.append({
                            'name': cells[0].text,
                            'position': cells[1].text,
                            'salary': cells[2].text,
                            'page': page
                        })
                
                # Try to find and click next page button
                try:
                    next_button = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Next')]")
                    if 'disabled' in next_button.get_attribute('class'):
                        break
                    next_button.click()
                    page += 1
                    time.sleep(2)  # Wait for page to load
                except NoSuchElementException:
                    break
            
            return all_data
            
        finally:
            self.close_driver()

# Usage example
if __name__ == "__main__":
    scraper = SeleniumScraper(headless=False)  # Set to True to run headless
    
    # Example 1: Scrape SPA website
    quotes = scraper.scrape_spa_website("http://quotes.toscrape.com/js/")
    print(f"Scraped {len(quotes)} quotes from SPA")
    
    # Save to file
    df = pd.DataFrame(quotes)
    df.to_csv('spa_quotes.csv', index=False)
    
    print("Scraping completed!")`
        }
      ]
    },
    {
      id: 'rest-apis',
      title: 'Building RESTful APIs',
      icon: '🔌',
      description: 'Design and implement RESTful APIs following best practices',
      content: `
        REST (Representational State Transfer) is an architectural style for designing networked applications.
        Learn to build robust, scalable APIs with proper HTTP methods, status codes, and design patterns.
      `,
      keyTopics: [
        'REST Architecture Principles',
        'HTTP Methods and Status Codes',
        'API Design Best Practices',
        'Request/Response Validation',
        'Error Handling and Logging',
        'API Authentication (JWT, OAuth)',
        'Rate Limiting and Throttling',
        'API Documentation'
      ],
      codeExamples: [
        {
          title: 'Complete REST API with Flask',
          description: 'Building a full-featured REST API with Flask',
          code: `from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from marshmallow import Schema, fields, ValidationError
from datetime import datetime, timedelta
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///api.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    posts = db.relationship('Post', backref='author', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'posts_count': len(self.posts)
        }

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author.username
        }

# Validation Schemas
class UserRegistrationSchema(Schema):
    username = fields.Str(required=True, validate=lambda x: len(x) >= 3)
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=lambda x: len(x) >= 8)

class UserLoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

class PostSchema(Schema):
    title = fields.Str(required=True, validate=lambda x: len(x) >= 5)
    content = fields.Str(required=True, validate=lambda x: len(x) >= 10)

# Error handlers
@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({
        'error': 'Validation failed',
        'messages': error.messages
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Authentication endpoints
@app.route('/api/register', methods=['POST'])
def register():
    schema = UserRegistrationSchema()
    try:
        data = schema.load(request.get_json())
    except ValidationError as err:
        return jsonify({'error': 'Validation failed', 'messages': err.messages}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 409
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    # Create new user
    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    logger.info(f"New user registered: {user.username}")
    
    return jsonify({
        'message': 'User registered successfully',
        'user': user.to_dict()
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    schema = UserLoginSchema()
    try:
        data = schema.load(request.get_json())
    except ValidationError as err:
        return jsonify({'error': 'Validation failed', 'messages': err.messages}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    access_token = create_access_token(identity=user.id)
    
    logger.info(f"User logged in: {user.username}")
    
    return jsonify({
        'access_token': access_token,
        'user': user.to_dict()
    }), 200

# User endpoints
@app.route('/api/users', methods=['GET'])
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 100)
    
    users = User.query.paginate(page=page, per_page=per_page)
    
    return jsonify({
        'users': [user.to_dict() for user in users.items],
        'pagination': {
            'page': page,
            'pages': users.pages,
            'per_page': per_page,
            'total': users.total
        }
    }), 200

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({'user': user.to_dict()}), 200

@app.route('/api/users/me', methods=['GET'])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.query.get_or_404(user_id)
    return jsonify({'user': user.to_dict()}), 200

# Post endpoints
@app.route('/api/posts', methods=['GET'])
def get_posts():
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 100)
    user_id = request.args.get('user_id', type=int)
    
    query = Post.query
    if user_id:
        query = query.filter_by(user_id=user_id)
    
    posts = query.order_by(Post.created_at.desc()).paginate(page=page, per_page=per_page)
    
    return jsonify({
        'posts': [post.to_dict() for post in posts.items],
        'pagination': {
            'page': page,
            'pages': posts.pages,
            'per_page': per_page,
            'total': posts.total
        }
    }), 200

@app.route('/api/posts', methods=['POST'])
@jwt_required()
def create_post():
    schema = PostSchema()
    try:
        data = schema.load(request.get_json())
    except ValidationError as err:
        return jsonify({'error': 'Validation failed', 'messages': err.messages}), 400
    
    user_id = get_jwt_identity()
    post = Post(
        title=data['title'],
        content=data['content'],
        user_id=user_id
    )
    
    db.session.add(post)
    db.session.commit()
    
    logger.info(f"New post created: {post.id} by user {user_id}")
    
    return jsonify({
        'message': 'Post created successfully',
        'post': post.to_dict()
    }), 201

@app.route('/api/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    post = Post.query.get_or_404(post_id)
    return jsonify({'post': post.to_dict()}), 200

@app.route('/api/posts/<int:post_id>', methods=['PUT'])
@jwt_required()
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    user_id = get_jwt_identity()
    
    # Check if user owns the post
    if post.user_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    schema = PostSchema()
    try:
        data = schema.load(request.get_json())
    except ValidationError as err:
        return jsonify({'error': 'Validation failed', 'messages': err.messages}), 400
    
    post.title = data['title']
    post.content = data['content']
    post.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    logger.info(f"Post updated: {post.id} by user {user_id}")
    
    return jsonify({
        'message': 'Post updated successfully',
        'post': post.to_dict()
    }), 200

@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
@jwt_required()
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    user_id = get_jwt_identity()
    
    # Check if user owns the post
    if post.user_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(post)
    db.session.commit()
    
    logger.info(f"Post deleted: {post_id} by user {user_id}")
    
    return jsonify({'message': 'Post deleted successfully'}), 200

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

# Create tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)`
        }
      ]
    }
  ]

  return (
    <div className="page">
      <div className="content">
        <div className="page-header">
          <h1>🌐 Complete Web Development with Python</h1>
          <p className="page-description">
            Master full-stack web development with Django, Flask, FastAPI, and modern web technologies.
            Build powerful web applications from simple sites to complex APIs.
          </p>
        </div>

        <div className="learning-path">
          <h2>🗺️ Web Development Learning Path</h2>
          <div className="path-steps">
            <div className="path-step">
              <div className="step-number">1</div>
              <h3>Django Framework</h3>
              <p>Learn Django's MVT pattern, ORM, and built-in features for rapid development</p>
            </div>
            <div className="path-step">
              <div className="step-number">2</div>
              <h3>Flask Microframework</h3>
              <p>Master Flask's flexibility for building lightweight and scalable applications</p>
            </div>
            <div className="path-step">
              <div className="step-number">3</div>
              <h3>FastAPI</h3>
              <p>Build modern, high-performance APIs with automatic documentation</p>
            </div>
            <div className="path-step">
              <div className="step-number">4</div>
              <h3>Web Scraping</h3>
              <p>Extract data from websites using Beautiful Soup and Selenium</p>
            </div>
            <div className="path-step">
              <div className="step-number">5</div>
              <h3>RESTful APIs</h3>
              <p>Design and implement robust APIs following REST principles</p>
            </div>
          </div>
        </div>

        <div className="section-tabs">
          {sections.map((section, index) => (
            <button
              key={section.id}
              className={`tab-button ${activeSection === index ? 'active' : ''}`}
              onClick={() => setActiveSection(index)}
            >
              <span className="tab-icon">{section.icon}</span>
              {section.title}
            </button>
          ))}
        </div>

        <div className="section-content">
          {sections.map((section, index) => (
            <div
              key={section.id}
              className={`section ${activeSection === index ? 'active' : ''}`}
            >
              <div className="section-header">
                <h2>
                  <span className="section-icon">{section.icon}</span>
                  {section.title}
                </h2>
                <p className="section-description">{section.description}</p>
              </div>

              <div className="section-overview">
                <p>{section.content}</p>
              </div>

              <div className="key-topics">
                <h3>🎯 Key Topics Covered</h3>
                <div className="topics-grid">
                  {section.keyTopics.map((topic, idx) => (
                    <div key={idx} className="topic-item">
                      <span className="topic-bullet">▶</span>
                      {topic}
                    </div>
                  ))}
                </div>
              </div>

              <div className="code-examples">
                <h3>💻 Code Examples & Implementation</h3>
                {section.codeExamples.map((example, idx) => (
                  <div key={idx} className="code-example">
                    <div className="example-header">
                      <h4>{example.title}</h4>
                      <p>{example.description}</p>
                      <button
                        className="toggle-code"
                        onClick={() => toggleCode(section.id, idx)}
                      >
                        {expandedCode[`${section.id}-${idx}`] ? 'Hide Code' : 'Show Code'}
                      </button>
                    </div>
                    
                    {expandedCode[`${section.id}-${idx}`] && (
                      <div className="code-block">
                        <pre><code>{example.code}</code></pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="practice-exercises">
                <h3>🏋️ Practice Exercises</h3>
                <div className="exercises">
                  <div className="exercise">
                    <h4>Beginner Exercise</h4>
                    <p>Create a simple blog application with user authentication and CRUD operations.</p>
                  </div>
                  <div className="exercise">
                    <h4>Intermediate Exercise</h4>
                    <p>Build a REST API with JWT authentication, rate limiting, and comprehensive error handling.</p>
                  </div>
                  <div className="exercise">
                    <h4>Advanced Exercise</h4>
                    <p>Develop a full-stack application with real-time features using WebSockets and background tasks.</p>
                  </div>
                </div>
              </div>

              <div className="real-world-projects">
                <h3>🚀 Real-World Project Ideas</h3>
                <div className="projects-grid">
                  <div className="project-card">
                    <h4>E-commerce Platform</h4>
                    <p>Build a complete e-commerce site with payment integration, inventory management, and admin dashboard.</p>
                  </div>
                  <div className="project-card">
                    <h4>Social Media API</h4>
                    <p>Create a scalable social media backend with user relationships, posts, comments, and real-time notifications.</p>
                  </div>
                  <div className="project-card">
                    <h4>Task Management System</h4>
                    <p>Develop a collaborative task management platform with teams, projects, and deadline tracking.</p>
                  </div>
                  <div className="project-card">
                    <h4>Content Management System</h4>
                    <p>Build a flexible CMS with custom fields, media management, and multi-language support.</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="next-steps">
          <h2>🎯 Next Steps in Your Web Development Journey</h2>
          <div className="next-steps-grid">
            <div className="next-step">
              <h3>🔧 DevOps & Deployment</h3>
              <p>Learn Docker, CI/CD pipelines, and cloud deployment strategies</p>
            </div>
            <div className="next-step">
              <h3>📊 Database Design</h3>
              <p>Master database optimization, indexing, and advanced SQL techniques</p>
            </div>
            <div className="next-step">
              <h3>🔒 Security Best Practices</h3>
              <p>Implement authentication, authorization, and security hardening</p>
            </div>
            <div className="next-step">
              <h3>⚡ Performance Optimization</h3>
              <p>Learn caching strategies, load balancing, and scalability patterns</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WebDevelopmentComplete