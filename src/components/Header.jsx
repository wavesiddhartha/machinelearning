import { Link } from 'react-router-dom'
import { useState } from 'react'

function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [activeDropdown, setActiveDropdown] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')

  const menuStructure = [
    {
      title: "Python Basics",
      items: [
        { path: "/python-introduction", name: "Introduction to Python", icon: "🐍" },
        { path: "/python-syntax", name: "Python Syntax & Variables", icon: "📝" },
        { path: "/python-data-types", name: "Data Types & Operations", icon: "🔢" },
        { path: "/python-control-flow", name: "Control Flow (if, loops)", icon: "🔄" },
        { path: "/python-functions", name: "Functions & Modules", icon: "⚙️" },
        { path: "/python-error-handling", name: "Error Handling", icon: "⚠️" },
        { path: "/python-file-io", name: "File I/O Operations", icon: "📁" },
        { path: "/python-debugging", name: "Debugging Techniques", icon: "🐛" }
      ]
    },
    {
      title: "Data Structures",
      items: [
        { path: "/python-lists", name: "Lists & List Comprehensions", icon: "📋" },
        { path: "/python-tuples", name: "Tuples & Named Tuples", icon: "📦" },
        { path: "/python-dictionaries", name: "Dictionaries & Sets", icon: "🗃️" },
        { path: "/python-strings", name: "String Manipulation", icon: "🔤" },
        { path: "/python-collections", name: "Collections Module", icon: "📚" },
        { path: "/python-iterators", name: "Iterators & Generators", icon: "🔁" },
        { path: "/python-comprehensions", name: "Advanced Comprehensions", icon: "🧠" },
        { path: "/python-memory", name: "Memory Management", icon: "💾" }
      ]
    },
    {
      title: "OOP in Python",
      items: [
        { path: "/python-classes", name: "Classes & Objects", icon: "🏗️" },
        { path: "/python-inheritance", name: "Inheritance & Polymorphism", icon: "👪" },
        { path: "/python-encapsulation", name: "Encapsulation & Abstraction", icon: "🔒" },
        { path: "/python-magic-methods", name: "Magic Methods (Dunder)", icon: "✨" },
        { path: "/python-properties", name: "Properties & Descriptors", icon: "🎛️" },
        { path: "/python-metaclasses", name: "Metaclasses", icon: "🎭" },
        { path: "/python-design-patterns", name: "Design Patterns", icon: "🎨" },
        { path: "/python-solid-principles", name: "SOLID Principles", icon: "💎" }
      ]
    },
    {
      title: "Advanced Python",
      items: [
        { path: "/python-decorators", name: "Decorators & Closures", icon: "🎪" },
        { path: "/python-context-managers", name: "Context Managers", icon: "🚪" },
        { path: "/python-threading", name: "Threading & Multiprocessing", icon: "🧵" },
        { path: "/python-async", name: "Async Programming", icon: "⚡" },
        { path: "/python-generators", name: "Advanced Generators", icon: "🌀" },
        { path: "/python-testing", name: "Testing (unittest, pytest)", icon: "🧪" },
        { path: "/python-packaging", name: "Packaging & Distribution", icon: "📦" },
        { path: "/python-performance", name: "Performance Optimization", icon: "🚀" }
      ]
    },
    {
      title: "Data Science & Analytics",
      items: [
        { path: "/data-science-complete", name: "Complete Data Science Guide", icon: "📊" },
        { path: "/numpy-complete", name: "NumPy Mastery", icon: "🔢" },
        { path: "/pandas-complete", name: "Pandas Data Analysis", icon: "🐼" },
        { path: "/matplotlib-complete", name: "Data Visualization", icon: "📈" },
        { path: "/data-cleaning", name: "Data Cleaning & Preprocessing", icon: "🧹" },
        { path: "/statistical-analysis", name: "Statistical Analysis", icon: "📐" },
        { path: "/data-workflows", name: "Data Science Workflows", icon: "🔄" },
        { path: "/business-intelligence", name: "Business Intelligence", icon: "💼" }
      ]
    },
    {
      title: "Web Development",
      items: [
        { path: "/web-development-complete", name: "Full-Stack Python Web Dev", icon: "🌐" },
        { path: "/django-complete", name: "Django Framework", icon: "🎸" },
        { path: "/flask-complete", name: "Flask Microframework", icon: "🌶️" },
        { path: "/fastapi-complete", name: "FastAPI Modern APIs", icon: "⚡" },
        { path: "/web-scraping", name: "Web Scraping", icon: "🕷️" },
        { path: "/rest-apis", name: "REST API Development", icon: "🔌" },
        { path: "/database-integration", name: "Database Integration", icon: "🗄️" },
        { path: "/web-deployment", name: "Deployment & DevOps", icon: "🚀" }
      ]
    },
    {
      title: "Machine Learning",
      items: [
        { path: "/machine-learning-complete", name: "ML Fundamentals", icon: "🤖" },
        { path: "/ml-mathematics", name: "ML Mathematics", icon: "📊" },
        { path: "/ml-data-preprocessing", name: "Data Preprocessing", icon: "🧹" },
        { path: "/ml-supervised-learning", name: "Supervised Learning", icon: "👨‍🏫" },
        { path: "/ml-unsupervised-learning", name: "Unsupervised Learning", icon: "🔍" },
        { path: "/ml-neural-networks", name: "Neural Networks", icon: "🧠" },
        { path: "/ml-deep-learning", name: "Deep Learning", icon: "🚀" },
        { path: "/ml-computer-vision", name: "Computer Vision", icon: "👁️" }
      ]
    },
    {
      title: "Advanced AI & Deep Learning",
      items: [
        { path: "/computer-vision-complete", name: "Complete Computer Vision", icon: "👁️" },
        { path: "/cv-fundamentals", name: "CV Fundamentals & OpenCV", icon: "🔍" },
        { path: "/object-detection", name: "Object Detection & YOLO", icon: "🎯" },
        { path: "/image-processing", name: "Image Processing & Analysis", icon: "🖼️" },
        { path: "/nlp-complete", name: "Complete NLP Guide", icon: "📝" },
        { path: "/text-processing", name: "Text Processing & Analysis", icon: "📊" },
        { path: "/sentiment-analysis", name: "Sentiment Analysis", icon: "😊" },
        { path: "/transformers-nlp", name: "Transformers & BERT", icon: "🤖" }
      ]
    },
    {
      title: "Python Projects",
      items: [
        { path: "/python-beginner-projects", name: "Beginner Projects", icon: "🌱" },
        { path: "/python-web-scraping", name: "Web Scraping Projects", icon: "🕷️" },
        { path: "/python-api-projects", name: "API Development", icon: "🔌" },
        { path: "/python-gui-projects", name: "GUI Applications", icon: "🖥️" },
        { path: "/python-data-projects", name: "Data Analysis Projects", icon: "📈" },
        { path: "/python-ml-projects", name: "Machine Learning Projects", icon: "🤖" },
        { path: "/python-automation", name: "Automation Scripts", icon: "⚙️" },
        { path: "/python-portfolio", name: "Portfolio Projects", icon: "💼" }
      ]
    }
  ]

  return (
    <header className="professional-header">
      <div className="header-top">
        <div className="container">
          <Link to="/" className="professional-logo">
            <img src="/logo.svg" alt="CodeMaster" className="logo-img" onError={(e) => {e.target.style.display='none'}} />
            <span className="logo-text">
              <strong>PythonMaster</strong>
              <small>Complete Python Learning Platform</small>
            </span>
          </Link>
          
          <div className="search-container">
            <input 
              type="text"
              placeholder="Search tutorials, problems, articles..."
              className="search-input"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button className="search-btn">🔍</button>
          </div>
          
          <div className="header-actions">
            <Link to="/login" className="auth-btn">Login</Link>
            <Link to="/signup" className="auth-btn signup">Sign Up</Link>
            <button className="theme-toggle">🌙</button>
          </div>
          
          <button 
            className="mobile-menu-toggle"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            ☰
          </button>
        </div>
      </div>
      
      <div className="header-nav">
        <div className="container">
          <nav className={`professional-nav ${isMenuOpen ? 'nav-open' : ''}`}>
            <Link to="/" className="nav-item active">Home</Link>
            
            {menuStructure.map((section, idx) => (
              <div 
                key={idx}
                className="nav-dropdown"
                onMouseEnter={() => setActiveDropdown(idx)}
                onMouseLeave={() => setActiveDropdown(null)}
              >
                <button className="nav-item dropdown-trigger">
                  {section.title}
                  <span className="dropdown-icon">▼</span>
                </button>
                {activeDropdown === idx && (
                  <div className="mega-dropdown">
                    <div className="dropdown-content">
                      <div className="dropdown-grid">
                        {section.items.map((item, itemIdx) => (
                          <Link 
                            key={itemIdx}
                            to={item.path} 
                            className="dropdown-link"
                            onClick={() => setActiveDropdown(null)}
                          >
                            <span className="link-icon">{item.icon}</span>
                            <div className="link-content">
                              <span className="link-title">{item.name}</span>
                              <span className="link-desc">Master {item.name.toLowerCase()}</span>
                            </div>
                          </Link>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
            
            <Link to="/contribute" className="nav-item contribute">Contribute</Link>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header