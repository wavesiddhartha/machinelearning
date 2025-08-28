import { Link } from 'react-router-dom'
import { useState } from 'react'

function Home() {
  const [activeTab, setActiveTab] = useState('trending')

  const heroStats = [
    { number: "10M+", label: "Students Worldwide", icon: "👥" },
    { number: "500+", label: "In-depth Articles", icon: "📚" },
    { number: "50+", label: "Programming Languages", icon: "💻" },
    { number: "1000+", label: "Practice Problems", icon: "💪" }
  ]

  const trendingArticles = [
    {
      title: "Complete Guide to Dynamic Programming",
      author: "CodeMaster Team",
      readTime: "15 min",
      difficulty: "Advanced",
      tags: ["Algorithms", "DP", "Interview"],
      link: "/dynamic-programming-guide",
      views: "125K"
    },
    {
      title: "Python Data Structures: Lists vs Arrays vs Tuples",
      author: "Sarah Chen",
      readTime: "8 min", 
      difficulty: "Intermediate",
      tags: ["Python", "Data Structures"],
      link: "/python-data-structures",
      views: "89K"
    },
    {
      title: "Machine Learning Algorithms from Scratch",
      author: "Dr. AI Expert",
      readTime: "25 min",
      difficulty: "Advanced", 
      tags: ["ML", "Algorithms", "Math"],
      link: "/ml-algorithms-scratch",
      views: "234K"
    },
    {
      title: "System Design: Designing a Chat Application",
      author: "System Architect",
      readTime: "20 min",
      difficulty: "Advanced",
      tags: ["System Design", "Architecture"],
      link: "/system-design-chat",
      views: "178K"
    }
  ]

  const quickAccessCards = [
    {
      title: "Data Structures & Algorithms",
      desc: "Master DSA for interviews and competitive programming",
      icon: "🏗️",
      color: "#FF6B6B",
      items: ["Arrays", "Linked Lists", "Trees", "Graphs", "Dynamic Programming"],
      link: "/data-structures"
    },
    {
      title: "Programming Languages", 
      desc: "Learn syntax, concepts, and best practices",
      icon: "💻",
      color: "#4ECDC4", 
      items: ["Python", "Java", "C++", "JavaScript", "Go"],
      link: "/languages"
    },
    {
      title: "Web Development",
      desc: "Full-stack development from frontend to backend",
      icon: "🌐",
      color: "#45B7D1",
      items: ["HTML/CSS", "React", "Node.js", "Databases", "APIs"],
      link: "/web-development"
    },
    {
      title: "AI & Machine Learning",
      desc: "Build intelligent systems and understand ML algorithms",
      icon: "🤖", 
      color: "#96CEB4",
      items: ["Neural Networks", "Deep Learning", "NLP", "Computer Vision"],
      link: "/machine-learning"
    },
    {
      title: "Interview Preparation",
      desc: "Crack coding interviews at top tech companies",
      icon: "🎯",
      color: "#FECA57",
      items: ["Coding Problems", "System Design", "Behavioral", "Mock Tests"],
      link: "/interview-prep"
    },
    {
      title: "Computer Science Fundamentals", 
      desc: "Core CS concepts every programmer should know",
      icon: "🧠",
      color: "#FF9FF3",
      items: ["OS", "Networks", "DBMS", "Compilers", "Architecture"],
      link: "/cs-fundamentals"
    }
  ]

  const practiceProblems = [
    {
      title: "Two Sum",
      difficulty: "Easy",
      category: "Array",
      solved: "2.3M",
      acceptance: "49.2%"
    },
    {
      title: "Longest Palindromic Substring", 
      difficulty: "Medium",
      category: "String",
      solved: "1.8M",
      acceptance: "32.1%"
    },
    {
      title: "Binary Tree Maximum Path Sum",
      difficulty: "Hard", 
      category: "Tree",
      solved: "890K",
      acceptance: "38.7%"
    },
    {
      title: "Merge k Sorted Lists",
      difficulty: "Hard",
      category: "Linked List", 
      solved: "1.2M",
      acceptance: "45.8%"
    }
  ]

  const companies = [
    { name: "Google", logo: "🔍", problems: 450 },
    { name: "Netflix", logo: "🎬", problems: 380 },
    { name: "Instagram", logo: "📷", problems: 320 },
    { name: "Spotify", logo: "🎵", problems: 290 },
    { name: "Dropbox", logo: "📁", problems: 250 },
    { name: "Pinterest", logo: "📌", problems: 210 }
  ]

  return (
    <div className="professional-homepage">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="container">
          <div className="hero-content">
            <div className="hero-text">
              <h1 className="hero-title">
                Master Python Programming
                <span className="highlight"> From Zero to Expert</span>
              </h1>
              <p className="hero-subtitle">
                Learn Python from basic syntax to advanced concepts like OOP, decorators, async programming, 
                data science, web development, and AI. Complete hands-on learning with real projects.
              </p>
              <div className="hero-cta">
                <Link to="/python-fundamentals" className="cta-primary">
                  Start Python Journey
                </Link>
                <Link to="/python-projects" className="cta-secondary">
                  View Python Projects
                </Link>
              </div>
            </div>
            <div className="hero-visual">
              <div className="floating-cards">
                <div className="code-card">
                  <div className="code-header">
                    <span className="lang">Python</span>
                    <span className="status">●</span>
                  </div>
                  <pre><code>{`class PythonStudent:
    def __init__(self, name, level="beginner"):
        self.name = name
        self.level = level
        self.skills = []
    
    def learn_topic(self, topic):
        self.skills.append(topic)
        print(f"{self.name} learned {topic}!")`}</code></pre>
                </div>
                <div className="stats-card">
                  <h4>Today's Progress</h4>
                  <div className="progress-item">
                    <span>Python Concepts</span>
                    <strong>24/30</strong>
                  </div>
                  <div className="progress-item">
                    <span>Projects Built</span>
                    <strong>8/12</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="hero-stats">
            {heroStats.map((stat, index) => (
              <div key={index} className="stat-item">
                <span className="stat-icon">{stat.icon}</span>
                <div className="stat-content">
                  <h3 className="stat-number">{stat.number}</h3>
                  <p className="stat-label">{stat.label}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Access Section */}
      <section className="quick-access-section">
        <div className="container">
          <div className="section-header">
            <h2>Choose Your Python Learning Path</h2>
            <p>Complete Python curriculum from beginner basics to advanced professional development</p>
          </div>
          
          <div className="quick-access-grid">
            {quickAccessCards.map((card, index) => (
              <Link key={index} to={card.link} className="access-card">
                <div className="card-header" style={{backgroundColor: card.color}}>
                  <span className="card-icon">{card.icon}</span>
                </div>
                <div className="card-content">
                  <h3 className="card-title">{card.title}</h3>
                  <p className="card-desc">{card.desc}</p>
                  <div className="card-items">
                    {card.items.map((item, idx) => (
                      <span key={idx} className="card-tag">{item}</span>
                    ))}
                  </div>
                </div>
                <div className="card-arrow">→</div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Content Sections */}
      <section className="content-section">
        <div className="container">
          <div className="content-tabs">
            <button 
              className={`tab-btn ${activeTab === 'trending' ? 'active' : ''}`}
              onClick={() => setActiveTab('trending')}
            >
              🔥 Trending Articles
            </button>
            <button 
              className={`tab-btn ${activeTab === 'practice' ? 'active' : ''}`}
              onClick={() => setActiveTab('practice')}
            >
              💪 Practice Problems
            </button>
            <button 
              className={`tab-btn ${activeTab === 'companies' ? 'active' : ''}`}
              onClick={() => setActiveTab('companies')}
            >
              🏢 Top Companies
            </button>
          </div>

          <div className="content-panels">
            {activeTab === 'trending' && (
              <div className="articles-panel">
                <div className="articles-grid">
                  {trendingArticles.map((article, index) => (
                    <Link key={index} to={article.link} className="article-card">
                      <div className="article-header">
                        <div className="article-meta">
                          <span className={`difficulty ${article.difficulty.toLowerCase()}`}>
                            {article.difficulty}
                          </span>
                          <span className="read-time">📖 {article.readTime}</span>
                        </div>
                        <div className="article-views">👁️ {article.views}</div>
                      </div>
                      <h3 className="article-title">{article.title}</h3>
                      <div className="article-tags">
                        {article.tags.map((tag, idx) => (
                          <span key={idx} className="tag">{tag}</span>
                        ))}
                      </div>
                      <div className="article-footer">
                        <span className="author">By {article.author}</span>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'practice' && (
              <div className="practice-panel">
                <div className="problems-list">
                  {practiceProblems.map((problem, index) => (
                    <div key={index} className="problem-row">
                      <div className="problem-info">
                        <h4 className="problem-title">{problem.title}</h4>
                        <div className="problem-meta">
                          <span className={`difficulty ${problem.difficulty.toLowerCase()}`}>
                            {problem.difficulty}
                          </span>
                          <span className="category">{problem.category}</span>
                        </div>
                      </div>
                      <div className="problem-stats">
                        <div className="stat">
                          <span className="stat-label">Solved</span>
                          <span className="stat-value">{problem.solved}</span>
                        </div>
                        <div className="stat">
                          <span className="stat-label">Acceptance</span>
                          <span className="stat-value">{problem.acceptance}</span>
                        </div>
                      </div>
                      <button className="solve-btn">Solve</button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'companies' && (
              <div className="companies-panel">
                <div className="companies-grid">
                  {companies.map((company, index) => (
                    <Link key={index} to={`/company/${company.name.toLowerCase()}`} className="company-card">
                      <div className="company-logo">{company.logo}</div>
                      <div className="company-info">
                        <h4 className="company-name">{company.name}</h4>
                        <p className="company-problems">{company.problems} Problems</p>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <div className="section-header">
            <h2>Why Choose PythonMaster?</h2>
            <p>The most comprehensive platform for mastering Python programming from basics to advanced</p>
          </div>
          
          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-icon">🎯</div>
              <h3>Interview-Focused</h3>
              <p>Practice problems from real interviews at Google, Amazon, Microsoft and more</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">🧠</div>
              <h3>Deep Understanding</h3>
              <p>Not just syntax - understand the 'why' behind algorithms and data structures</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">💻</div>
              <h3>Hands-on Practice</h3>
              <p>Built-in code editor with instant feedback and test cases</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">📊</div>
              <h3>Progress Tracking</h3>
              <p>Detailed analytics on your learning journey and skill development</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">🌟</div>
              <h3>Expert Content</h3>
              <p>Created by industry professionals and computer science PhDs</p>
            </div>
            <div className="feature-item">
              <div className="feature-icon">🚀</div>
              <h3>Career Support</h3>
              <p>Resume reviews, mock interviews, and job referral programs</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-content">
            <h2>Ready to Master Python?</h2>
            <p>Join hundreds of thousands of developers who've mastered Python with PythonMaster</p>
            <div className="cta-buttons">
              <Link to="/signup" className="cta-primary large">
                Start Free Trial
              </Link>
              <Link to="/pricing" className="cta-secondary large">
                View Pricing
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home