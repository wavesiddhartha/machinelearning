import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import Foundations from './pages/Foundations'
import ComputerScience from './pages/ComputerScience'
import PythonBasics from './pages/PythonBasics'
import DataStructures from './pages/DataStructures'
import Mathematics from './pages/Mathematics'
import DataScience from './pages/DataScience'
import MachineLearning from './pages/MachineLearning'
import DeepLearning from './pages/DeepLearning'
import Projects from './pages/Projects'
import PythonOOP from './pages/PythonOOP'
import PythonAdvanced from './pages/PythonAdvanced'
import PythonProjects from './pages/PythonProjects'
import MachineLearningComplete from './pages/MachineLearningComplete'
import DataScienceComplete from './pages/DataScienceComplete'
import WebDevelopmentComplete from './pages/WebDevelopmentComplete'
import ComputerVisionComplete from './pages/ComputerVisionComplete'
import NaturalLanguageProcessingComplete from './pages/NaturalLanguageProcessingComplete'
import './App.css'

// Placeholder components for pages that will be created
const PlaceholderPage = ({ title }) => (
  <div className="page">
    <div className="content">
      <h1>{title}</h1>
      <p>This comprehensive section is coming soon! It will contain detailed explanations and examples.</p>
      <div className="placeholder-info">
        <h3>What will be covered:</h3>
        <ul>
          <li>Complete theoretical foundations</li>
          <li>Step-by-step practical examples</li>
          <li>Real-world applications</li>
          <li>Interactive code demonstrations</li>
        </ul>
      </div>
    </div>
  </div>
)

function App() {
  return (
    <div className="app">
      <Header />
      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          
          {/* Foundations */}
          <Route path="/foundations" element={<Foundations />} />
          <Route path="/computer-science" element={<ComputerScience />} />
          <Route path="/math-basics" element={<PlaceholderPage title="🧮 Math Foundations" />} />
          
          {/* Python Learning Paths */}
          <Route path="/python-fundamentals" element={<PythonBasics />} />
          <Route path="/python-oop" element={<PythonOOP />} />
          <Route path="/python-classes" element={<PythonOOP />} />
          <Route path="/python-advanced" element={<PythonAdvanced />} />
          <Route path="/python-decorators" element={<PythonAdvanced />} />
          <Route path="/python-generators" element={<PythonAdvanced />} />
          <Route path="/python-context-managers" element={<PythonAdvanced />} />
          <Route path="/python-async" element={<PythonAdvanced />} />
          <Route path="/python-performance" element={<PythonAdvanced />} />
          <Route path="/python-projects" element={<PythonProjects />} />
          <Route path="/python-beginner-projects" element={<PythonProjects />} />
          <Route path="/python-web-projects" element={<PythonProjects />} />
          <Route path="/python-data-projects" element={<PythonProjects />} />
          <Route path="/python-data-structures" element={<DataStructures />} />
          
          {/* Machine Learning */}
          <Route path="/machine-learning-complete" element={<MachineLearningComplete />} />
          <Route path="/ml-mathematics" element={<MachineLearningComplete />} />
          <Route path="/ml-data-preprocessing" element={<MachineLearningComplete />} />
          <Route path="/ml-supervised-learning" element={<MachineLearningComplete />} />
          <Route path="/ml-neural-networks" element={<MachineLearningComplete />} />
          
          {/* Advanced AI */}
          <Route path="/computer-vision-complete" element={<ComputerVisionComplete />} />
          <Route path="/cv-fundamentals" element={<ComputerVisionComplete />} />
          <Route path="/object-detection" element={<ComputerVisionComplete />} />
          <Route path="/image-processing" element={<ComputerVisionComplete />} />
          <Route path="/nlp-complete" element={<NaturalLanguageProcessingComplete />} />
          <Route path="/text-processing" element={<NaturalLanguageProcessingComplete />} />
          <Route path="/sentiment-analysis" element={<NaturalLanguageProcessingComplete />} />
          <Route path="/transformers-nlp" element={<NaturalLanguageProcessingComplete />} />
          
          {/* Mathematics */}
          <Route path="/mathematics-complete" element={<Mathematics />} />
          <Route path="/linear-algebra" element={<PlaceholderPage title="🔢 Linear Algebra Deep Dive" />} />
          <Route path="/calculus" element={<PlaceholderPage title="📈 Calculus & Statistics" />} />
          
          {/* Data Science */}
          <Route path="/data-science-complete" element={<DataScienceComplete />} />
          <Route path="/data-fundamentals" element={<DataScienceComplete />} />
          <Route path="/numpy-complete" element={<DataScienceComplete />} />
          <Route path="/pandas-complete" element={<DataScienceComplete />} />
          <Route path="/matplotlib-complete" element={<DataScienceComplete />} />
          <Route path="/data-visualization" element={<DataScienceComplete />} />
          
          {/* Web Development */}
          <Route path="/web-development-complete" element={<WebDevelopmentComplete />} />
          <Route path="/django-complete" element={<WebDevelopmentComplete />} />
          <Route path="/flask-complete" element={<WebDevelopmentComplete />} />
          <Route path="/fastapi-complete" element={<WebDevelopmentComplete />} />
          <Route path="/web-scraping" element={<WebDevelopmentComplete />} />
          <Route path="/rest-apis" element={<WebDevelopmentComplete />} />
          
          {/* AI & ML */}
          <Route path="/ml-theory" element={<MachineLearning />} />
          <Route path="/neural-networks" element={<DeepLearning />} />
          <Route path="/advanced-ai" element={<PlaceholderPage title="🚀 Advanced AI Techniques" />} />
          
          {/* Projects */}
          <Route path="/projects-beginner" element={<Projects />} />
          <Route path="/projects-advanced" element={<PlaceholderPage title="🔥 Advanced AI Projects" />} />
          <Route path="/real-world" element={<PlaceholderPage title="🌍 Real-World Applications" />} />
          
          {/* Legacy routes for backward compatibility */}
          <Route path="/basics" element={<PythonBasics />} />
          <Route path="/data-structures" element={<DataStructures />} />
          <Route path="/mathematics" element={<Mathematics />} />
          <Route path="/data-science" element={<DataScience />} />
          <Route path="/machine-learning" element={<MachineLearning />} />
          <Route path="/deep-learning" element={<DeepLearning />} />
          <Route path="/projects" element={<Projects />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App
