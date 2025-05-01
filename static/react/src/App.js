import React from 'react';
import { NavLink, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import './App.css';
import PhotoUploader from './components/PhotoUploader';
import Squares from './components/Squares';
import TextUploader from './components/TextUploader';
import VideoUploader from './components/VideoUploader';
import Home from './pages/Home';

function App() {
  return (
    <Router>
      <div className="app">
        <Squares 
          speed={0.5} 
          squareSize={60}
          direction='diagonal'
          borderColor='rgb(97, 98, 95)'
          hoverFillColor='rgb(34, 34, 34)'
        />
        <nav className="navbar">
          <div className="nav-brand">
            <NavLink to="/" className="brand-link">
              AI Presence
            </NavLink>
          </div>
          
          <div className="nav-links">
            <NavLink 
              to="/" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              end
            >
              Home
            </NavLink>
            
            <NavLink 
              to="/photo" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            >
              Image Detection
            </NavLink>
            <NavLink 
              to="/text" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            >
              Text Analysis
            </NavLink>
            <NavLink 
              to="/video" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            >
              Video Detection
            </NavLink>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/photo" element={<PhotoUploader />} />
            <Route path="/text" element={<TextUploader />} />
            <Route path="/video" element={<VideoUploader />} />
          </Routes>
        </main>

        <footer className="footer">
          <p>Powered by KMIT • Built with ❤️</p>
        </footer>
      </div>
    </Router>
  );
}

export default App; 