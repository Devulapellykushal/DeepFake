import React from 'react';
import { Link } from 'react-router-dom';
import logo from '../assets/logo.png';
import './Navbar.css';

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="navbar-container">
                <Link to="/" className="navbar-logo">
                    <img src={logo} alt="AI Presence Logo" className="logo-image" />
                </Link>
                <div className="nav-links">
                    <Link to="/photo" className="nav-link">Image Detection</Link>
                    <Link to="/text" className="nav-link">Text Analysis</Link>
                    <Link to="/video" className="nav-link">Video Detection</Link>
                    <a 
                        href="https://calendly.com/kushalkumar2506/30min" 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="nav-link cta-button"
                    >
                        Book a Demo
                    </a>
                </div>
            </div>
        </nav>
    );
};

export default Navbar; 