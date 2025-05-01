import React from 'react';
import { Link } from 'react-router-dom';
import MetaBalls from '../components/MetaBalls';
import PixelCard from '../components/PixelCard';
import TrueFocus from '../components/TrueFocus';
import './Home.css';

// Add environment variables
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Home = () => {
    const renderCardContent = (icon, title, description, link, linkText) => (
        <div className="card-content">
            <div className="feature-icon">{icon}</div>
            <h2>{title}</h2>
            <p>{description}</p>
            <Link to={link} className="feature-link">
                {linkText} →
            </Link>
        </div>
    );

    // Add error boundary
    class ErrorBoundary extends React.Component {
        state = { hasError: false };
        
        static getDerivedStateFromError(error) {
            return { hasError: true };
        }
        
        render() {
            if (this.state.hasError) {
                return <h1>Something went wrong.</h1>;
            }
            return this.props.children;
        }
    }

    return (
        <div className="home-container">
            <div className="hero-section">
                <div className="hero-content">
                    <TrueFocus 
                        sentence="AI Presence"
                        manualMode={false}
                        blurAmount={5}
                        borderColor="#ffffff"
                        glowColor="rgba(255, 255, 255, 0.6)"
                        animationDuration={2}
                        pauseBetweenAnimations={1}
                    />
                    <p className="subtitle">Reveal the Truth Behind Every Frame</p>
                    <div className="cta-buttons">
                        <Link to="/photo" className="cta-button primary">Get Started</Link>
                    </div>
                </div>
            </div>

            <div className="metaballs-section">
                <MetaBalls
                    color="#ffffff"
                    cursorBallColor="#ffffff"
                    cursorBallSize={2}
                    ballCount={15}
                    animationSize={30}
                    enableMouseInteraction={true}
                    enableTransparency={true}
                    hoverSmoothness={0.05}
                    clumpFactor={1}
                    speed={0.3}
                />
            </div>

            <div className="features-grid">
                <PixelCard
                    variant="blue"
                    className="feature-pixel-card"
                >
                    {renderCardContent(
                        "📸",
                        "Image Analysis",
                        "Analyze images for signs of manipulation using state-of-the-art deep learning models.",
                        "/photo",
                        "Try Image Detection"
                    )}
                </PixelCard>

                <PixelCard
                    variant="yellow"
                    className="feature-pixel-card"
                >
                    {renderCardContent(
                        "📝",
                        "Text Analysis",
                        "Detect AI-generated text content using our advanced neural network model.",
                        "/text",
                        "Try Text Analysis"
                    )}
                </PixelCard>

                <PixelCard
                    variant="pink"
                    className="feature-pixel-card"
                >
                    {renderCardContent(
                        "🎥",
                        "Video Analysis",
                        "Upload videos to detect deepfake content using advanced frame-by-frame analysis.",
                        "/video",
                        "Try Video Detection"
                    )}
                </PixelCard>
            </div>

            <div className="info-section">
                <h2>How It Works</h2>
                <div className="steps">
                    <div className="step">
                        <div className="step-number">1</div>
                        <h3>Upload Content</h3>
                        <p>Choose between video, image, or text upload</p>
                    </div>
                    <div className="step">
                        <div className="step-number">2</div>
                        <h3>Analysis</h3>
                        <p>Our AI models process your content in real-time</p>
                    </div>
                    <div className="step">
                        <div className="step-number">3</div>
                        <h3>Results</h3>
                        <p>Get detailed analysis with confidence scores</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Home; 