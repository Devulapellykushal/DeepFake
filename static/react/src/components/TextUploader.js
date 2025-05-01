import React, { useState } from 'react';
import './TextUploader.css';

const TextUploader = () => {
    const [text, setText] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleTextChange = (event) => {
        setText(event.target.value);
        setError(null);
        setResult(null);
    };

    const handleUpload = async () => {
        if (!text.trim()) {
            setError('Please enter some text to analyze.');
            return;
        }

        setIsProcessing(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:8000/Detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error('Upload error:', err);
            setError('An error occurred while processing the text. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleClear = () => {
        setText('');
        setError(null);
        setResult(null);
    };

    return (
        <div className="text-uploader">
            <h2>Text Analysis</h2>
            
            <div className="upload-section">
                <div className="text-input-container">
                    <textarea
                        value={text}
                        onChange={handleTextChange}
                        placeholder="Enter your text here..."
                        className="text-input"
                        disabled={isProcessing}
                        rows={10}
                    />
                    {text && (
                        <button 
                            className="clear-button"
                            onClick={handleClear}
                            aria-label="Clear text"
                        >
                            ×
                        </button>
                    )}
                </div>

                <button
                    onClick={handleUpload}
                    disabled={!text.trim() || isProcessing}
                    className={`upload-button ${isProcessing ? 'processing' : ''}`}
                    aria-busy={isProcessing}
                >
                    {isProcessing ? (
                        <>
                            <span className="spinner"></span>
                            <span>Analyzing...</span>
                        </>
                    ) : (
                        'Analyze Text'
                    )}
                </button>

                {error && (
                    <div className="error-message" role="alert">
                        <span role="img" aria-label="error">⚠️</span> {error}
                    </div>
                )}

                {result && (
                    <div 
                        className={`result-section ${(result.result || '').toLowerCase()}`}
                        role="status"
                    >
                        <h3>Analysis Results</h3>
                        <div className="result-text">
                            This text appears to be: {result.result || 'Unknown'}
                        </div>
                        {result.confidence !== undefined && (
                            <div className="confidence">
                                Confidence: {result.confidence.toFixed(2)}%
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default TextUploader; 