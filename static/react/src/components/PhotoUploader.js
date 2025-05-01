import React, { useState } from 'react';
import './PhotoUploader.css';

const PhotoUploader = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            // Check if file is an image
            if (!file.type.startsWith('image/')) {
                setError('Please select an image file (JPEG, PNG, etc.)');
                return;
            }
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setError(null);
            setResult(null);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select an image file first.');
            return;
        }

        setIsProcessing(true);
        setError(null);

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('http://localhost:8000/Detect', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error('Upload error:', err);
            setError('An error occurred while processing the image. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="photo-uploader">
            <h2>Image Deepfake Detection</h2>
            
            <div className="upload-section">
                <div className="file-input-container">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="file-input"
                        disabled={isProcessing}
                    />
                    <div className="file-input-label">
                        {selectedFile ? 'Change Image' : 'Select Image'}
                    </div>
                </div>

                {previewUrl && (
                    <div className="preview-section">
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className="image-preview"
                        />
                    </div>
                )}

<button
    onClick={handleUpload}
    disabled={!selectedFile || isProcessing}
    className={`upload-button ${isProcessing ? 'processing' : ''}`}
    aria-busy={isProcessing}
>
    {isProcessing ? (
        <>
            <span className="spinner"></span>
            <span>Analyzing...</span>
        </>
    ) : (
        'Analyze Image'
    )}
</button>


                {error && (
                    <div className="error-message">
                        <span role="img" aria-label="error">⚠️</span> {error}
                    </div>
                )}

                {result && (
                    <div className={`result-section ${result.result.toLowerCase()}`}>
                        <h3>Analysis Results</h3>
                        <div className="result-text">
                            This image appears to be: {result.result}
                        </div>
                        <div className="confidence">
                            Confidence: {result.confidence.toFixed(2)}%
                        </div>
                        <div className="analysis-details">
                            <h4>Analysis Details:</h4>
                            <ul>
                                <li>Face Detection: {result.face_detected ? '✓' : '✗'}</li>
                                <li>Image Quality: {result.quality}</li>
                                <li>Processing Time: {result.processing_time}s</li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PhotoUploader; 