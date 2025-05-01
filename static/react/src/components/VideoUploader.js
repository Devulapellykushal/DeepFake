import React, { useRef, useState } from 'react';
import './VideoUploader.css';

const VideoUploader = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const validateFile = (file) => {
        if (!file.type.startsWith('video/')) {
            setError('Please select a video file.');
            return false;
        }
        if (file.size > MAX_FILE_SIZE) {
            setError(`File size exceeds ${formatFileSize(MAX_FILE_SIZE)}. Please choose a smaller file.`);
            return false;
        }
        return true;
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            if (validateFile(file)) {
                setSelectedFile(file);
                setPreviewUrl(URL.createObjectURL(file));
                setError(null);
                setResult(null);
            }
        }
    };

    const handleDragEnter = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        if (file && validateFile(file)) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setError(null);
            setResult(null);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a video file first.');
            return;
        }

        setIsProcessing(true);
        setError(null);

        const formData = new FormData();
        formData.append('video', selectedFile);

        try {
            const response = await fetch('http://localhost:8000/Detect', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server response:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error('Upload error:', err);
            setError('An error occurred while processing the video. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleClear = (e) => {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        setSelectedFile(null);
        setPreviewUrl(null);
        setError(null);
        setResult(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const triggerFileInput = (e) => {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        if (!isProcessing && !selectedFile) {
            fileInputRef.current.click();
        }
    };

    const handleDropZoneClick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!selectedFile) {
            triggerFileInput();
        }
    };

    return (
        <div className="video-uploader">
            <h2>Video Deepfake Detection</h2>
            
            <div className="upload-section">
                <div 
                    className={`drop-zone ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={handleDropZoneClick}
                    role="button"
                    tabIndex="0"
                    aria-label="Upload video file"
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*"
                        onChange={handleFileChange}
                        className="file-input"
                        disabled={isProcessing}
                        aria-label="Choose video file"
                        onClick={(e) => e.stopPropagation()}
                    />
                    <div className="drop-zone-content">
                        {selectedFile ? (
                            <div className="selected-file">
                                <div className="file-info">
                                    <span className="file-name" title={selectedFile.name}>{selectedFile.name}</span>
                                    <span className="file-size">({formatFileSize(selectedFile.size)})</span>
                                </div>
                                <button 
                                    className="clear-button"
                                    onClick={handleClear}
                                    aria-label="Clear selected file"
                                >
                                    ×
                                </button>
                            </div>
                        ) : (
                            <>
                                <div className="upload-icon">📁</div>
                                <p>Drag and drop your video here or click to browse</p>
                                <span className="supported-formats">
                                    Supported formats: MP4, AVI, MOV (Max size: {formatFileSize(MAX_FILE_SIZE)})
                                </span>
                            </>
                        )}
                    </div>
                </div>

                {previewUrl && (
                    <div className="preview-section">
                        <video
                            src={previewUrl}
                            className="video-preview"
                            controls
                            preload="metadata"
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
                        'Analyze Video'
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
                            This video appears to be: {result.result || 'Unknown'}
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

export default VideoUploader; 