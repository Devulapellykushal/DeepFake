# AI Presence - Advanced AI Content Detection System

## 📋 Overview

AI Presence is a sophisticated web application that detects AI-generated or manipulated content across multiple media types. The system employs state-of-the-art machine learning models to analyze images, videos, and text, providing accurate detection and comprehensive analysis. The project combines multiple deep learning architectures and traditional computer vision techniques to achieve high accuracy in content authenticity verification.

## 📁 Project Structure

```
PS-1/
├── .venv/                    # Python virtual environment
├── Uploaded_Files/          # Temporary storage for uploaded media
├── Images/                  # Sample images and test data
├── Videos/                  # Sample videos and test data
├── functions/              # Backend utility functions
├── static/                 # Frontend and static assets
├── DeepFake Project Screenshots/  # Project documentation images
├── Models/                 # Machine Learning Models
│   ├── efficientnet_trained_model.keras  # Primary model for image/video analysis (98.6% accuracy)
│   └── text_classification_pipeline.h5  # Text analysis pipeline
├── server.py              # Main Flask server (API endpoints and core logic)
├── final.py              # Core detection algorithms and model integration
├── evaluate_ann.py       # Model evaluation and testing script
├── requirements.txt      # Python dependencies
├── logoo.png            # Application logo
└── README.md            # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- macOS (for tensorflow-macos compatibility)
- Sufficient disk space for ML models (approximately 200MB)
- 8GB RAM minimum (16GB recommended for video processing)

### Installation Steps

1. **Clone and Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate 
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

2. **Verify Model Files**
   Ensure all model files are present in the root directory:
   - efficientnet_trained_model.keras
   - resnet_model.h5
   - mobilenet_model.h5
   - text_classification_pipeline.h5
   - deepfake_model.pth

### Running the Application

1. **Start the Backend Server**
   ```bash
   python server.py
   ```
   The backend API will be available at: http://localhost:8000

## 🔄 How It Works

### 1. Content Upload and Processing
The system accepts three types of content:

#### Images
- **Supported Formats**: JPG, JPEG, PNG
- **Processing Pipeline**:
  1. Face detection using face_recognition
  2. Image quality assessment (blur detection)
  3. Deep learning analysis using EfficientNet (98.6% accuracy)
  4. Confidence score calculation
  5. Result generation with detailed metrics

#### Videos
- **Supported Formats**: MP4, AVI, MOV
- **Processing Pipeline**:
  1. Frame extraction (5 frames by default)
  2. Per-frame analysis using EfficientNet model
  3. Aggregated result generation with confidence scores

#### Text
- **Supported Formats**: Plain text, TXT files
- **Processing Pipeline**:
  1. Text preprocessing and cleaning
  2. Feature extraction
  3. Classification using custom pipeline
  4. Confidence scoring
  5. Result generation

### 2. Technical Implementation

#### Backend Architecture
- **Flask Server** (`server.py`):
  - RESTful API endpoints
  - File upload handling
  - Model orchestration
  - Response generation

- **Core Detection** (`final.py`):
  - Model integration
  - Feature extraction
  - Analysis algorithms
  - Result aggregation

#### Machine Learning Models
1. **Image and Video Analysis**:
   - EfficientNet (efficientnet_trained_model.keras): Primary model for both image and video analysis
     - Achieves 98.6% accuracy on test sets
     - Used for both image classification and video frame analysis
     - Optimized for real-time processing
   - Face Recognition: Additional face detection and analysis

2. **Text Analysis**:
   - Custom classification pipeline
   - Feature extraction and analysis
   - Confidence scoring

### 3. API Endpoints

#### 1. Content Detection
```
POST /Detect
Content-Type: multipart/form-data
```
**Parameters**:
- `file`: Media file (image/video/text)
- `type`: Content type (image/video/text)

**Response**:
```json
{
    "result": "REAL/FAKE",
    "confidence": float,
    "details": {
        "face_detected": boolean,
        "face_count": integer,
        "quality": "High/Medium/Low",
        "processing_time": float,
        "model_confidence": float
    }
}
```

#### 2. Analysis Status
```
GET /status
```
Returns current processing status and progress for video analysis.

## 📊 Performance Metrics

### Image Detection
- Accuracy: 98.6% on standard test sets (EfficientNet model)
- Processing Time: <2 seconds per image
- Face Detection: >90% accuracy
- Quality Assessment: Blur detection with Laplacian variance

### Video Detection
- Frame Analysis: 5 frames per video
- Processing Time: ~10 seconds per 30-second video
- Model: Same EfficientNet model used for frame-by-frame analysis

### Text Analysis
- Minimum Text Length: 100 words
- Processing Time: <1 second
- Classification Accuracy: >90%

## 🛠️ Technical Stack

### Backend
- **Flask 3.0.2**: Web framework
- **OpenCV 4.9.0**: Video and image processing
- **TensorFlow 2.16.1**: Deep learning framework
- **PyTorch 2.2.1**: Deep learning framework
- **Transformers 4.38.2**: Hugging Face models
- **scikit-learn 1.4.0**: Machine learning utilities
- **face-recognition 1.3.0**: Face detection
- **NumPy 1.26.4**: Numerical computations

### Dependencies
- **Pillow 10.2.0**: Image processing
- **Flask-CORS 4.0.0**: Cross-origin resource sharing
- **python-dotenv 1.0.1**: Environment management
- **Werkzeug 3.0.1**: WSGI utilities

## 📝 Best Practices

### For Optimal Results
1. **Images**
   - Resolution: Minimum 224x224 pixels
   - Format: JPG, JPEG, PNG
   - Lighting: Well-lit, clear images
   - Face Visibility: Clear, unobstructed faces
   - File Size: <10MB

2. **Videos**
   - Duration: <30 seconds recommended
   - Resolution: 720p or higher
   - Format: MP4, AVI, MOV
   - Stability: Minimal camera movement
   - File Size: <128MB

3. **Text**
   - Length: Minimum 100 words
   - Format: Plain text, TXT
   - Language: English
   - Content: Clear, well-formatted text
   - File Size: <1MB

## 🔒 Security Considerations

1. **File Upload**
   - Maximum file size: 128MB
   - Secure filename handling
   - File type validation
   - Temporary storage with automatic cleanup

2. **Model Security**
   - Model files integrity verification
   - Secure model loading
   - Error handling and fallbacks

3. **API Security**
   - CORS configuration
   - Request validation
   - Error handling
   - Rate limiting (implemented)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

