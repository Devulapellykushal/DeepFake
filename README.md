# AI Presence - AI Content Detection System

## 📋 Overview

AI Presence is a powerful web application that detects AI-generated or manipulated content across multiple media types. The system leverages advanced machine learning models to analyze images, videos, and text, providing accurate detection and detailed analysis.

## 📁 Project Structure

```
PS-1/
├── .venv/                    # Python virtual environment
├── Uploaded_Files/           # Temporary storage for uploaded media
├── functions/                # Backend utility functions
├── static/                   # Frontend and static assets
│   └── react/               # React frontend application
│       ├── node_modules/    # Dependencies (ignored)
│       ├── public/          # Public assets
│       └── src/             # React source code
│           ├── components/  # Reusable UI components
│           └── pages/       # Page components
├── Models/                   # Machine Learning Models
│   ├── efficientnet_trained_model.keras
│   ├── resnet_model.h5
│   ├── mobilenet_model.h5
│   ├── text_classification_pipeline.h5
│   └── deepfake_model.pth
├── server.py                # Main Flask server
├── final.py                 # Core detection logic
├── evaluate_ann.py          # Model evaluation script
├── requirements.txt         # Python dependencies
├── logoo.png               # Application logo
└── README.md               # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- npm 6 or higher

### Installation Steps

1. **Clone and Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

2. **Frontend Setup**
   ```bash
   cd static/react
   npm install
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   python server.py
   ```
   The backend API will be available at: http://localhost:8000

2. **Start the Frontend Development Server**
   ```bash
   cd static/react
   npm start
   ```
   The frontend will be available at: http://localhost:3000

## 🔄 How It Works

### 1. Content Upload
- Users can upload images, videos, or text through the web interface
- Files are temporarily stored in the `Uploaded_Files/` directory
- Supported formats:
  - Images: JPG, JPEG, PNG
  - Videos: MP4, AVI, MOV
  - Text: Plain text, TXT files

### 2. Processing Pipeline
1. **File Validation**
   - Format verification
   - Size and quality checks
   - Content type detection

2. **Analysis Based on Content Type**
   - **Images**: Deepfake detection using multiple models (EfficientNet, ResNet, MobileNet)
   - **Videos**: Frame-by-frame analysis using OpenCV
   - **Text**: AI-generated text detection using text classification pipeline

3. **Result Generation**
   - Confidence scores
   - Detailed analysis
   - Visual feedback (for images/videos)

### 3. Response Handling
- Results are returned to the frontend
- Temporary files are cleaned up
- Analysis history is maintained

## 🛠️ Technical Stack

### Backend
- **Flask**: Web framework
- **OpenCV**: Video and image processing
- **TensorFlow/Keras**: Machine learning models
- **scikit-learn**: Text analysis
- **NumPy**: Numerical computations

### Frontend
- **React**: UI framework
- **Axios**: API communication
- **CSS Modules**: Styling

## 📊 API Endpoints

### 1. Content Detection
```
POST /Detect
```
- Accepts multipart/form-data
- Returns JSON with detection results

### 2. Analysis Status
```
GET /status
```
- Returns current processing status
- Progress updates for video analysis

## 📝 Best Practices

### For Optimal Results
1. **Images**
   - Use high-resolution images (minimum 224x224 pixels)
   - Ensure good lighting
   - Clear face visibility
   - Supported formats: JPG, JPEG, PNG

2. **Videos**
   - Keep videos under 30 seconds
   - Use stable camera movement
   - Good audio quality
   - Supported formats: MP4, AVI, MOV

3. **Text**
   - Minimum 100 words
   - Clear formatting
   - Avoid special characters
   - Supported formats: TXT, plain text

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

