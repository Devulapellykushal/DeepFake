# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import os
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision import transforms
# import numpy as np
# import cv2
# import face_recognition
# import time
# from dotenv import load_dotenv
# import warnings
# import traceback
# import pickle

# warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Setup
# load_dotenv()
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
# MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.pth")
# TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
# SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# # Load Text Classification Pipeline Model
# try:
#     print(f"Loading text classification pipeline from {TEXT_PIPELINE_PATH}...")
#     with open(TEXT_PIPELINE_PATH, 'rb') as handle:
#         text_pipeline = pickle.load(handle)
#     print("Text classification pipeline loaded successfully")
# except Exception as e:
#     print(f"Error loading text classification pipeline: {str(e)}")
#     traceback.print_exc()
#     text_pipeline = None

# class Model(nn.Module):
#     def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
#         super(Model, self).__init__()
#         model = models.resnext50_32x4d(pretrained=True)
#         self.model = nn.Sequential(*list(model.children())[:-2])
#         self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
#         self.relu = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.4)
#         self.linear1 = nn.Linear(2048, num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
#         x = x.view(batch_size * seq_length, c, h, w)
#         fmap = self.model(x)
#         x = self.avgpool(fmap)
#         x = x.view(batch_size, seq_length, 2048)
#         x_lstm, _ = self.lstm(x, None)
#         return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# # Load custom PyTorch model
# try:
#     print(f"Loading custom model from {MODEL_PATH}...")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     model = Model(num_classes=1)
#     print("Model architecture initialized")
    
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
#     state_dict = torch.load(MODEL_PATH, map_location=device)
#     model.load_state_dict(state_dict)
#     model = model.to(device)
#     model.eval()
#     print("Model successfully loaded and set to eval mode")
# except Exception as e:
#     print(f"Error loading custom model: {str(e)}")
#     traceback.print_exc()
#     model = None

# # Image preprocessing transforms
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # App config
# app = Flask("__main__", template_folder="templates")
# CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB
# app.config['MAX_CONTENT_PATH'] = None
# app.secret_key = SECRET_KEY

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # Load Hugging Face model for image detection
# try:
#     HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"
#     print(f"Loading model from {HF_MODEL_ID}...")
#     image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
#     image_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
#     image_model.eval()
#     LABELS = image_model.config.id2label
#     print("Successfully loaded model and processor")
# except Exception as e:
#     print(f"Error loading Hugging Face model: {str(e)}")
#     image_model = None
#     image_processor = None
#     LABELS = {0: 'REAL', 1: 'FAKE'}

# def detect_faces(image):
#     try:
#         face_locations = face_recognition.face_locations(image)
#         return len(face_locations) > 0, len(face_locations)
#     except:
#         return False, 0

# def assess_image_quality(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
#     if blur_score > 100:
#         return "High"
#     elif blur_score > 30:
#         return "Medium"
#     else:
#         return "Low"

# def detect_fake_image(image_path):
#     try:
#         start_time = time.time()
#         if image_model is None or image_processor is None:
#             raise Exception("Image detection model not properly initialized")
        
#         image = Image.open(image_path)
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')
#         image_np = np.array(image)

#         has_faces, face_count = detect_faces(image_np)
#         quality = assess_image_quality(image_np)

#         inputs = image_processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             outputs = image_model(**inputs)
#             logits = outputs.logits
#             predicted_idx = logits.argmax(-1).item()
#             confidence = logits.softmax(-1)[0, predicted_idx].item() * 100

#         processing_time = time.time() - start_time
        
#         return {
#             'result': LABELS[predicted_idx],
#             'confidence': confidence,
#             'face_detected': has_faces,
#             'face_count': face_count,
#             'quality': quality,
#             'processing_time': round(processing_time, 2)
#         }
#     except Exception as e:
#         print(f"Error in detect_fake_image: {str(e)}")
#         raise e

# def detect_fake_video(video_path, frame_count=5):
#     if image_model is None or image_processor is None:
#         return {'result': 'ERROR', 'confidence': 0, 'error': 'Image detection model not properly initialized'}
    
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise Exception("Failed to open video file")
            
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_indices = np.linspace(0, total_frames - 1, frame_count).astype(int)
#         fake_predictions = 0
        
#         for i in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(rgb_frame)

#             inputs = image_processor(images=image, return_tensors="pt")
#             with torch.no_grad():
#                 outputs = image_model(**inputs)
#                 logits = outputs.logits
#                 predicted_idx = logits.argmax(-1).item()
#                 if predicted_idx == 1:
#                     fake_predictions += 1

#         cap.release()

#         confidence = (fake_predictions / frame_count) * 100
#         result = "Fake" if fake_predictions > frame_count / 2 else "Real"

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'prediction_value': 1 if result == "Fake" else 0
#         }
                
#     except Exception as e:
#         print(f"Error processing video: {str(e)}")
#         return {'result': 'ERROR', 'confidence': 0, 'error': str(e)}

# def preprocess_text(text):
#     text = text.lower().strip()
#     if len(text.split()) < 5:
#         return None
#     return text

# def predict_text(text):
#     try:
#         if text_pipeline is None:
#             raise Exception("Text classification pipeline not properly initialized")

#         processed_text = preprocess_text(text)
#         if processed_text is None:
#             return {
#                 'result': 'Human Written',
#                 'confidence': 100.0,
#                 'prediction_value': 0,
#                 'threshold_used': 0.0,
#                 'analysis_details': {
#                     'text_length': len(text),
#                     'word_count': len(text.split()),
#                     'note': 'Text too short for reliable analysis'
#                 }
#             }

#         prediction = text_pipeline.predict([processed_text])[0]
#         prediction_proba = text_pipeline.predict_proba([processed_text])[0]

#         ai_confidence = prediction_proba[1] * 100
#         human_confidence = prediction_proba[0] * 100

#         result = "AI Generated" if prediction == 1 else "Human Written"
#         confidence = ai_confidence if prediction == 1 else human_confidence

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'prediction_value': int(prediction),
#             'threshold_used': 0.5,
#             'analysis_details': {
#                 'text_length': len(text),
#                 'word_count': len(text.split()),
#                 'ai_confidence': round(ai_confidence, 2),
#                 'human_confidence': round(human_confidence, 2),
#                 'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text
#             }
#         }
#     except Exception as e:
#         print(f"Error in text prediction: {str(e)}")
#         raise e

# @app.route('/', methods=['GET', 'POST'])
# def homepage():
#     return render_template('./index.html')

# @app.route('/Detect', methods=['POST'])
# def DetectPage():
#     if request.method == 'POST':
#         try:
#             if request.is_json:
#                 data = request.get_json()
#                 if 'text' in data:
#                     result = predict_text(data['text'])
#                     return jsonify(result)

#             content_length = request.content_length
#             if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
#                 return jsonify({'error': f'File too large. Max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

#             if 'video' in request.files:
#                 file = request.files['video']
#                 is_video = True
#             elif 'image' in request.files:
#                 file = request.files['image']
#                 is_video = False
#             else:
#                 return jsonify({'error': 'No file provided'}), 400

#             if file.filename == '':
#                 return jsonify({'error': 'No file selected'}), 400

#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             if is_video:
#                 result = detect_fake_video(filepath)
#             else:
#                 result = detect_fake_image(filepath)

#             os.remove(filepath)
#             return jsonify(result)

#         except Exception as e:
#             print(f"Error in DetectPage: {str(e)}")
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid request method'}), 405

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=8000, debug=True)





# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image as keras_image
# import os
# import numpy as np
# import cv2
# import face_recognition
# import time
# from dotenv import load_dotenv
# import warnings
# import traceback
# import pickle

# warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Setup
# load_dotenv()
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
# MODEL_PATH = os.path.join(BASE_DIR, "efficientnet_trained_model.keras")
# TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
# SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# # Load Text Classification Pipeline Model
# try:
#     print(f"Loading text classification pipeline from {TEXT_PIPELINE_PATH}...")
#     with open(TEXT_PIPELINE_PATH, 'rb') as handle:
#         text_pipeline = pickle.load(handle)
#     print("Text classification pipeline loaded successfully")
# except Exception as e:
#     print(f"Error loading text classification pipeline: {str(e)}")
#     traceback.print_exc()
#     text_pipeline = None

# # Load EfficientNet model
# try:
#     print(f"Loading EfficientNet model from {MODEL_PATH}...")
#     efficientnet_model = load_model(MODEL_PATH)
#     print("EfficientNet model loaded successfully")
# except Exception as e:
#     print(f"Error loading EfficientNet model: {str(e)}")
#     traceback.print_exc()
#     efficientnet_model = None

# # App config
# app = Flask("__main__", template_folder="templates")
# CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB
# app.config['MAX_CONTENT_PATH'] = None
# app.secret_key = SECRET_KEY

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # Utility Functions
# def detect_faces(image_np):
#     try:
#         face_locations = face_recognition.face_locations(image_np)
#         return len(face_locations) > 0, len(face_locations)
#     except:
#         return False, 0

# def assess_image_quality(image_np):
#     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
#     blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
#     if blur_score > 100:
#         return "High"
#     elif blur_score > 30:
#         return "Medium"
#     else:
#         return "Low"

# from tensorflow.keras.applications.efficientnet import preprocess_input

# def detect_fake_image(filepath):
#     try:
#         start_time = time.time()
#         if efficientnet_model is None:
#             raise Exception("EfficientNet model not properly initialized")

#         img = keras_image.load_img(filepath, target_size=(224, 224))
#         img_array = keras_image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)

#         img_array = preprocess_input(img_array)  # <- Important EfficientNet preprocessing

#         prediction = efficientnet_model.predict(img_array)
        
#         # If your model is binary classification (output shape (1,1)), threshold at 0.5
#         predicted_label = int(prediction[0][0] > 0.5)
#         confidence = float(prediction[0][0]) * 100 if predicted_label == 1 else (1 - float(prediction[0][0])) * 100

#         original_img = cv2.imread(filepath)
#         rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

#         has_faces, face_count = detect_faces(rgb_img)
#         quality = assess_image_quality(rgb_img)

#         processing_time = time.time() - start_time

#         result_label = "Real" if predicted_label == 1 else "Fake"

#         return {
#             'result': result_label,
#             'confidence': round(confidence, 2),
#             'face_detected': has_faces,
#             'face_count': face_count,
#             'quality': quality,
#             'processing_time': round(processing_time, 2)
#         }
#     except Exception as e:
#         print(f"Error in detect_fake_image: {str(e)}")
#         raise e

# def preprocess_text(text):
#     text = text.lower().strip()
#     if len(text.split()) < 5:
#         return None
#     return text

# def predict_text(text):
#     try:
#         if text_pipeline is None:
#             raise Exception("Text classification pipeline not properly initialized")

#         processed_text = preprocess_text(text)
#         if processed_text is None:
#             return {
#                 'result': 'Human Written',
#                 'confidence': 100.0,
#                 'prediction_value': 0,
#                 'threshold_used': 0.0,
#                 'analysis_details': {
#                     'text_length': len(text),
#                     'word_count': len(text.split()),
#                     'note': 'Text too short for reliable analysis'
#                 }
#             }

#         prediction = text_pipeline.predict([processed_text])[0]
#         prediction_proba = text_pipeline.predict_proba([processed_text])[0]

#         ai_confidence = prediction_proba[1] * 100
#         human_confidence = prediction_proba[0] * 100

#         result = "AI Generated" if prediction == 1 else "Human Written"
#         confidence = ai_confidence if prediction == 1 else human_confidence

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'prediction_value': int(prediction),
#             'threshold_used': 0.5,
#             'analysis_details': {
#                 'text_length': len(text),
#                 'word_count': len(text.split()),
#                 'ai_confidence': round(ai_confidence, 2),
#                 'human_confidence': round(human_confidence, 2),
#                 'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text
#             }
#         }
#     except Exception as e:
#         print(f"Error in text prediction: {str(e)}")
#         raise e

# # Routes
# @app.route('/', methods=['GET', 'POST'])
# def homepage():
#     return render_template('./index.html')

# @app.route('/Detect', methods=['POST'])
# def DetectPage():
#     if request.method == 'POST':
#         try:
#             if request.is_json:
#                 data = request.get_json()
#                 if 'text' in data:
#                     result = predict_text(data['text'])
#                     return jsonify(result)

#             content_length = request.content_length
#             if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
#                 return jsonify({'error': f'File too large. Max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

#             if 'image' in request.files:
#                 file = request.files['image']
#             else:
#                 return jsonify({'error': 'No file provided'}), 400

#             if file.filename == '':
#                 return jsonify({'error': 'No file selected'}), 400

#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             result = detect_fake_image(filepath)

#             os.remove(filepath)
#             return jsonify(result)

#         except Exception as e:
#             print(f"Error in DetectPage: {str(e)}")
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid request method'}), 405

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=8000, debug=True)


















# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import time
# import pickle
# import traceback
# import warnings
# from dotenv import load_dotenv

# warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Setup
# load_dotenv()
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
# RESNET_MODEL_PATH = os.path.join(BASE_DIR, "resnet_model.h5")
# TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
# SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# # Load ResNet model
# try:
#     print(f"Loading ResNet model from {RESNET_MODEL_PATH}...")
#     resnet_model = load_model(RESNET_MODEL_PATH)
#     print("ResNet model loaded successfully")
# except Exception as e:
#     print(f"Error loading ResNet model: {str(e)}")
#     traceback.print_exc()
#     resnet_model = None

# # Load Text Classification Pipeline
# try:
#     print(f"Loading text classification pipeline from {TEXT_PIPELINE_PATH}...")
#     with open(TEXT_PIPELINE_PATH, 'rb') as handle:
#         text_pipeline = pickle.load(handle)
#     print("Text classification pipeline loaded successfully")
# except Exception as e:
#     print(f"Error loading text classification pipeline: {str(e)}")
#     traceback.print_exc()
#     text_pipeline = None

# # Flask app setup
# app = Flask("__main__", template_folder="templates")
# CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB limit
# app.secret_key = SECRET_KEY

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # Utils
# def preprocess_image_for_resnet(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = np.array(img).astype('float32') / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def detect_fake_image(image_path):
#     try:
#         start_time = time.time()
#         if resnet_model is None:
#             raise Exception("ResNet model not properly initialized")

#         img_preprocessed = preprocess_image_for_resnet(image_path)
#         prediction = resnet_model.predict(img_preprocessed)[0][0]

#         result = 'FAKE' if prediction > 0.5 else 'REAL'
#         confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
#         processing_time = time.time() - start_time

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'processing_time': round(processing_time, 2)
#         }
#     except Exception as e:
#         print(f"Error in detect_fake_image: {str(e)}")
#         raise e

# def detect_fake_video(video_path, frame_count=5):
#     try:
#         if resnet_model is None:
#             raise Exception("ResNet model not properly initialized")

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise Exception("Failed to open video file")

#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_indices = np.linspace(0, total_frames - 1, frame_count).astype(int)
#         fake_predictions = 0

#         for i in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             temp_path = os.path.join(UPLOAD_FOLDER, "temp_frame.jpg")
#             cv2.imwrite(temp_path, rgb_frame)
#             img_preprocessed = preprocess_image_for_resnet(temp_path)

#             prediction = resnet_model.predict(img_preprocessed)[0][0]
#             if prediction > 0.5:
#                 fake_predictions += 1

#         cap.release()
#         os.remove(temp_path)

#         confidence = (fake_predictions / frame_count) * 100
#         result = "FAKE" if fake_predictions > frame_count / 2 else "REAL"

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'frame_count_analyzed': frame_count
#         }
#     except Exception as e:
#         print(f"Error in detect_fake_video: {str(e)}")
#         raise e

# def preprocess_text(text):
#     text = text.lower().strip()
#     return text

# def predict_text(text):
#     try:
#         if text_pipeline is None:
#             raise Exception("Text classification pipeline not properly initialized")

#         processed_text = preprocess_text(text)

#         prediction = text_pipeline.predict([processed_text])[0]
#         prediction_proba = text_pipeline.predict_proba([processed_text])[0]

#         ai_confidence = prediction_proba[1] * 100
#         human_confidence = prediction_proba[0] * 100

#         result = "AI Generated" if prediction == 1 else "Human Written"
#         confidence = ai_confidence if prediction == 1 else human_confidence

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'prediction_value': int(prediction),
#             'threshold_used': 0.5,
#             'analysis_details': {
#                 'text_length': len(text),
#                 'word_count': len(text.split()),
#                 'ai_confidence': round(ai_confidence, 2),
#                 'human_confidence': round(human_confidence, 2),
#                 'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text
#             }
#         }
#     except Exception as e:
#         print(f"Error in predict_text: {str(e)}")
#         raise e

# # Routes
# @app.route('/', methods=['GET', 'POST'])
# def homepage():
#     return render_template('./index.html')

# @app.route('/Detect', methods=['POST'])
# def DetectPage():
#     if request.method == 'POST':
#         try:
#             if request.is_json:
#                 data = request.get_json()
#                 if 'text' in data:
#                     result = predict_text(data['text'])
#                     return jsonify(result)

#             content_length = request.content_length
#             if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
#                 return jsonify({'error': f'File too large. Max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

#             if 'video' in request.files:
#                 file = request.files['video']
#                 is_video = True
#             elif 'image' in request.files:
#                 file = request.files['image']
#                 is_video = False
#             else:
#                 return jsonify({'error': 'No file provided'}), 400

#             if file.filename == '':
#                 return jsonify({'error': 'No file selected'}), 400

#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             if is_video:
#                 result = detect_fake_video(filepath)
#             else:
#                 result = detect_fake_image(filepath)

#             os.remove(filepath)
#             return jsonify(result)

#         except Exception as e:
#             print(f"Error in DetectPage: {str(e)}")
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid request method'}), 405

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=8000, debug=True)

# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from PIL import Image
# import time
# import pickle
# import traceback
# import warnings
# from dotenv import load_dotenv

# warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Setup
# load_dotenv()
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
# RESNET_MODEL_PATH = os.path.join(BASE_DIR, "resnet_model.h5")
# TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
# SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# # Load ResNet model
# try:
#     print(f"Loading ResNet model from {RESNET_MODEL_PATH}...")
#     resnet_model = load_model(RESNET_MODEL_PATH)
#     print("ResNet model loaded successfully")
# except Exception as e:
#     print(f"Error loading ResNet model: {str(e)}")
#     traceback.print_exc()
#     resnet_model = None

# # Load Text Classification Pipeline
# try:
#     print(f"Loading text classification pipeline from {TEXT_PIPELINE_PATH}...")
#     with open(TEXT_PIPELINE_PATH, 'rb') as handle:
#         text_pipeline = pickle.load(handle)
#     print("Text classification pipeline loaded successfully")
# except Exception as e:
#     print(f"Error loading text classification pipeline: {str(e)}")
#     traceback.print_exc()
#     text_pipeline = None

# # Flask app setup
# app = Flask("__main__", template_folder="templates")
# CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB limit
# app.secret_key = SECRET_KEY

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # Utils
# def preprocess_image_for_resnet(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = np.array(img).astype('float32')
#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def crop_face(image_path):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:
#         return None

#     x, y, w, h = faces[0]
#     face = img[y:y+h, x:x+w]
#     temp_face_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
#     cv2.imwrite(temp_face_path, face)
#     return temp_face_path

# def detect_fake_image(image_path, threshold=0.6):
#     try:
#         start_time = time.time()

#         cropped_face_path = crop_face(image_path)
#         if cropped_face_path is None:
#             raise Exception("No face detected in the image")

#         img_preprocessed = preprocess_image_for_resnet(cropped_face_path)
#         prediction = resnet_model.predict(img_preprocessed)[0][0]

#         result = 'FAKE' if prediction > threshold else 'REAL'
#         confidence = prediction * 100 if prediction > threshold else (1 - prediction) * 100
#         processing_time = time.time() - start_time

#         os.remove(cropped_face_path)

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'processing_time': round(processing_time, 2)
#         }
#     except Exception as e:
#         print(f"Error in detect_fake_image: {str(e)}")
#         raise e

# def preprocess_text(text):
#     text = text.lower().strip()
#     return text

# def predict_text(text):
#     try:
#         if text_pipeline is None:
#             raise Exception("Text classification pipeline not properly initialized")

#         processed_text = preprocess_text(text)

#         prediction = text_pipeline.predict([processed_text])[0]
#         prediction_proba = text_pipeline.predict_proba([processed_text])[0]

#         ai_confidence = prediction_proba[1] * 100
#         human_confidence = prediction_proba[0] * 100

#         result = "AI Generated" if prediction == 1 else "Human Written"
#         confidence = ai_confidence if prediction == 1 else human_confidence

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'prediction_value': int(prediction),
#             'threshold_used': 0.5,
#             'analysis_details': {
#                 'text_length': len(text),
#                 'word_count': len(text.split()),
#                 'ai_confidence': round(ai_confidence, 2),
#                 'human_confidence': round(human_confidence, 2),
#                 'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text
#             }
#         }
#     except Exception as e:
#         print(f"Error in predict_text: {str(e)}")
#         raise e

# # Routes
# @app.route('/', methods=['GET', 'POST'])
# def homepage():
#     return render_template('./index.html')

# @app.route('/Detect', methods=['POST'])
# def DetectPage():
#     if request.method == 'POST':
#         try:
#             if request.is_json:
#                 data = request.get_json()
#                 if 'text' in data:
#                     result = predict_text(data['text'])
#                     return jsonify(result)

#             content_length = request.content_length
#             if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
#                 return jsonify({'error': f'File too large. Max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

#             if 'video' in request.files:
#                 file = request.files['video']
#                 is_video = True
#             elif 'image' in request.files:
#                 file = request.files['image']
#                 is_video = False
#             else:
#                 return jsonify({'error': 'No file provided'}), 400

#             if file.filename == '':
#                 return jsonify({'error': 'No file selected'}), 400

#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             if is_video:
#                 result = detect_fake_video(filepath)
#             else:
#                 result = detect_fake_image(filepath)

#             os.remove(filepath)
#             return jsonify(result)

#         except Exception as e:
#             print(f"Error in DetectPage: {str(e)}")
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid request method'}), 405

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=8000, debug=True)



from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input  # Not needed now but keeping for future
from PIL import Image
import time
import pickle
import traceback
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setup
load_dotenv()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "resnet_model.h5")
TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# Load ResNet model
try:
    print(f"Loading ResNet model from {RESNET_MODEL_PATH}...")
    resnet_model = load_model(RESNET_MODEL_PATH)
    print("ResNet model loaded successfully")
except Exception as e:
    print(f"Error loading ResNet model: {str(e)}")
    traceback.print_exc()
    resnet_model = None

# Load Text Classification Pipeline
try:
    print(f"Loading text classification pipeline from {TEXT_PIPELINE_PATH}...")
    with open(TEXT_PIPELINE_PATH, 'rb') as handle:
        text_pipeline = pickle.load(handle)
    print("Text classification pipeline loaded successfully")
except Exception as e:
    print(f"Error loading text classification pipeline: {str(e)}")
    traceback.print_exc()
    text_pipeline = None

# Flask app setup
app = Flask("__main__", template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB limit
app.secret_key = SECRET_KEY

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Utils
def preprocess_image_for_resnet(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img.astype('float32') / 255.0  # Normalize between 0-1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_fake_image(image_path, threshold=0.5):
    try:
        start_time = time.time()

        img_preprocessed = preprocess_image_for_resnet(image_path)
        prediction = resnet_model.predict(img_preprocessed)[0][0]

        predicted_label = int(prediction > threshold)

        result = 'REAL' if predicted_label == 1 else 'FAKE'
        confidence = prediction * 100 if predicted_label == 1 else (1 - prediction) * 100
        processing_time = time.time() - start_time

        return {
            'result': result,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 2)
        }
    except Exception as e:
        print(f"Error in detect_fake_image: {str(e)}")
        raise e

def preprocess_text(text):
    text = text.lower().strip()
    return text

def predict_text(text):
    try:
        if text_pipeline is None:
            raise Exception("Text classification pipeline not properly initialized")

        processed_text = preprocess_text(text)

        prediction = text_pipeline.predict([processed_text])[0]
        prediction_proba = text_pipeline.predict_proba([processed_text])[0]

        ai_confidence = prediction_proba[1] * 100
        human_confidence = prediction_proba[0] * 100

        result = "AI Generated" if prediction == 1 else "Human Written"
        confidence = ai_confidence if prediction == 1 else human_confidence

        return {
            'result': result,
            'confidence': round(confidence, 2),
            'prediction_value': int(prediction),
            'threshold_used': 0.5,
            'analysis_details': {
                'text_length': len(text),
                'word_count': len(text.split()),
                'ai_confidence': round(ai_confidence, 2),
                'human_confidence': round(human_confidence, 2),
                'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text
            }
        }
    except Exception as e:
        print(f"Error in predict_text: {str(e)}")
        raise e

# Routes
@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('./index.html')

@app.route('/Detect', methods=['POST'])
def DetectPage():
    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json()
                if 'text' in data:
                    result = predict_text(data['text'])
                    return jsonify(result)

            content_length = request.content_length
            if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({'error': f'File too large. Max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

            if 'video' in request.files:
                file = request.files['video']
                is_video = True
            elif 'image' in request.files:
                file = request.files['image']
                is_video = False
            else:
                return jsonify({'error': 'No file provided'}), 400

            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if is_video:
                result = detect_fake_video(filepath)
            else:
                result = detect_fake_image(filepath)

            os.remove(filepath)
            return jsonify(result)

        except Exception as e:
            print(f"Error in DetectPage: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)




# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import time
# import pickle
# import traceback
# import warnings
# from dotenv import load_dotenv

# warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Setup
# load_dotenv()
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
# RESNET_MODEL_PATH = os.path.join(BASE_DIR, "resnet_model.h5")
# TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
# SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# # Load ResNet model
# try:
#     print(f"Loading ResNet model from {RESNET_MODEL_PATH}...")
#     resnet_model = load_model(RESNET_MODEL_PATH)
#     print("ResNet model loaded successfully")
# except Exception as e:
#     print(f"Error loading ResNet model: {str(e)}")
#     traceback.print_exc()
#     resnet_model = None

# # Load Text Classification Pipeline
# try:
#     print(f"Loading text classification pipeline from {TEXT_PIPELINE_PATH}...")
#     with open(TEXT_PIPELINE_PATH, 'rb') as handle:
#         text_pipeline = pickle.load(handle)
#     print("Text classification pipeline loaded successfully")
# except Exception as e:
#     print(f"Error loading text classification pipeline: {str(e)}")
#     traceback.print_exc()
#     text_pipeline = None

# # Flask app setup
# app = Flask("__main__", template_folder="templates")
# CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB limit
# app.secret_key = SECRET_KEY

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # Utils
# def preprocess_image_for_resnet(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     img_array = np.array(img).astype('float32')  # NO /255.0 normalization
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def detect_fake_image(image_path, threshold=0.5):
#     try:
#         start_time = time.time()

#         img_preprocessed = preprocess_image_for_resnet(image_path)
#         prediction = resnet_model.predict(img_preprocessed)[0][0]

#         predicted_label = int(prediction > threshold)

#         result = 'REAL' if predicted_label == 1 else 'FAKE'
#         confidence = prediction * 100 if predicted_label == 1 else (1 - prediction) * 100
#         processing_time = time.time() - start_time

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'processing_time': round(processing_time, 2)
#         }
#     except Exception as e:
#         print(f"Error in detect_fake_image: {str(e)}")
#         raise e

# def preprocess_text(text):
#     text = text.lower().strip()
#     return text

# def predict_text(text):
#     try:
#         if text_pipeline is None:
#             raise Exception("Text classification pipeline not properly initialized")

#         processed_text = preprocess_text(text)

#         prediction = text_pipeline.predict([processed_text])[0]
#         prediction_proba = text_pipeline.predict_proba([processed_text])[0]

#         ai_confidence = prediction_proba[1] * 100
#         human_confidence = prediction_proba[0] * 100

#         result = "AI Generated" if prediction == 1 else "Human Written"
#         confidence = ai_confidence if prediction == 1 else human_confidence

#         return {
#             'result': result,
#             'confidence': round(confidence, 2),
#             'prediction_value': int(prediction),
#             'threshold_used': 0.5,
#             'analysis_details': {
#                 'text_length': len(text),
#                 'word_count': len(text.split()),
#                 'ai_confidence': round(ai_confidence, 2),
#                 'human_confidence': round(human_confidence, 2),
#                 'processed_text': processed_text[:100] + '...' if len(processed_text) > 100 else processed_text
#             }
#         }
#     except Exception as e:
#         print(f"Error in predict_text: {str(e)}")
#         raise e

# # Routes
# @app.route('/', methods=['GET', 'POST'])
# def homepage():
#     return render_template('./index.html')

# @app.route('/Detect', methods=['POST'])
# def DetectPage():
#     if request.method == 'POST':
#         try:
#             if request.is_json:
#                 data = request.get_json()
#                 if 'text' in data:
#                     result = predict_text(data['text'])
#                     return jsonify(result)

#             content_length = request.content_length
#             if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
#                 return jsonify({'error': f'File too large. Max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413

#             if 'video' in request.files:
#                 file = request.files['video']
#                 is_video = True
#             elif 'image' in request.files:
#                 file = request.files['image']
#                 is_video = False
#             else:
#                 return jsonify({'error': 'No file provided'}), 400

#             if file.filename == '':
#                 return jsonify({'error': 'No file selected'}), 400

#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             if is_video:
#                 result = detect_fake_video(filepath)  # Note: You haven't given detect_fake_video logic, keep same
#             else:
#                 result = detect_fake_image(filepath)

#             os.remove(filepath)
#             return jsonify(result)

#         except Exception as e:
#             print(f"Error in DetectPage: {str(e)}")
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid request method'}), 405

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=8000, debug=True)
