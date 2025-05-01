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



from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import numpy as np
import cv2
import time
import pickle
import traceback
import warnings
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setup
load_dotenv()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploaded_Files")
MODEL_PATH = os.path.join(BASE_DIR, "efficientnet_trained_model.keras")
TEXT_PIPELINE_PATH = os.path.join(BASE_DIR, "text_classification_pipeline.h5")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# Load Models
try:
    print(f"Loading EfficientNet model from {MODEL_PATH}...")
    efficientnet_model = load_model(MODEL_PATH)
    print("EfficientNet model loaded successfully")
except Exception as e:
    print(f"Error loading EfficientNet model: {str(e)}")
    traceback.print_exc()
    efficientnet_model = None

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
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.secret_key = SECRET_KEY

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Utils
def preprocess_image(filepath):
    img = keras_image.load_img(filepath, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def detect_fake_image(filepath):
    try:
        start_time = time.time()
        if efficientnet_model is None:
            raise Exception("EfficientNet model not properly initialized")

        img_preprocessed = preprocess_image(filepath)
        prediction = efficientnet_model.predict(img_preprocessed)[0][0]

        result = "Fake" if prediction < 0.5 else "Real"
        confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100
        processing_time = time.time() - start_time

        return {
            'result': result,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 2)
        }
    except Exception as e:
        print(f"Error in detect_fake_image: {str(e)}")
        raise e

def detect_fake_video(filepath, frame_count=5):
    try:
        if efficientnet_model is None:
            raise Exception("EfficientNet model not properly initialized")

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, frame_count).astype(int)
        fake_predictions = 0

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_frame.jpg")
            cv2.imwrite(temp_path, rgb_frame)

            img_preprocessed = preprocess_image(temp_path)
            prediction = efficientnet_model.predict(img_preprocessed)[0][0]

            if prediction < 0.5:
                fake_predictions += 1

        cap.release()
        os.remove(temp_path)

        confidence = (fake_predictions / frame_count) * 100
        result = "Fake" if fake_predictions > frame_count / 2 else "Real"

        return {
            'result': result,
            'confidence': round(confidence, 2),
            'frame_count_analyzed': frame_count
        }
    except Exception as e:
        print(f"Error in detect_fake_video: {str(e)}")
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
