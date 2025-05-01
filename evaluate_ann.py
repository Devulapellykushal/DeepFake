import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model = load_model('ANN.h5')

# Load your test data
# Replace these with your actual test texts and labels
test_texts = [
    "Example text 1...",
    "Example text 2...",
    # ...
]
test_labels = [
    0,  # 0 = Human, 1 = AI (or whatever your label convention is)
    1,
    # ...
]

# Preprocess and vectorize the test texts
X_test = vectorizer.transform(test_texts).toarray()
y_test = np.array(test_labels)

# Predict
y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_probs))
print("\nClassification Report:\n", classification_report(y_test, y_pred))