import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from skimage import color
from skimage.io import imread
from skimage.transform import resize
from skimage.metrics import structural_similarity
from skimage.filters import gaussian, threshold_otsu
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
from firebase_admin import credentials, initialize_app, storage
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Allow cross-origin requests from the specified frontend URL
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Initialize Firebase
cred = credentials.Certificate('missioncapstone-21b12-firebase-adminsdk-9p748-0a02dc3abd.json')
initialize_app(cred, {'storageBucket': 'missioncapstone-21b12.appspot.com'})

# Load the model
loaded_model = joblib.load('final_model.pkl')

reference_image_urls = {
    "PMI": "https://firebasestorage.googleapis.com/v0/b/missioncapstone-21b12.appspot.com/o/images%2FPMI(1).jpg?alt=media&token=08aab8f7-83dc-4f83-a9db-db9903ada689",
    "HB": "https://firebasestorage.googleapis.com/v0/b/missioncapstone-21b12.appspot.com/o/images%2FHB(1).jpg?alt=media&token=78b8d2d6-0be6-4616-ad74-74357a6cb3c7",
    "Normal": "https://firebasestorage.googleapis.com/v0/b/missioncapstone-21b12.appspot.com/o/images%2FNormal(1).jpg?alt=media&token=8339f28d-3025-43ff-b241-3f14087363f3",
    "MI": "https://firebasestorage.googleapis.com/v0/b/missioncapstone-21b12.appspot.com/o/images%2FMI(1).jpg?alt=media&token=c5907959-7a64-4ea5-8a9e-63da5aeed9eb"
}

import os

import os
import tempfile

def fetch_image_from_firebase(image_name):
    """Fetch the specified image from Firebase Storage."""
    bucket = storage.bucket()
    blob = bucket.blob(image_name)

    # Use tempfile to create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        temp_image_path = tmp_file.name

    try:
        blob.download_to_filename(temp_image_path)
        return temp_image_path
    except Exception as e:
        print(f"Error downloading image '{image_name}': {e}")
        return None


def fetch_most_recent_image():
    """Fetch the most recent image from Firebase Storage."""
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix='images/')  # Adjust prefix as necessary

    # Extract image names and their last modified times
    images = [(blob.name, blob.updated) for blob in blobs if blob.name.endswith('.jpg')]
    
    if not images:
        print("No images found in the specified path.")
        return None

    # Find the most recent image based on last modified time
    most_recent_image = max(images, key=lambda x: x[1])[0]  # Get the name of the most recent image
    print(f"Most recent image found: {most_recent_image}")  # Debugging output

    # Download the most recent image
    return fetch_image_from_firebase(most_recent_image)

import imageio
from skimage import color

def preprocess_image(filepath):
    """Preprocess the ECG image and extract leads."""
    image = imread(filepath)
    image_gray = color.rgb2gray(image)
    image_gray = resize(image_gray, (1572, 2213))

    similarity_scores = []
    for label, url in reference_image_urls.items():
        # Read the reference image from the URL
        reference_image = imageio.imread(url)
        reference_image_gray = color.rgb2gray(reference_image)
        reference_image_gray = resize(reference_image_gray, (1572, 2213))

        # Calculate similarity score
        score = structural_similarity(image_gray, reference_image_gray, data_range=1)
        similarity_scores.append(score)

    max_similarity_score = max(similarity_scores)
    print(f"Max similarity score: {max_similarity_score}")  # Debugging output

    # Check similarity score threshold
    if max_similarity_score < 0.70:
        return None, "Image similarity is below threshold"

    # Extract leads from the ECG image
    leads = []
    lead_positions = [
        (300, 600, 150, 643), (300, 600, 646, 1135), (300, 600, 1140, 1625),
        (300, 600, 1630, 2125), (600, 900, 150, 643), (600, 900, 646, 1135),
        (600, 900, 1140, 1625), (600, 900, 1630, 2125), (900, 1200, 150, 643),
        (900, 1200, 646, 1135), (900, 1200, 1140, 1625), (900, 1200, 1630, 2125),
        (1250, 1480, 150, 2125)
    ]

    for start_row, end_row, start_col, end_col in lead_positions:
        lead = resize(image[start_row:end_row, start_col:end_col], (300, 450))
        leads.append(lead)

    return leads, max_similarity_score


def extract_and_scale_leads(leads):
    """Extract signals from the leads and scale the data."""
    scaler = MinMaxScaler()
    combined_signals = []

    for lead_no, lead in enumerate(leads):
        grayscale = color.rgb2gray(lead)
        blurred_image = gaussian(grayscale, sigma=0.9)
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh
        contours = measure.find_contours(binary_global, 0.8)

        # Assuming the largest contour corresponds to the signal
        if contours:
            contour = max(contours, key=len)
            contour_scaled = resize(contour, (255, 2))
            fit_transform_data = scaler.fit_transform(contour_scaled)
            normalized_scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X']).T
            combined_signals.append(normalized_scaled)

    # Combine all leads into one DataFrame
    if combined_signals:
        final_signals = pd.concat(combined_signals, axis=1).fillna(0)
        return final_signals
    return None

@app.route('/predict', methods=['POST'])
def predict_ecg():
    """Predict ECG condition from the most recent image in Firebase Storage."""
    
    # Step 1: Fetch the most recent image from Firebase Storage
    most_recent_image = fetch_most_recent_image()
    if not most_recent_image:
        return jsonify({"error": "No recent image found in Firebase"}), 404

    # Step 2: Preprocess the fetched image
    leads, similarity_score = preprocess_image(most_recent_image)
    if leads is None:
        return jsonify({"error": f"Preprocessing error: {similarity_score}"}), 400

    # Step 3: Extract and scale leads
    final_signals = extract_and_scale_leads(leads)
    if final_signals is None:
        return jsonify({"error": "Error extracting signals from leads"}), 500

    # Step 4: Ensure the input features match the model's expectations
    num_expected_features = loaded_model.n_features_in_
    if final_signals.shape[1] > num_expected_features:
        final_signals = final_signals.iloc[:, :num_expected_features]
    elif final_signals.shape[1] < num_expected_features:
        for _ in range(num_expected_features - final_signals.shape[1]):
            final_signals[final_signals.shape[1]] = 0  # Add missing columns

    # Step 5: Predict ECG condition
    result = loaded_model.predict(final_signals)

    diagnosis_map = {
        0: "Myocardial Infarction",
        1: "Abnormal Heartbeat",
        2: "Normal",
        3: "History of Myocardial Infarction"
    }
    diagnosis = diagnosis_map.get(result[0], "Unknown Condition")

    return jsonify({
        "diagnosis": diagnosis,
        "similarity_score": similarity_score
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=3008)