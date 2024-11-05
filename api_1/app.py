import json
import os
from flask import Flask, request, jsonify
from firebase_admin import credentials, initialize_app, storage
from flask_cors import CORS
import uuid
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Firebase credentials from environment variable
firebase_cred_json = os.environ.get("FIREBASE_CREDENTIALS")
if firebase_cred_json:
    cred = credentials.Certificate(json.loads(firebase_cred_json))
    initialize_app(cred, {'storageBucket': 'missioncapstone-21b12.appspot.com'})
    bucket = storage.bucket()
else:
    raise ValueError("Firebase credentials not found in environment variables.")

@app.route('/store-patient-data', methods=['POST'])
def store_patient_data():
    try:
        data = request.json
        logging.info(f"Received data: {data}")
        
        if 'Name' not in data:
            return jsonify({'message': 'Name field is required.'}), 400
        
        # Create a unique file name
        file_name = f"{data['Name']}_{uuid.uuid4()}.json"
        blob = bucket.blob(file_name)

        # Upload data as JSON file to Firebase Storage
        blob.upload_from_string(
            json.dumps(data),
            content_type='application/json'
        )

        return jsonify({'message': f"Data for {data['Name']} stored successfully.", 'stored_data': data}), 200
    except Exception as e:
        logging.error(f"Error storing patient data: {e}")
        return jsonify({'message': 'Error storing patient data.', 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))  # Use the environment port or default to 3000
    app.run(debug=os.environ.get("FLASK_DEBUG") == '1', port=port)
