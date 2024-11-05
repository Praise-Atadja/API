from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load label encoders and scaler
label_encoders = {col: joblib.load(f'label_encoders/{col}_label_encoder.pkl') for col in [
    'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
    'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 
    'SkinCancer', 'AgeCategory', 'Race', 'GenHealth', 
    'ECG_Classification']}

scaler = joblib.load('standard_scaler.pkl')
model = load_model('first_heartmodel.h5')

# Define a mapping for binary categorical variables
binary_mapping = {
    'Yes': 1,
    'No': 0
}

# Define original labels for predictions
diagnosis_labels = [
    'Stable Post-MI; No Current Risk',
    'Healthy Cardiac Status',
    'Minor Arrhythmia; No Immediate Concern',
    'Acute Coronary Syndrome with High Risk of Recurrence',
    'Previous Myocardial Infarction; Stable Condition',
    'History of Myocardial Infarction; Risk of Complications',
    'Early Signs of Cardiac Risk',
    'Chronic Arrhythmia'
]

recommendation_labels = [
    'Annual cardiovascular exam; maintain healthy lifestyle.',
    'Maintain a balanced diet, regular exercise, and healthy sleep habits.',
    'Lifestyle modifications; reduce caffeine and stress.',
    'Immediate referral to a cardiologist; prescribe anticoagulants and lifestyle changes.',
    'Regular cardiovascular assessment every 3 months; monitor symptoms.',
    'Lifestyle counseling and medication adherence; low-dose aspirin advised.',
    'Begin preventive medication and cardiovascular fitness plan.',
    'Anti-arrhythmic drugs and possible ablation therapy; avoid strenuous activities.'
]

followup_labels = [
    'Annual check-up.',
    'Annual wellness check-up.',
    '3-month follow-up if symptoms develop.',
    'Weekly follow-up until condition stabilizes.',
    '3-month follow-up with primary care.',
    'Monthly check-ins to assess risk.',
    '6-month follow-up for early intervention.',
    'Bi-weekly follow-up with ECG monitoring.'
]

critical_alert_labels = [
    'Low',
    'Medium',
    'High'
]

referral_labels = [
    'No',
    'Yes',
    'Consider cardiology if symptoms worsen',
    'Consider preventive cardiology'
]

# Define the preprocessing function
def preprocess_new_data(data):
    # Prepare the data for prediction
    processed_data = []

    # Process each record
    for record in data:
        # Initialize a list for the encoded record
        encoded_record = []
        for col in ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 
                    'PhysicalHealth', 'MentalHealth', 'DiffWalking', 
                    'Sex', 'AgeCategory', 'Race', 'Diabetic', 
                    'PhysicalActivity', 'GenHealth', 'SleepTime', 
                    'Asthma', 'KidneyDisease', 'SkinCancer', 'ECG_Classification']:
            if col in record:
                # Handle binary categorical variables manually
                if col in ['Smoking', 'AlcoholDrinking', 'Stroke', 
                            'DiffWalking', 'Diabetic', 
                            'PhysicalActivity', 'Asthma', 
                            'KidneyDisease', 'SkinCancer']:
                    # Map 'Yes' to 1 and 'No' to 0, defaulting to 0 for unseen labels
                    encoded_value = binary_mapping.get(record[col], 0)  # Default to 0 if not found
                    encoded_record.append(encoded_value)
                elif col in ['Sex', 'AgeCategory', 'Race', 'GenHealth', 'ECG_Classification']:
                    # Encode other categorical variables using label encoders
                    try:
                        encoded_value = label_encoders[col].transform([record[col]])[0]
                        encoded_record.append(encoded_value)
                    except ValueError:
                        print(f"Warning: Unseen label '{record[col]}' in column '{col}'. Assigning 0.")
                        encoded_record.append(0)  # Default to 0 if unseen label found
                else:
                    encoded_record.append(record[col])  # Append continuous variables directly
            else:
                encoded_record.append(0)  # Assign 0 for missing values

        processed_data.append(encoded_record)

    # Convert to numpy array for model prediction
    processed_data = np.array(processed_data)

    # Standardize continuous columns
    continuous_columns_indices = [0, 4, 5, 13]  # Indices for ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    processed_data[:, continuous_columns_indices] = scaler.transform(processed_data[:, continuous_columns_indices])

    return processed_data

# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Incoming data:", data)  # Log incoming data

    # Check if data is a single entry or a list
    if isinstance(data, dict):  # Single entry
        input_data = [data]  # Wrap dict in a list to process
    elif isinstance(data, list):  # List of entries
        input_data = data
    else:
        return jsonify({"error": "Invalid input format, must be a JSON object or list of objects"}), 400

    try:
        processed_data = preprocess_new_data(input_data)
    except Exception as e:
        print("Error during preprocessing:", e)  # Log preprocessing errors
        return jsonify({"error": str(e)}), 400

    # Make predictions
    predictions = model.predict(processed_data)

    # Map predictions to original labels
    results = {
        "Diagnosis": [diagnosis_labels[i] for i in predictions[0].argmax(axis=1)],
        "Recommendation": [recommendation_labels[i] for i in predictions[1].argmax(axis=1)],
        "Critical Alert": [critical_alert_labels[i] for i in predictions[2].argmax(axis=1)],
        "Follow-Up": [followup_labels[i] for i in predictions[3].argmax(axis=1)],
        "Referral": [referral_labels[i] for i in predictions[4].argmax(axis=1)],
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=3009)