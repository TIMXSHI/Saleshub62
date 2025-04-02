from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # a dict of all LabelEncoders
true_value_encoder = label_encoders["True Value"]

# Define expected fields in correct order
expected_fields = ['Project Name', 'Project Type', 'Project Stage', 'Invoice Amount',
                   'Recurring Expense?', 'Useful Life Expectancy', 'Vendor Name', 'Department']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Encode text fields
        encoded_features = []
        for field in expected_fields:
            value = data[field]
            if field in label_encoders:
                le = label_encoders[field]
                value = le.transform([value])[0]
            encoded_features.append(value)

        features_np = np.array(encoded_features).reshape(1, -1)
        prediction = model.predict(features_np)
        decoded = true_value_encoder.inverse_transform(prediction)

        return jsonify({
            "prediction_code": int(prediction[0]),
            "prediction_label": decoded[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
