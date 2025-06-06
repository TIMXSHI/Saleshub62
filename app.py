from flask import Flask, request, jsonify
import joblib
import numpy as np
from flasgger import Swagger

app = Flask(__name__)

# Flasgger config to serve OpenAPI JSON at a consistent endpoint
app.config['SWAGGER'] = {
    'title': 'Cost Categorisation API',
    'uiversion': 3,
    'specs': [
        {
            'endpoint': 'apispec',
            'route': '/swagger.json',
            'rule_filter': lambda rule: True,
            'model_filter': lambda tag: True,
        }
    ]
}


Swagger(app)  # Initialize Swagger UI

# Load model and label encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoder.pkl")  # a dict of all LabelEncoders
true_value_encoder = label_encoders["True Value"]

# Define expected fields in correct order
expected_fields = [ 'Invoice Amount','Project Stage', 
                   'Recurring Expense?', 'Useful Life Expectancy','Project Type']

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict cost category (Capex/Opex)
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            Project Type:
              type: string
            Project Stage:
              type: string
            Invoice Amount:
              type: number
            Recurring Expense?:
              type: integer
            Useful Life Expectancy:
              type: number
            
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction_label:
              type: string
    """
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
            "prediction_label": decoded[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/", methods=["GET"])
def home():
    return "🚀 Flask ML API is live!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
