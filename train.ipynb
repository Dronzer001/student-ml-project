import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json

# 🔹 Load model for SageMaker
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

# 🔹 Input processing
def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return pd.DataFrame([[float(request_body)]], columns=['hours'])
    raise ValueError("Unsupported content type")

# 🔹 Prediction
def predict_fn(input_data, model):
    return model.predict(input_data)

# 🔹 Output formatting
def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())

# 🔹 Training (local run)
if __name__ == "__main__":
    df = pd.read_csv("data/student.csv")

    X = df[['hours']]
    y = df['pass']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "model.joblib")

    print("✅ Model trained and saved!")
