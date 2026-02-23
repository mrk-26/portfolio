import os
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# Configuration and Data Loading
# -----------------------------
DATASET_PATH = 'UCI HAR Dataset'
MODEL_PATH = 'final_model.h5'

# Load activity labels (assumes codes are 1-indexed)
activity_labels = pd.read_csv(
    os.path.join(DATASET_PATH, 'activity_labels.txt'),
    sep=' ',
    header=None,
    names=['code', 'label']
)

# Function to load test set data
def load_test_data(dataset_path):
    test_path = os.path.join(dataset_path, 'test')
    X = pd.read_csv(os.path.join(test_path, 'X_test.txt'),
                    delim_whitespace=True, header=None).values
    y = pd.read_csv(os.path.join(test_path, 'y_test.txt'),
                    header=None, names=['activity']).values
    return X, y

# Load test data
X_test, y_test = load_test_data(DATASET_PATH)

# For demonstration we fit a StandardScaler on X_test.
# In production, you should use the scaler fitted on training data.
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Also create a label encoder based on y_test.
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test.ravel())
num_classes = len(label_encoder.classes_)

# -----------------------------
# Load the Saved Model
# -----------------------------
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_activity(sample_index: int):
    """
    Given a sample index, this function fetches the preprocessed test sample,
    runs the model prediction, and returns the predicted activity along with
    the class probability distribution.
    """
    sample = X_test_scaled[sample_index].reshape(1, -1)  # shape (1, num_features)
    pred_probs = model.predict(sample)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    # Adjust for activity_labels (assumes labels in file are 1-indexed)
    predicted_label = activity_labels.loc[
        activity_labels['code'] == (pred_class + 1), 'label'
    ].values[0]
    
    # Create a formatted probability output
    prob_dict = {
        activity_labels.loc[i, 'label']: float(pred_probs[0][i])
        for i in range(len(pred_probs[0]))
    }
    # Sort probabilities in descending order
    prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    prob_text = "\n".join([f"{k}: {v:.4f}" for k, v in prob_dict.items()])
    
    return f"Predicted Activity: {predicted_label}\n\nClass Probabilities:\n{prob_text}"

# -----------------------------
# Build Gradio GUI
# -----------------------------
# Custom CSS for styling and animations
custom_css = """
body {
    background: linear-gradient(to right, #83a4d4, #b6fbff);
}
h1, p {
    text-align: center;
    color: #333;
}
#component-0 {
    animation: fadeIn 2s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>Human Activity Recognition (HAR) Model</h1>")
    gr.Markdown("<p>Select a test sample index and click 'Predict' to see the model's prediction.</p>")
    
    with gr.Row():
        sample_index_input = gr.Slider(
            minimum=0,
            maximum=X_test_scaled.shape[0] - 1,
            step=1,
            label="Test Sample Index",
            value=0
        )
    
    predict_button = gr.Button("Predict")
    output_text = gr.Textbox(label="Prediction Output", lines=10)
    
    predict_button.click(
        fn=predict_activity,
        inputs=sample_index_input,
        outputs=output_text
    )

# -----------------------------
# Launch the Interface
# -----------------------------
if __name__ == "__main__":
    demo.launch()
