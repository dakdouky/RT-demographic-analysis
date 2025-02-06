import torch
import onnx
import onnxruntime as ort
import numpy as np
import cv2

import os
import sys
sys.path.append(os.path.abspath('./src'))


from demographic_analysis import MultiTaskFaceClassifier


# Load the original PyTorch model
model_path = './demographic_analysis/checkpoints/best_model.pth'
classifier = MultiTaskFaceClassifier(model_path)
classifier_model = classifier.model


classifier_model.eval()


# Set a dummy input for the model (e.g., random image of size (1, 3, 224, 224) for the classifier)
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust dimensions as required by your model

# Run inference on the PyTorch model
with torch.no_grad():
    pytorch_output = classifier_model(dummy_input.to(classifier.device))
    print("PyTorch Model Output:", pytorch_output)

# Export the PyTorch model to ONNX
onnx_model_path = './weights/demographic_analysis.onnx'
torch.onnx.export(classifier_model, dummy_input.to(classifier.device), onnx_model_path, opset_version=11)

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Validate the ONNX model
onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime session for inference
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare the input data in the correct format for ONNX
input_data_onnx = dummy_input.numpy()  # Convert the PyTorch tensor to NumPy for ONNX Runtime

# Run inference on the ONNX model
onnx_output = ort_session.run(None, {"input.1": input_data_onnx})  # Replace 'input' with the correct input name for your model

# Print the ONNX model output
print("ONNX Model Output:", onnx_output)

# Compare the outputs
tolerance = 1e-5  # Set a tolerance level to account for any small numerical differences
difference = np.abs(pytorch_output[0].cpu().numpy() - onnx_output[0])  # Compare the outputs
if np.all(difference < tolerance):
    print("The outputs are nearly identical!")
else:
    print("The outputs differ.")

# Optional: Print the difference for each output element
print("Differences between PyTorch and ONNX model outputs:", difference)
