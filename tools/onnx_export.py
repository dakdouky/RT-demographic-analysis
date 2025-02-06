""" import torch
import torch.onnx

import os
import sys
sys.path.append(os.path.abspath('./src'))

from demographic_analysis import MultiTaskFaceClassifier

# Load the classifier model
model_path = './demographic_analysis/checkpoints/best_model.pth'
classifier = MultiTaskFaceClassifier(model_path)
classifier_model = classifier.model


classifier_model.eval()

# Set a dummy input for the model (e.g., a random tensor of shape (1, 3, 224, 224) for image input)
dummy_input = torch.randn(1, 3, 224, 224).to(classifier.device)  # Replace with the actual input size (batch size x channels x height x width)

# Export the model to ONNX format
onnx_model_path = './weights/demographic_analysis.onnx'
torch.onnx.export(classifier_model, dummy_input, onnx_model_path, opset_version=11)




 """
## validate 

import torch
import onnxruntime as ort
import numpy as np

# Load the converted PyTorch model
torch_model = torch.load('./weights/scrfd_10g_bnkps.pth')
torch_model.eval()

# Load the original ONNX model
onnx_model_path = './weights/scrfd_10g_bnkps.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust dimensions as needed

# Get PyTorch model output
with torch.no_grad():
    pytorch_output = torch_model(dummy_input)

# Get ONNX model output
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
onnx_output = ort_session.run(None, ort_inputs)

# Compare the outputs
difference = np.abs(pytorch_output.numpy() - onnx_output[0])
if np.all(difference < 1e-5):
    print("The outputs are nearly identical!")
else:
    print("The outputs differ.")

