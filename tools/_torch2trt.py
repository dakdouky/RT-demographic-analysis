import torch2trt
from your_model_definition import MultiTaskFaceClassifier  # Replace with your actual import

# Load your pre-trained PyTorch model
model_path = './demographic_analysis/checkpoints/best_model.pth'
classifier = MultiTaskFaceClassifier()
classifier.load_state_dict(torch.load(model_path))
classifier.eval().cuda()

# Create example input tensor
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# Convert to TensorRT
classifier_trt = torch2trt(classifier, [dummy_input])

# Save the TensorRT model
torch.save(classifier_trt.state_dict(), 'multitask_mobilenetv2_age_classification_trt.pth')
