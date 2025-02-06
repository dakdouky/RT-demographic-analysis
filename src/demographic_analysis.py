import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2 

class MultiTaskFaceClassifier:
    def __init__(self, model_path, device=None):
        """
        Initialize the MultiTaskFaceClassifier.

        Args:
            model_path (str): Path to the trained model checkpoint.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to None (auto-detect).
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model
        self.model = self._load_model(model_path, self.device)

        # Define the same transform used during training
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define label mappings
        self.age_categories = ['Child (0-17)', 'Adult (18-64)', 'Senior (65+)']
        self.gender_categories = ['Male', 'Female']
        self.ethnicity_categories = ['White', 'Black', 'Asian', 'Indian', 'Others']

    def _load_model(self, model_path, device):
        """
        Load the trained model from the checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.
            device (str): Device to load the model onto.

        Returns:
            torch.nn.Module: Loaded model.
        """
        model = MultiTaskMobileNetV2(num_age_classes=3, num_gender_classes=2, num_ethnicity_classes=5)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        return model

    def _preprocess_image(self, image):
        """
        Preprocess the input image for inference.

        Args:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image

    def _get_labels(self, pred_age, pred_gender, pred_ethnicity):
        """
        Map predicted class indices to human-readable labels.

        Args:
            pred_age (int): Predicted age class index.
            pred_gender (int): Predicted gender class index.
            pred_ethnicity (int): Predicted ethnicity class index.

        Returns:
            tuple: (age_label, gender_label, ethnicity_label)
        """
        age_label = self.age_categories[pred_age]
        gender_label = self.gender_categories[pred_gender]
        ethnicity_label = self.ethnicity_categories[pred_ethnicity]
        return age_label, gender_label, ethnicity_label

    def predict(self, image):
        """
        Perform inference on a single image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: (age_label, gender_label, ethnicity_label)
        """
        # Preprocess the image
        image = self._preprocess_image(image)
        image = image.to(self.device)

        # Perform inference
        with torch.no_grad():
            age_logits, gender_logits, ethnicity_logits = self.model(image)

        # Get predictions
        pred_age = torch.argmax(age_logits, dim=1).item()
        pred_gender = torch.argmax(gender_logits, dim=1).item()
        pred_ethnicity = torch.argmax(ethnicity_logits, dim=1).item()

        # Map predictions to labels
        age_label, gender_label, ethnicity_label = self._get_labels(pred_age, pred_gender, pred_ethnicity)

        return age_label, gender_label, ethnicity_label

    def postprocess_trt(self, trt_logits): 

        age_logits, gender_logits, ethnicity_logits = trt_logits

        # Get predictions
        pred_age = torch.argmax(age_logits, dim=1).item()
        pred_gender = torch.argmax(gender_logits, dim=1).item()
        pred_ethnicity = torch.argmax(ethnicity_logits, dim=1).item()
        age_label, gender_label, ethnicity_label = self._get_labels(pred_age, pred_gender, pred_ethnicity)

        return age_label, gender_label, ethnicity_label

# Define the model architecture (same as during training)
class MultiTaskMobileNetV2(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes, num_ethnicity_classes):
        super(MultiTaskMobileNetV2, self).__init__()
        self.backbone = models.mobilenet_v2()
        self.backbone.classifier[1] = nn.Identity()  # Remove the final classification layer

        # Add task-specific heads
        self.age_head = nn.Linear(1280, num_age_classes)  # 3 classes for age
        self.gender_head = nn.Linear(1280, num_gender_classes)
        self.ethnicity_head = nn.Linear(1280, num_ethnicity_classes)

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        ethnicity = self.ethnicity_head(features)
        return age, gender, ethnicity


# Main function for testing
if __name__ == "__main__":
    # Path to the trained model checkpoint
    model_path = './demographic_analysis/checkpoints/multitask_mobilenetv2_age_classification.pth'

    # Initialize the classifier
    classifier = MultiTaskFaceClassifier(model_path)

    # Path to the input image
    image_path = '/home/toolkit/users/dakdouky_ws/RT-demographic-analysis/datasets/UTKFace/crop_part1/20_1_3_20170104222029447.jpg.chip.jpg'  # Replace with your image path

    # Perform inference
    age_label, gender_label, ethnicity_label = classifier.predict(image_path)

    # Print results
    print(f"Predicted Age: {age_label}")
    print(f"Predicted Gender: {gender_label}")
    print(f"Predicted Ethnicity: {ethnicity_label}")