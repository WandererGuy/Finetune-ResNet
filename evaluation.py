
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision.models as models

# Define a function to perform inference on an input image
def infer(image_path, model, device):
    # Load the image
    img = Image.open(image_path)
    
    # Apply the transformations
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    
    # Forward pass to get predictions
    with torch.no_grad():  # No need to track gradients during inference
        outputs = model(img_tensor)
        
    # Get the predicted class
    _, predicted_class = torch.max(outputs, 1)
    
    # Convert the predicted class index back to class label if necessary
    class_idx = predicted_class.item()
    print(f"Predicted class index: {class_idx}")
    # If you have class names, you can map it to the corresponding label like this:
    # class_names = train_dataset.classes  # The class names from your ImageFolder dataset
    # predicted_label = class_names[class_idx]
    # print(f"Predicted class label: {predicted_label}")

    return class_idx  # Return the predicted class index

image_path = r"C:\Users\Admin\CODE\work\OBJECT_COLOR\color_classify\test_images\0a8c3d92-3a63-4165-855d-bcb7253c67db_mask.jpg"

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
model.to(device)

# Load the trained model (assuming it's saved and restored, otherwise use the code you already have)
model.eval()  # Set the model to evaluation mode

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Example usage
class_idx = infer(image_path, model, device)
