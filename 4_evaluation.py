
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision.models as models
import yaml 
with open('class.yaml', 'r') as file:
    class_config = yaml.safe_load(file)

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
    print (f"Predicted class name: {class_config[int(class_idx)]}")
    # If you have class names, you can map it to the corresponding label like this:
    # class_names = train_dataset.classes  # The class names from your ImageFolder dataset
    # predicted_label = class_names[class_idx]
    # print(f"Predicted class label: {predicted_label}")

    return class_idx  # Return the predicted class index

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL_INFER_PATH = config['MODEL_INFER_PATH']
image_path = config['IMAGE_INFER_PATH']
import yaml 

# Load the pre-trained ResNet-18 model
# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=False)  # set pretrained=False since you are loading custom weights
num_classes = config['num_classes']
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
# Load the model weights from the .pth file
checkpoint = torch.load(MODEL_INFER_PATH)
# Load the model weights into the ResNet-18 model
model.load_state_dict(checkpoint)

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
