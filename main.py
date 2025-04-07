
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision.models as models
import yaml 
with open('class.yaml', 'r') as file:
    class_config = yaml.safe_load(file)

from torch.nn import functional as F
# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def add_prob_file(image_path,real_prob):
    
    with open('probs.txt', 'a') as file:
        file.write(image_path)
        file.write('\t')
        file.write(str(real_prob))
        file.write('\n')

import shutil
import uuid 
def add_pseudo_dataset(image_path, classified_class):
    pseudo_folder = "pseudo_dataset"
    if not os.path.exists(pseudo_folder):
        os.makedirs(pseudo_folder)
    save_path = os.path.join(pseudo_folder, classified_class, str(uuid.uuid4()) + ".jpg")
    os.makedirs(os.path.join(pseudo_folder, classified_class), exist_ok=True)
    shutil.copy(image_path, save_path)

def get_highest_prob(probs_tensor):
    # Get the index of the highest value
    max_index = torch.argmax(probs_tensor)

    # Get the highest probability
    max_prob = probs_tensor[0, max_index].item()  # Using `.item()` to convert the tensor value to a native Python type

    # Print the result
    print("Highest Probability:", max_prob)
    print("Index of Highest Probability:", max_index.item())
    return max_prob
# Define a function to perform inference on an input image
def infer(image_path, model, device):
    # Load the image
    img = Image.open(image_path)
    
    # Apply the transformations
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    
    # Forward pass to get predictions
    with torch.no_grad():  # No need to track gradients during inference
        outputs = model(img_tensor)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probs = F.softmax(outputs, dim=1)
    # print ("-------- outputs logits --------")
    # print (outputs)
    # print ("-------- probs --------")
    # print (probs)
    # Get the predicted class (highest probability)
    _, predicted_class = torch.max(probs, 1)
    real_prob =  get_highest_prob(probs)





    # Convert the predicted class index back to class label if necessary
    class_idx = predicted_class.item()
    print(f"Predicted class index: {class_idx}")
    class_name = class_config[int(class_idx)]
    print (f"Predicted class name: {class_name}")

    # add_prob_file(image_path,real_prob)
    # average_prob = 0.965928918293414
    # average_prob =  0.96
    # if real_prob >= average_prob:
    #     add_pseudo_dataset(image_path, class_name)


    return class_name  # Return the predicted class index

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL_INFER_PATH = config['MODEL_INFER_PATH']
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


from fastapi import FastAPI
import os 
import uvicorn
import logging
import configparser
from fastapi.staticfiles import StaticFiles
from time import sleep
import sys
from change_ip import main as change_ip_main
from fastapi import FastAPI, HTTPException, Form, APIRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='fastapi.log', filemode='w')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(os.path.join('config','config.ini'))
host_ip = config['DEFAULT']['host'] 
port_num = config['DEFAULT']['port'] 
production = config['DEFAULT']['production']

script_name = "main"

app = FastAPI()

import time 
@app.post("/recognize-color")
async def recognize_color(image_path: str = Form(...)):
    # Example usage
    start = time.time()
    class_name = infer(image_path, model, device)
    print ("inference time: ", time.time() - start)
    return {"color": class_name}

def main():
    change_ip_main()
    sleep(2)
    print('INITIALIZING FASTAPI SERVER')
    uvicorn.run(f"{script_name}:app", host=host_ip, port=int(port_num), reload=False, workers=1)

if __name__ == "__main__":
    main()
