import requests


def send_request(image_path):
    url = "http://192.168.1.7:4003/recognize-color"

    payload = {'target_class': 'car',
    'image_path': image_path}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)




def average_prob(file_prob_path):
    prob_sum = 0
    prob_count = 0
    with open(file_prob_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                img_path , prob = line.strip().split("\t")
                prob_sum += float(prob)
                prob_count += 1
            except:
                continue

    return prob_sum/prob_count

import os 



# Define the supported image file extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def find_images_in_folder(root_folder):
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if the file has an image extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Get the full path of the image file
                image_files.append(os.path.join(dirpath, filename))
    
    return image_files


with open('probs.txt', 'w') as f:
    pass

# Example usage
current_dir = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.join(current_dir, "dataset")  # Replace with your folder path"dataset"  # Replace with your folder path
image_files = find_images_in_folder(root_folder)

from tqdm import tqdm
for filepath in tqdm(image_files, total=len(image_files)):
    send_request(filepath)


avg = average_prob('probs.txt')
print (avg)