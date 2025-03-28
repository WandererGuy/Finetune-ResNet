import os 

for foldername in ["dataset_balanced", "dataset_balanced_augmented", "dataset_split"]:
    import shutil
    if os.path.exists(foldername):
        shutil.rmtree(foldername)


dataset_folder = "dataset"
total = 0 
for foldername in os.listdir(dataset_folder):
    folderpath = os.path.join(dataset_folder, foldername)
    total += len(os.listdir(folderpath))
    print (foldername, len(os.listdir(folderpath)))

average = int(total/len(os.listdir(dataset_folder)))
print ("average", average)
import random 
def get_random_elements(lst, n):
    if len(lst) < n:
        return lst  # Return all elements if n is greater than the list length
    else:
        return random.sample(lst, n)  # Return n random elements


dataset_folder = "dataset"
dest_dataset_folder = "dataset_balanced"
os.makedirs(dest_dataset_folder, exist_ok=True)

import shutil
for foldername in os.listdir(dataset_folder):
    dest_specific_folder = os.path.join(dest_dataset_folder, foldername)
    os.makedirs(dest_specific_folder, exist_ok=True)
    print (dest_specific_folder)
    folderpath = os.path.join(dataset_folder, foldername)
    all_file_ls = []
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        all_file_ls.append(filepath)

    tmp = get_random_elements(all_file_ls, average)

    for i in tmp:
        shutil.copy(i, dest_specific_folder)

dataset_folder = "dataset_balanced"
for foldername in os.listdir(dataset_folder):
    folderpath = os.path.join(dataset_folder, foldername)
    total += len(os.listdir(folderpath))
    print (foldername, len(os.listdir(folderpath)))