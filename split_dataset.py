import shutil
import os
import random
import time 
"""
given a folder with class as subfolders
split the data into train, val, test
for each class folder, take train, val, test ratio of images
"""


source_path = r"D:\color_classify\dataset"
dest = r"D:\color_classify\dataset_split"
os.makedirs(dest, exist_ok=True)
dest_train_folder = os.path.join(dest, 'train')
dest_val_folder = os.path.join(dest, 'val')
dest_test_folder = os.path.join(dest, 'test') # if you don'test'

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
for folder in [dest_train_folder, dest_val_folder, dest_test_folder]:
    if os.path.exists(folder):
        raise FileExistsError(f"folder {folder} already exists. Please remove the folder or rename")
    else:
        os.makedirs(folder)


for class_folder in os.listdir(source_path):
    train_img_ls = []
    val_img_ls = []
    test_img_ls = []
    new_train_folder = os.path.join(dest_train_folder, class_folder)
    new_val_folder = os.path.join(dest_val_folder, class_folder)
    new_test_folder = os.path.join(dest_test_folder, class_folder)
    for folder in [new_train_folder, new_val_folder, new_test_folder]:
        if os.path.exists(folder):
            raise FileExistsError(f"folder {folder} already exists. Please remove the folder or rename")
        else:
            os.makedirs(folder)

    class_folder_path = os.path.join(source_path, class_folder)
    total_num = len(os.listdir(class_folder_path))
    train_num, val_num, test_num = int(total_num*train_ratio), int(total_num*val_ratio), int(total_num*test_ratio)
    print ('-------------------------------------')
    print ('total num', total_num)
    print ("train num", train_num)
    print ("val num", val_num)
    print ("test num", test_num)
    # take image to put in train 
    all_img_set = set([os.path.join(class_folder_path, imge_name) for imge_name in os.listdir(class_folder_path)])
    tmp = random.sample(all_img_set, int(train_num))
    train_img_ls.extend(tmp)
    for i in tmp:
        all_img_set.remove(i)
    print ('total num after train', len(all_img_set))
    tmp = random.sample(all_img_set, int(val_num))
    val_img_ls.extend(tmp)
    for i in tmp:
        all_img_set.remove(i)
    print ('total num after val or approximate test number', len(all_img_set))
    test_img_ls.extend(list(all_img_set))
    for i in train_img_ls:
        shutil.copy(i, new_train_folder)
    for i in val_img_ls:
        shutil.copy(i, new_val_folder)
    for i in test_img_ls:
        shutil.copy(i, new_test_folder)
    time.sleep(2)
