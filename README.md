environment to set up cuda capable 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 
pip install ultralytics 
pip install fastapi uvicorn pydantic python-multipart

```
FINETUNE RESNET ON CLASSIFICATION TASK 

based on this 2 tutorial 
https://sidthoviti.com/fine-tuning-resnet50-pretrained-on-imagenet-for-cifar-10/
https://alirezasamar.com/blog/2023/03/fine-tuning-pre-trained-resnet-18-model-image-classification-pytorch/


this repo handles all kind of classification 
this time , i handle color classfication , but u can do with all kind of classification
# prepare
create folder ./dataset with each subfolder is a folder have class name which have images  
# split dataset
```
python 0_balance.py
python 1_augment.py
python 2_split_dataset.py
```

# training, fix num_classes inn config.yaml
```
python 3_train.py
```

# for evaluation or inference
```
python 4_evaluation.py
```

# for color server 
```
python main.py
```