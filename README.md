# UPAHAR
Ultrasound Placental image texture Analysis for prediction of Hypertension during pregnancy using Artificial intelligence Research



```
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import lr_scheduler
from torch.optim import Adam, SGD
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
import sys,os
import cv2
import PIL
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import time
import os
import shutil
import copy
import matplotlib.pyplot as plt
from random import randint
from PIL import Image 
cwd = os.getcwd()
cwd

```




    '/content'




```
# initialization
selected_cases="CH" # CH for control and htn , CF for control and fgr, CFH for a
selected_trimester="123" # 1 for first , 2 for second, 3 for third , 12 for first second, 23 for second third , 13 for first third, 123 for all
image_folder="saved_11_"+selected_cases+"_"+selected_trimester
results_folder="results_11_"+selected_cases+"_"+selected_trimester
c_path_control='/content/drive/MyDrive/Deep_Learning_Altrasound_Images/placenta_folder_filtered/placenta latest/controls---'
c_path_fgr='/content/drive/My Drive/Deep_Learning_Altrasound_Images/placenta_folder_filtered/placenta  folder/cases /fgr copy'
c_path_htn='/content/drive/MyDrive/Deep_Learning_Altrasound_Images/placenta_folder_filtered/placenta latest/cases htn images'
PATH="/content/drive/My Drive/Deep_Learning_Altrasound_Images/"
if not os.path.exists(PATH+image_folder):
      os.chdir(PATH)
      os.mkdir(image_folder)
if not os.path.exists(PATH+results_folder):
      os.chdir(PATH)
      os.mkdir(results_folder)
new_images_path=PATH+image_folder+"/"
PATH=PATH+results_folder+"/"

# most frequently changing parameters
k_fold=5 # defining fold change number to split data
entry_gate=False # means dividing images in train and val as well as labels wise from other source
num_epochs=25

new_image_size_1=256
new_image_size_2=224
check_image=400
test_size=.20
batch_size=8
num_images=2
pretrained_1='imagenet'
pretrained_2='imagenet'
pretrained_3=True
feature_extract = False # it makes three cases when pretrianed False, pretrianed True with feature_extract True, pretrianed True with feature_extract False
check="best_acc" # two options, best acc or best_loss
lr=.001
gamma=.1
step_size=5
other = str(check)+"_"+str(pretrained_1)+"_"+str(feature_extract)+\
                   "_"+str(num_epochs)+"_"+str(batch_size)+\
                   "_"+str(new_image_size_1)+"_"+str(new_image_size_2)+"_"+str(check_image)+\
                   "_"+str(test_size)+"_"+str(selected_cases)+"_"+str(selected_trimester)+\
                   "_"+str(lr)+"_"+str(gamma)+"_"+str(step_size)
models_list=["wide_resnet50_2","wide_resnet101_2",
             "resnext50_32x4d","resnext101_32x8d",
             "googlenet"]
```




    'models_list=["wide_resnet50_2","wide_resnet101_2",\n             "resnext50_32x4d","resnext101_32x8d",\n             "googlenet"]'




```
def get_trimester_wise_folder_list(folder_path,selected_trimester):
  path=os.listdir(folder_path)
  print("folder_path: ",folder_path)
  print("sub_folder_path : ",path)
  print("sub_folder_path_len : ",len(path))
  sub1=[]
  sub2=[]
  sub3=[]
  for i in path:
    sub_path=os.listdir(folder_path+"/"+i)
    sub1.extend([folder_path+"/"+i+"/"+k  for k in sub_path if k.find("first")!=-1])
    sub1.extend([folder_path+"/"+i+"/"+k  for k in sub_path if k.find("f")!=-1])
    sub2.extend([folder_path+"/"+i+"/"+k  for k in sub_path if k.find("second")!=-1])
    sub2.extend([folder_path+"/"+i+"/"+k  for k in sub_path if k.find("s")!=-1])
    sub3.extend([folder_path+"/"+i+"/"+k for k in sub_path if k.find("third")!=-1])
    sub3.extend([folder_path+"/"+i+"/"+k for k in sub_path if k.find("t")!=-1])
    temp=sorted([int(k)  for k in sub_path if k.find("first")==-1 and k.find("second")==-1 and k.find("third")==-1 and k.find("f")==-1 and k.find("s")==-1 and k.find("t")==-1])
    if(len(temp)>0):
      sub1.extend([folder_path+"/"+i+"/"+str(temp[0])])
    if(len(temp)>1):
      sub2.extend([folder_path+"/"+i+"/"+str(temp[1])])
    if(len(temp)>2):
      sub3.extend([folder_path+"/"+i+"/"+str(temp[2])])
  """print(sub1)
  print(sub2)
  print(sub3)"""
  if selected_trimester =="1":
    selected_path=sub1
  if selected_trimester =="2":
    selected_path=sub2
  if selected_trimester =="3":
    selected_path=sub3
  if selected_trimester =="12":
    selected_path=sub1
    selected_path.extend(sub2)
  if selected_trimester =="23":
    selected_path=sub2
    selected_path.extend(sub3)
  if selected_trimester =="13":
    selected_path=sub1
    selected_path.extend(sub3)
  if selected_trimester =="123":
    selected_path=sub1
    selected_path.extend(sub2)
    selected_path.extend(sub3)
  print("selected_path: ",selected_path)
  print("selected_path_len: ",len(selected_path))
  return (selected_path)
```


```
data_augmentation= {
    "RandomHorizontalFlip":transforms.RandomHorizontalFlip(p=1),
    "RandomVerticalFlip":transforms.RandomVerticalFlip(p=1),
    "RandomRotation":transforms.RandomRotation(degrees=(-180, 180)),
    "RandomAffine":transforms.RandomAffine(degrees=(-180,180),translate=(1,1)),
    "RandomCrop":transforms.RandomCrop(size=(64,64))
}
def exec_data_augmentation(same_path, im_ext ,old_image,data_set_type,rep_vector=[1,1,2]): # just to rep
  for k in data_augmentation:  # for data augmentation
      if k in ("RandomHorizontalFlip","RandomVerticalFlip"):
        for j in range(rep_vector[0]):
          print("size before image_augmentation "+k+" ",old_image.size)
          data_augmentation[k](old_image).save(same_path+"_"+k+"_"+str(j)+im_ext)
          final_count_images[data_set_type]+=1
      if k in ("RandomRotation","RandomAffine"):
        for j in range(rep_vector[1]):
          print("size before image_augmentation "+k+" ",old_image.size)
          data_augmentation[k](old_image).save(same_path+"_"+k+"_"+str(j)+im_ext)
          final_count_images[data_set_type]+=1
      if k in ("RandomCrop"):
        old_image=transforms.Resize(size=128)(old_image)
        old_image=transforms.CenterCrop(size=112)(old_image)
        for j in range(rep_vector[2]):
          print("size before image_augmentation "+k+" ",old_image.size)
          data_augmentation[k](old_image).save(same_path+"_"+k+"_"+str(j)+im_ext)
          final_count_images[data_set_type]+=1
  
def image_augmentation(same_path, im_ext ,old_image,data_set_type):
  # old_image=transforms.Resize(size=256)(old_image) # changing all images with same size
  if data_set_type in ["train_htn"]: # for downsampling
    if randint(0, 0)==0: # if randint(0, 3)==0: for downsampling as 1/4, if randint(0, 3)==4: to stop and randint(0, 0)==0: to keep one copy of original images
      transforms.ToPILImage()(old_image).save(same_path+im_ext)
      final_count_images[data_set_type]+=1
      exec_data_augmentation(same_path, im_ext ,transforms.ToPILImage()(old_image),data_set_type,rep_vector=[2,2,4])
  if data_set_type in ["train_fgr"]: # for downsampling
    if randint(0, 0)==0: # if randint(0, 1)==0: for downsampling as 1/2, if randint(0, 3)==4: to stop and randint(0, 0)==0: to keep one copy of original images
      transforms.ToPILImage()(old_image).save(same_path+im_ext)
      final_count_images[data_set_type]+=1
      exec_data_augmentation(same_path, im_ext ,transforms.ToPILImage()(old_image),data_set_type,rep_vector=[2,2,4])
  if data_set_type in ["train_control"]: # for downsampling
    if randint(0, 0)==0: # if randint(0, 3)==0: for downsampling as 1/4, if randint(0, 3)==4: to stop and randint(0, 0)==0: to keep one copy of original images
      transforms.ToPILImage()(old_image).save(same_path+im_ext)
      final_count_images[data_set_type]+=1
      exec_data_augmentation(same_path, im_ext ,transforms.ToPILImage()(old_image),data_set_type,rep_vector=[2,2,4])
  if data_set_type in ["val_control"]: # for neither downsampling nor upsampling
    if randint(0, 0)==0: # to keep one copy of original images
      transforms.ToPILImage()(old_image).save(same_path+im_ext)
      final_count_images[data_set_type]+=1
      exec_data_augmentation(same_path, im_ext ,transforms.ToPILImage()(old_image),data_set_type,rep_vector=[0,0,0])
  if data_set_type in ["val_fgr"]: # for neither downsampling nor upsampling
    if randint(0, 0)==0: # to keep one copy of original images
      transforms.ToPILImage()(old_image).save(same_path+im_ext)
      final_count_images[data_set_type]+=1
      exec_data_augmentation(same_path, im_ext ,transforms.ToPILImage()(old_image),data_set_type,rep_vector=[0,0,0])
  if data_set_type in ["val_htn"]: # for neither downsampling nor upsampling
    if randint(0, 0)==0: # to keep one copy of original images
      transforms.ToPILImage()(old_image).save(same_path+im_ext)
      final_count_images[data_set_type]+=1
      exec_data_augmentation(same_path, im_ext ,transforms.ToPILImage()(old_image),data_set_type,rep_vector=[0,0,0])
```


```
if entry_gate:
  !pip install SimpleITK
  import SimpleITK as sitk
  count_image={'control':0,'fgr':0,'htn':0}
  final_count_images={'train_control':0,'train_fgr':0,'train_htn':0,'val_control':0,'val_fgr':0,'val_htn':0}
  os.chdir(new_images_path)
  print("selected_cases: ",selected_cases)
  print("selected_trimester: ",selected_trimester)
  # reading control images path
  selected_path=get_trimester_wise_folder_list(c_path_control,selected_trimester)
  image_path_with_control=[]
  for path in selected_path:
    path = os.path.join(path)
    for path, subdirs, files in os.walk(path):
      for name in files:
        image_path_with_control.append(os.path.join(path, name))
  # reading control path images
  image_data_with_control=[]
  image_name_with_control=[]
  for i in range(len(image_path_with_control)):
    if(image_path_with_control[i].endswith("JPG")):
      sitk_t1 = sitk.ReadImage(image_path_with_control[i])
      sitk_t1=sitk.GetArrayFromImage(sitk_t1)
      if(sitk_t1.shape[0]<check_image or sitk_t1.shape[1]<check_image):
        count_image['control']+=1
        sitk_t2=sitk_t1
        print("control ",sitk_t2.shape)
        image_data_with_control.append(sitk_t2)
        image_name_with_control.append(image_path_with_control[i])

  # reading fgr images path
  selected_path=get_trimester_wise_folder_list(c_path_fgr,selected_trimester)
  image_path_with_fgr=[]
  for path in selected_path:
    path = os.path.join(path)
    for path, subdirs, files in os.walk(path):
      for name in files:
        image_path_with_fgr.append(os.path.join(path, name))
  # reading fgr path images
  image_data_with_fgr=[]
  image_name_with_fgr=[]
  for i in range(len(image_path_with_fgr)):
    if(image_path_with_fgr[i].endswith("JPG")):
      sitk_t1 = sitk.ReadImage(image_path_with_fgr[i])
      sitk_t1=sitk.GetArrayFromImage(sitk_t1)
      if(sitk_t1.shape[0]<check_image or sitk_t1.shape[1]<check_image):
        count_image['fgr']+=1
        sitk_t2=sitk_t1
        print("fgr ",sitk_t2.shape)
        image_data_with_fgr.append(sitk_t2)
        image_name_with_fgr.append(image_path_with_fgr[i])

  # reading htn images path
  selected_path=get_trimester_wise_folder_list(c_path_htn,selected_trimester)
  image_path_with_htn=[]
  for path in selected_path:
    path = os.path.join(path)
    for path, subdirs, files in os.walk(path):
      for name in files:
        image_path_with_htn.append(os.path.join(path, name))
  # reading control path images
  image_data_with_htn=[]
  image_name_with_htn=[]
  for i in range(len(image_path_with_htn)):
    if(image_path_with_htn[i].endswith("JPG")):
      sitk_t1 = sitk.ReadImage(image_path_with_htn[i])
      sitk_t1=sitk.GetArrayFromImage(sitk_t1)
      if(sitk_t1.shape[0]<check_image or sitk_t1.shape[1]<check_image):
        count_image['htn']+=1
        sitk_t2=sitk_t1
        print("htn ",sitk_t2.shape)
        image_data_with_htn.append(sitk_t2)
        image_name_with_htn.append(image_path_with_htn[i])

  print("control_len: ",len(image_data_with_control))
  print("fgr_len: ",len(image_data_with_fgr))
  print("htn_len: ",len(image_data_with_htn))
  print("Name control_len: ",len(image_name_with_control))
  print("Name fgr_len: ",len(image_name_with_fgr))
  print("Name htn_len: ",len(image_name_with_htn))

  Image_data=[]
  Image_data.extend(image_data_with_control)
  Image_data.extend(image_data_with_fgr)
  Image_data.extend(image_data_with_htn)
  Image_name=[]
  Image_name.extend(image_name_with_control)
  Image_name.extend(image_name_with_fgr)
  Image_name.extend(image_name_with_htn)
  Image_name=[n[98:-4].replace('/','_').replace(" ","_") for n in Image_name]
  # Image_label = np.array(["control", "fgr","htn"])
  Image_label = np.array([0,1,2])
  Image_label = np.repeat(Image_label, [len(image_data_with_control), len(image_data_with_fgr),len(image_data_with_htn)], axis=0)
  print("Image_data_len: ",len(Image_data))
  print("Image_data_label_len: ",len(Image_label))
  print("Image_name_label_len: ",len(Image_name))
  print("Image names list are here",Image_name)
  # time.sleep(10000)




  """Name_train, Name_test, X_train, X_test, Y_train, Y_test = train_test_split(Image_name, Image_data, Image_label, test_size=test_size, random_state=42, stratify=Image_label)
  Name_train_1, Name_test_1, X_train_1, X_test_1, Y_train_1, Y_test_1=Name_train, Name_test, X_train, X_test, Y_train, Y_test"""
  Image_name, Image_data, Image_label = shuffle(Image_name, Image_data, Image_label, random_state=0)
  kf = KFold(n_splits=k_fold)
  kk=0
  for train_index, test_index in kf.split(Image_data):
      os.chdir(new_images_path)
      kk=kk+1
      time.sleep(100)
      # print("TRAIN:", train_index, "TEST:", test_index)
      """X_train, X_test = Image_data[train_index], Image_data[test_index]
      Y_train, Y_test = Image_label[train_index], Image_label[test_index]
      Name_train, Name_test = Image_name[train_index], Image_name[test_index]"""
      X_train=[Image_data[i] for i in train_index]
      X_test=[Image_data[i] for i in test_index]
      Y_train=[Image_label[i] for i in train_index]
      Y_test=[Image_label[i] for i in test_index]
      Name_train=[Image_name[i] for i in train_index]
      Name_test=[Image_name[i] for i in test_index]
      
      Name_train_1, Name_test_1, X_train_1, X_test_1, Y_train_1, Y_test_1=Name_train, Name_test, X_train, X_test, Y_train, Y_test
      print("Train: ",len(X_train_1))
      print("Test: ",len(X_test_1))
      print("but its total of all cases and "+selected_trimester+" selected_trimester")
      print("Now in next steps it will divide further in selected_cases ",selected_cases)

      # saving images in corresponding folders
      if selected_cases=="CFH":
        if os.path.exists(new_images_path+"/train_"+str(kk)):
          shutil.rmtree("train_"+str(kk))
        if os.path.exists(new_images_path+"/val_"+str(kk)):
          shutil.rmtree("val_"+str(kk))
        os.mkdir("train_"+str(kk))
        os.mkdir("val_"+str(kk))
        os.chdir(new_images_path+"/train_"+str(kk))
        os.mkdir("control")
        os.mkdir("fgr")
        os.mkdir("htn")
        os.chdir(new_images_path+"/val_"+str(kk))
        os.mkdir("control")
        os.mkdir("fgr")
        os.mkdir("htn")
        for i in range(len(Y_train_1)):
          if Y_train_1[i]==0:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/control/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_control")
          if Y_train_1[i]==1:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/fgr/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_fgr")
          if Y_train_1[i]==2:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/htn/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_htn")
        for i in range(len(Y_test_1)):
          if Y_test_1[i]==0:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/control/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_control") 
          if Y_test_1[i]==1:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/fgr/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_fgr")
          if Y_test_1[i]==2:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/htn/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_htn")
      if selected_cases=="CH":
        if os.path.exists(new_images_path+"/train_"+str(kk)):
          shutil.rmtree("train_"+str(kk))
        if os.path.exists(new_images_path+"/val_"+str(kk)):
          shutil.rmtree("val_"+str(kk))
        os.mkdir("train_"+str(kk))
        os.mkdir("val_"+str(kk))
        os.chdir(new_images_path+"/train_"+str(kk))
        os.mkdir("control")
        os.mkdir("htn")
        os.chdir(new_images_path+"/val_"+str(kk))
        os.mkdir("control")
        os.mkdir("htn")
        for i in range(len(Y_train_1)):
          if Y_train_1[i]==0:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/control/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_control")
            # print("just for testing",X_train_1[i])
          if Y_train_1[i]==2:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/htn/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_htn")
        for i in range(len(Y_test_1)):
          if Y_test_1[i]==0:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/control/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_control") 
          if Y_test_1[i]==2:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/htn/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_htn")
      if selected_cases=="CF":
        if os.path.exists(new_images_path+"/train_"+str(kk)):
          shutil.rmtree("train_"+str(kk))
        if os.path.exists(new_images_path+"/val_"+str(kk)):
          shutil.rmtree("val_"+str(kk))
        os.mkdir("train_"+str(kk))
        os.mkdir("val_"+str(kk))
        os.chdir(new_images_path+"/train_"+str(kk))
        os.mkdir("control")
        os.mkdir("fgr")
        os.chdir(new_images_path+"/val_"+str(kk))
        os.mkdir("control")
        os.mkdir("fgr")
        for i in range(len(Y_train_1)):
          if Y_train_1[i]==0:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/control/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_control")
          if Y_train_1[i]==1:
            image_augmentation(same_path=new_images_path+"/train_"+str(kk)+"/fgr/image_"+str(Name_train_1[i]), im_ext=".jpg",old_image=X_train_1[i],data_set_type="train_fgr")
        for i in range(len(Y_test_1)):
          if Y_test_1[i]==0:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/control/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_control") 
          if Y_test_1[i]==1:
            image_augmentation(same_path=new_images_path+"/val_"+str(kk)+"/fgr/image_"+str(Name_test_1[i]), im_ext=".jpg",old_image=X_test_1[i],data_set_type="val_fgr")
      print("sample sizes before augmentation: ",count_image)
      print("sample_sizes after augmentation: ",final_count_images)
```


```
"""data_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=new_image_size_1, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=1),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=new_image_size_2),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=new_image_size_1),
        transforms.CenterCrop(size=new_image_size_2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}"""
```




    "data_transforms = {\n    # Train uses data augmentation\n    'train':\n    transforms.Compose([\n        transforms.RandomResizedCrop(size=new_image_size_1, scale=(0.8, 1.0)),\n        transforms.RandomRotation(degrees=1),\n        transforms.ColorJitter(),\n        transforms.RandomHorizontalFlip(),\n        transforms.CenterCrop(size=new_image_size_2),  # Image net standards\n        transforms.ToTensor(),\n        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards\n    ]),\n    # Validation does not use augmentation\n    'val':\n    transforms.Compose([\n        transforms.Resize(size=new_image_size_1),\n        transforms.CenterCrop(size=new_image_size_2),\n        transforms.ToTensor(),\n        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n    ]),\n}"




```
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
```


```
def train_model(model, criterion, optimizer, scheduler, num_epochs, PATH, models_name,kk):
    val_loss=[]
    val_acc=[]
    train_loss=[]
    train_acc=[]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_'+str(kk), 'val_'+str(kk)]:
            if phase == 'train_'+str(kk):
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train_'+str(kk)):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train_'+str(kk):
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train_'+str(kk):
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if(phase=="train_"+str(kk)):
              train_loss.append(epoch_loss)
              train_acc.append(epoch_acc.cpu().numpy())
            if(phase=="val_"+str(kk)):
              val_loss.append(epoch_loss)
              val_acc.append(epoch_acc.cpu().numpy())
            # deep copy the model
            if check=="best_acc":
              if phase == 'val_'+str(kk) and epoch_acc > best_acc:
                  best_acc = epoch_acc
                  best_model_wts = copy.deepcopy(model.state_dict())
            if check=="best_loss":
              if phase == 'val_'+str(kk) and epoch_loss < best_loss:
                  best_loss = epoch_loss
                  best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if check=="best_acc":
      print('Best val Acc: {:4f}'.format(best_acc))
    if check=="best_loss":
      print('Best val Loss: {:4f}'.format(best_loss))

    # saving acc and loss for each epoch
    models_result_plot[models_name+'_train_loss']=train_loss
    models_result_plot[models_name+'_train_acc']=train_acc
    models_result_plot[models_name+'_val_loss']=val_loss
    models_result_plot[models_name+'_val_acc']=val_acc
    # load best model weights
    model.load_state_dict(best_model_wts)
    """torch.save(model, PATH)
    model=torch.load(PATH)"""
    torch.save(model.state_dict(), PATH)
    # model.load_state_dict(torch.load(PATH))
    return model
```


```
def visualize_model(model, models_name, num_images,kk):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        tl=[]
        pl=[]
        for i, (inputs, labels) in enumerate(dataloaders['val_'+str(kk)]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # print(outputs)
            _, preds = torch.max(outputs, 1)
            tl.extend(labels.cpu().numpy())
            pl.extend(preds.cpu().numpy())
        # print(pl)
        # print(tl)
        models_result[models_name+'_truth']=tl
        models_result[models_name+'_pred']=pl
        models_score[models_name]=[cohen_kappa_score(tl,pl),
                                   accuracy_score(tl,pl),
                                   balanced_accuracy_score(tl,pl),
                                   f1_score(tl,pl,average='weighted'),
                                   precision_score(tl,pl,average='weighted'),
                                   recall_score(tl,pl,average='weighted'),
                                   jaccard_score(tl,pl,average='weighted')]
        for i, (inputs, labels) in enumerate(dataloaders['val_'+str(kk)]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                print(preds[j])
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```


```
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return model
```


```
for  kk in range(k_fold):
    kk=kk+1
    models_result=pd.DataFrame()
    models_result_plot=pd.DataFrame()
    models_score=pd.DataFrame(index=["cohen_kappa_score",
                                 "accuracy_score",
                                 "balanced_accuracy_score",
                                 "f1_score",
                                 "precision_score",
                                 "recall_score",
                                 "jaccard_score"])
    print(kk)
    if os.path.exists(PATH+str(kk)+"_models_score"+"_"+other+".csv"):
      models_score=pd.read_csv(PATH+str(kk)+"_models_score"+"_"+other+".csv")
      print(models_score.columns)
    if os.path.exists(PATH+str(kk)+"_models_result"+"_"+other+".csv"):
      models_result=pd.read_csv(PATH+str(kk)+"_models_result"+"_"+other+".csv")
    if os.path.exists(PATH+str(kk)+"_models_result_plot"+"_"+other+".csv"):
      models_result_plot=pd.read_csv(PATH+str(kk)+"_models_result_plot"+"_"+other+".csv")
    data_transforms = {
        # Train uses data augmentation
        'train_'+str(kk):
        transforms.Compose([
            transforms.Resize(size=new_image_size_1),
            transforms.CenterCrop(size=new_image_size_2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'val_'+str(kk):
        transforms.Compose([
            transforms.Resize(size=new_image_size_1),
            transforms.CenterCrop(size=new_image_size_2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(new_images_path, x),
                                              data_transforms[x])
                      for x in ['train_'+str(kk), 'val_'+str(kk)]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                  for x in ['train_'+str(kk), 'val_'+str(kk)]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train_'+str(kk), 'val_'+str(kk)]}
    class_names = image_datasets['train_'+str(kk)].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train_'+str(kk)]))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


    # For running all models
    if not os.path.exists(PATH+"models"):
      os.chdir(PATH)
      os.mkdir("models")
    for i in models_list:
      print(i)
      if i not in models_score.columns:
        if(i=="resnet18"):
          model_ft = models.resnet18(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="resnet34"):
          model_ft = models.resnet34(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="resnet50"):
          model_ft = models.resnet50(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="resnet101"):
          model_ft = models.resnet101(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="resnet152"):
          model_ft = models.resnet152(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="vgg11"):
          model_ft = models.vgg11(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg13"):
          model_ft = models.vgg13(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg16"):
          model_ft = models.vgg16(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg19"):
          model_ft = models.vgg19(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg11_bn"):
          model_ft = models.vgg11_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg13_bn"):
          model_ft = models.vgg13_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg16_bn"):
          model_ft = models.vgg16_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="vgg19_bn"):
          model_ft = models.vgg19_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
        if(i=="densenet121"):
          model_ft = models.densenet121(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="densenet161"):
          model_ft = models.densenet161(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="densenet169"):
          model_ft = models.densenet169(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="densenet201"):
          model_ft = models.densenet201(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="wide_resnet50_2"):
          model_ft = models.wide_resnet50_2(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="wide_resnet101_2"):
          model_ft = models.wide_resnet101_2(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="resnext50_32x4d"):
          model_ft = models.resnext50_32x4d(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="resnext101_32x8d"):
          model_ft = models.resnext101_32x8d(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="squeezenet1_1"): 
          model_ft = models.squeezenet1_1(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          in_ftrs = model_ft.classifier[1].in_channels
          out_ftrs = model_ft.classifier[1].out_channels
          features = list(model_ft.classifier.children())
          features[1] = nn.Conv2d(in_ftrs, len(class_names), kernel_size=(3,3),stride=1)
          features[3] = nn.AvgPool2d(2, stride=1)
          model_ft.classifier = nn.Sequential(*features)
          model_ft.num_classes = len(class_names)
          pretrained=pretrained_1
        if(i=="squeezenet1_0"): 
          model_ft = models.squeezenet1_0(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          in_ftrs = model_ft.classifier[1].in_channels
          out_ftrs = model_ft.classifier[1].out_channels
          features = list(model_ft.classifier.children())
          features[1] = nn.Conv2d(in_ftrs, len(class_names), kernel_size=(3,3),stride=1)
          features[3] = nn.AvgPool2d(2, stride=1)
          model_ft.classifier = nn.Sequential(*features)
          model_ft.num_classes = len(class_names)
          pretrained=pretrained_1
        if(i=="googlenet"):
          model_ft = models.googlenet(pretrained=pretrained_2)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_2
        if(i=="alexnet"): 
          model_ft = models.alexnet(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          """layers_to_freeze = [model_ft.features[0],model_ft.features[3],model_ft.features[6],model_ft.features[8]]
          for layer in layers_to_freeze:
            for params in layer.parameters():
              params.requires_grad = False"""
          model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
          pretrained=pretrained_1
        if(i=="shufflenet_v2_x1_0"):
          model_ft = models.shufflenet_v2_x1_0(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="mobilenet_v2"):
          model_ft = models.mobilenet_v2(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[1].in_features
          model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="mnasnet1_0"):
          model_ft = models.mnasnet1_0(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[1].in_features
          model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
        if(i=="inception_v3"):
          model_ft = models.inception_v3(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          model_ft.aux_logits=False
          """num_ftrs = model_ft.AuxLogits.fc.in_features
          model_ft.AuxLogits.fc = nn.Linear(num_ftrs, len(class_names))"""
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1

        # for more then 2 classes in any model
        # model_ft = nn.Sequential(model_ft,nn.Softmax(1))

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=.9)

        # optimizer_ft = torch.optim.Adam(model_ft.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epochs,
                              PATH=PATH+"models/"+str(kk)+"_"+i+"_"+other, models_name=i,kk=kk)

        visualize_model(model_ft, models_name=i, num_images=num_images,kk=kk)

        # save the results immediately after execution for each model
        models_score.to_csv(PATH+str(kk)+"_models_score"+"_"+other+".csv")
        models_result.to_csv(PATH+str(kk)+"_models_result"+"_"+other+".csv")
        models_result_plot.to_csv(PATH+str(kk)+"_models_result_plot"+"_"+other+".csv")
        models_score.transpose()
        print("Done")
```

    1
    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
           'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 'resnet50', 'resnet152',
           'densenet161', 'vgg19', 'vgg19_bn', 'wide_resnet50_2',
           'wide_resnet101_2', 'resnext50_32x4d'],
          dtype='object')



![png](ultrasound_images_new_11_imp_files/ultrasound_images_new_11_imp_11_1.png)


    resnet50
    resnet152
    densenet161
    vgg19
    vgg19_bn
    wide_resnet50_2
    wide_resnet101_2
    resnext50_32x4d
    resnext101_32x8d


    Downloading: "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth" to /root/.cache/torch/hub/checkpoints/resnext101_32x8d-8ba56ff5.pth



    HBox(children=(FloatProgress(value=0.0, max=356082095.0), HTML(value='')))


    
    Epoch 0/99
    ----------
    train_1 Loss: 0.7026 Acc: 0.5449
    val_1 Loss: 0.7054 Acc: 0.4211
    Epoch 1/99
    ----------
    train_1 Loss: 0.6837 Acc: 0.5678
    val_1 Loss: 0.7543 Acc: 0.4079
    Epoch 2/99
    ----------
    train_1 Loss: 0.6692 Acc: 0.5874
    val_1 Loss: 0.6335 Acc: 0.7237
    Epoch 3/99
    ----------
    train_1 Loss: 0.6504 Acc: 0.6099
    val_1 Loss: 0.7502 Acc: 0.6447
    Epoch 4/99
    ----------
    train_1 Loss: 0.5977 Acc: 0.6711
    val_1 Loss: 0.6453 Acc: 0.6842
    Epoch 5/99
    ----------
    train_1 Loss: 0.4159 Acc: 0.8027
    val_1 Loss: 0.7772 Acc: 0.7105
    Epoch 6/99
    ----------
    train_1 Loss: 0.3297 Acc: 0.8472
    val_1 Loss: 0.7754 Acc: 0.7632
    Epoch 7/99
    ----------
    train_1 Loss: 0.2586 Acc: 0.8806
    val_1 Loss: 0.8553 Acc: 0.7237
    Epoch 8/99
    ----------
    train_1 Loss: 0.2049 Acc: 0.9060
    val_1 Loss: 0.9963 Acc: 0.7105
    Epoch 9/99
    ----------
    train_1 Loss: 0.1781 Acc: 0.9172
    val_1 Loss: 1.1418 Acc: 0.7237
    Epoch 10/99
    ----------
    train_1 Loss: 0.1543 Acc: 0.9297
    val_1 Loss: 0.9157 Acc: 0.7368
    Epoch 11/99
    ----------
    train_1 Loss: 0.1361 Acc: 0.9354
    val_1 Loss: 0.8488 Acc: 0.7368
    Epoch 12/99
    ----------
    train_1 Loss: 0.1271 Acc: 0.9415
    val_1 Loss: 0.9364 Acc: 0.7368
    Epoch 13/99
    ----------
    train_1 Loss: 0.1237 Acc: 0.9411
    val_1 Loss: 1.1382 Acc: 0.7105
    Epoch 14/99
    ----------
    train_1 Loss: 0.1188 Acc: 0.9455
    val_1 Loss: 0.8415 Acc: 0.7368
    Epoch 15/99
    ----------
    train_1 Loss: 0.1182 Acc: 0.9443
    val_1 Loss: 0.8351 Acc: 0.7632
    Epoch 16/99
    ----------
    train_1 Loss: 0.1106 Acc: 0.9467
    val_1 Loss: 0.9732 Acc: 0.7368
    Epoch 17/99
    ----------
    train_1 Loss: 0.1129 Acc: 0.9500
    val_1 Loss: 0.9643 Acc: 0.7500
    Epoch 18/99
    ----------
    train_1 Loss: 0.1126 Acc: 0.9459
    val_1 Loss: 0.9356 Acc: 0.7500
    Epoch 19/99
    ----------
    train_1 Loss: 0.1077 Acc: 0.9496
    val_1 Loss: 1.1016 Acc: 0.6974
    Epoch 20/99
    ----------
    train_1 Loss: 0.1110 Acc: 0.9453
    val_1 Loss: 1.0195 Acc: 0.7368
    Epoch 21/99
    ----------
    train_1 Loss: 0.1103 Acc: 0.9484
    val_1 Loss: 1.0232 Acc: 0.7237
    Epoch 22/99
    ----------
    train_1 Loss: 0.1145 Acc: 0.9447
    val_1 Loss: 0.9962 Acc: 0.7368
    Epoch 23/99
    ----------



```
for  kk in range(k_fold):
    kk=kk+1
    print(kk)
    models_result=pd.read_csv(PATH+str(kk)+"_models_result"+"_"+other+".csv")
    """[confusion_matrix(models_result[i+'_truth'],models_result[i+'_pred']) for i in models_list[0:1]]"""
    if selected_cases in ["CH"]:
      CM=pd.DataFrame(index=['Truth_controls_Pred_control','Truth_controls_Pred_hypertension','Truth_hypertension_Pred_control','Truth_hypertension_Pred_hypertension',
                          'Truth_controls','Truth_hypertension','Pred_controls','Pred_hypertension'])
    if selected_cases in ["CF"]:
      CM=pd.DataFrame(index=['Truth_controls_Pred_control','Truth_controls_Pred_fgr','Truth_fgr_Pred_control','Truth_fgr_Pred_fgr',
                          'Truth_controls','Truth_fgr','Pred_controls','Pred_fgr'])
    if selected_cases in ["CFH"]:
      CM=pd.DataFrame(index=['Truth_controls_Pred_control','Truth_controls_Pred_fgr','Truth_control_Pred_hypertension',
                            'Truth_fgr_Pred_control','Truth_fgr_Pred_fgr','Truth_fgr_Pred_hypertension',
                            'Truth_hypertension_Pred_control','Truth_hypertension_Pred_fgr','Truth_hypertension_Pred_hypertension',
                          'Truth_controls','Truth_fgr','Truth_hypertension','Pred_controls','Pred_fgr','Pred_hypertension'])

    for i in models_list:
      print(i)
      cm=confusion_matrix(models_result[i+'_truth'],models_result[i+'_pred'])
      if cm.shape[0]==2:
        CM[i]=np.append(cm.reshape(-1),[cm[0][0]+cm[0][1],cm[1][0]+cm[1][1],cm[0][0]+cm[1][0],cm[0][1]+cm[1][1]])
      if cm.shape[0]==3:
        CM[i]=np.append(cm.reshape(-1),[cm[0][0]+cm[0][1]+cm[0][2],cm[1][0]+cm[1][1]+cm[1][2],cm[2][0]+cm[2][1]+cm[2][2],
                                        cm[0][0]+cm[1][0]+cm[2][0],cm[0][0]+cm[1][1]+cm[2][1],cm[0][2]+cm[1][2]+cm[2][2]])
    CM.to_csv(PATH+str(kk)+"_models_confusion_matrix"+"_"+other+".csv")
    CM
```


```
import matplotlib.pyplot as plt
for  kk in range(k_fold):
    kk=kk+1
    print(kk)
    models_result_plot_1=pd.read_csv(PATH+str(kk)+"_models_result_plot"+"_"+other+".csv")
    if not os.path.exists(PATH+"figures_"+str(kk)):
        os.chdir(PATH)
        os.mkdir("figures_"+str(kk))
    if not os.path.exists(PATH+"models_"+str(kk)):
        os.chdir(PATH)
        os.mkdir("models_"+str(kk))
    for model in models_list:
        print(model)
        saved_model=model+"_"+other
        model_path_1=PATH+"models_"+str(kk)+"/"+saved_model
        model_path_2=PATH+"figures_"+str(kk)
        os.chdir(model_path_2)
        if not os.path.exists(saved_model):
            os.makedirs(saved_model)
        plt.plot(models_result_plot_1[model+"_train_loss"],label="train")
        plt.plot(models_result_plot_1[model+"_val_loss"],label="val")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('training and validating loss with each epoch')
        plt.legend()
        plt.savefig(model_path_2+"/"+saved_model+"/"+"loss_plot"+".png")
        plt.close()
        plt.plot(models_result_plot_1[model+"_train_acc"],label="train")
        plt.plot(models_result_plot_1[model+"_val_acc"],label="val")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('training and validating accuracy with each epoch')
        plt.legend()
        plt.savefig(model_path_2+"/"+saved_model+"/"+"acc_plot"+".png")
        plt.close()
```


```
for kk in range(k_fold):
  kk+=1
  print(kk)
  data_transforms = {
        # Train uses data augmentation
        'train_'+str(kk):
        transforms.Compose([
            transforms.Resize(size=new_image_size_1),
            transforms.CenterCrop(size=new_image_size_2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'val_'+str(kk):
        transforms.Compose([
            transforms.Resize(size=new_image_size_1),
            transforms.CenterCrop(size=new_image_size_2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
      }
  models_result_image=pd.DataFrame()
  for i in models_list:
      print(i)
      if(i=="resnet18"):
          model_ft = models.resnet18(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="resnet34"):
          model_ft = models.resnet34(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="resnet50"):
          model_ft = models.resnet50(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="resnet101"):
          model_ft = models.resnet101(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="resnet152"):
          model_ft = models.resnet152(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="vgg11"):
          model_ft = models.vgg11(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg13"):
          model_ft = models.vgg13(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg16"):
          model_ft = models.vgg16(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg19"):
          model_ft = models.vgg19(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg11_bn"):
          model_ft = models.vgg11_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg13_bn"):
          model_ft = models.vgg13_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg16_bn"):
          model_ft = models.vgg16_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="vgg19_bn"):
          model_ft = models.vgg19_bn(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[6].in_features
          features = list(model_ft.classifier.children())[:-1]
          features.extend([nn.Linear(num_ftrs, len(class_names))])
          model_ft.classifier = nn.Sequential(*features)
          pretrained=pretrained_1
      if(i=="densenet121"):
          model_ft = models.densenet121(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="densenet161"):
          model_ft = models.densenet161(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="densenet169"):
          model_ft = models.densenet169(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="densenet201"):
          model_ft = models.densenet201(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier.in_features
          model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="wide_resnet50_2"):
          model_ft = models.wide_resnet50_2(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="wide_resnet101_2"):
          model_ft = models.wide_resnet101_2(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="resnext50_32x4d"):
          model_ft = models.resnext50_32x4d(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="resnext101_32x8d"):
          model_ft = models.resnext101_32x8d(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="squeezenet1_1"): 
          model_ft = models.squeezenet1_1(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          in_ftrs = model_ft.classifier[1].in_channels
          out_ftrs = model_ft.classifier[1].out_channels
          features = list(model_ft.classifier.children())
          features[1] = nn.Conv2d(in_ftrs, len(class_names), kernel_size=(3,3),stride=1)
          features[3] = nn.AvgPool2d(2, stride=1)
          model_ft.classifier = nn.Sequential(*features)
          model_ft.num_classes = len(class_names)
          pretrained=pretrained_1
      if(i=="squeezenet1_0"): 
          model_ft = models.squeezenet1_0(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          in_ftrs = model_ft.classifier[1].in_channels
          out_ftrs = model_ft.classifier[1].out_channels
          features = list(model_ft.classifier.children())
          features[1] = nn.Conv2d(in_ftrs, len(class_names), kernel_size=(3,3),stride=1)
          features[3] = nn.AvgPool2d(2, stride=1)
          model_ft.classifier = nn.Sequential(*features)
          model_ft.num_classes = len(class_names)
          pretrained=pretrained_1
      if(i=="googlenet"):
          model_ft = models.googlenet(pretrained=pretrained_2)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_2
      if(i=="alexnet"): 
          model_ft = models.alexnet(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          """layers_to_freeze = [model_ft.features[0],model_ft.features[3],model_ft.features[6],model_ft.features[8]]
          for layer in layers_to_freeze:
            for params in layer.parameters():
              params.requires_grad = False"""
          model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
          pretrained=pretrained_1
      if(i=="shufflenet_v2_x1_0"):
          model_ft = models.shufflenet_v2_x1_0(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="mobilenet_v2"):
          model_ft = models.mobilenet_v2(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[1].in_features
          model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="mnasnet1_0"):
          model_ft = models.mnasnet1_0(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.classifier[1].in_features
          model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      if(i=="inception_v3"):
          model_ft = models.inception_v3(pretrained=pretrained_1)
          model_ft = set_parameter_requires_grad(model_ft, feature_extract)
          model_ft.aux_logits=False
          """num_ftrs = model_ft.AuxLogits.fc.in_features
          model_ft.AuxLogits.fc = nn.Linear(num_ftrs, len(class_names))"""
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, len(class_names))
          pretrained=pretrained_1
      model_ft.load_state_dict(torch.load("/content/drive/MyDrive/Deep_Learning_Altrasound_Images/results_11_CH_123/models/"+str(kk)+"_"+i+"_best_acc_imagenet_False_25_8_256_224_400_0.2_CH_123_0.001_0.1_5"))
      with torch.no_grad():
        tl=[]
        pl=[]
        im_n=[]
        for jj in ["control","htn"]:
          path=new_images_path+"val_"+str(kk)+"/"+jj+"/"
          images_list=os.listdir(path)
          for ii in images_list:
            im_n.append(ii)
            # print(ii)
            im=Image.open(path+ii)
            inputs=data_transforms['val_'+str(kk)](im)
            inputs = inputs.to(device)
            if len(inputs.shape)==3:
              # inputs = inputs.unsqueeze(0) 
              inputs=torch.reshape(inputs,(1,inputs.shape[0],inputs.shape[1],inputs.shape[2]))
            # print(inputs.shape)
            if jj=="control":
              tl.append("control")
            else:
              tl.append("htn")
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            if preds.cpu().numpy()==1:
              pl.append("htn")
            else:
              pl.append("control")
            # print(preds.cpu().numpy(),len(tl),len(pl),tl,pl)
      models_result_image[i+'_name']=im_n
      models_result_image[i+'_truth']=tl
      models_result_image[i+'_pred']=pl
  models_result_image.to_csv(PATH+str(kk)+"_models_result_image"+"_"+other+".csv")
      
```


```
m=[]
sm=[]
s=[]
for kk in range(k_fold):
    kk+=1
    # print(kk)
    temp=pd.read_csv(PATH+str(kk)+"_models_score"+"_"+other+".csv")
    for k in temp.columns:
        if k in models_list:
            m.extend([k]*4)
            sm.extend(["cohen_kappa","accuracy","F1","precision"])
            s.extend(temp[k].values[[0,1,3,4]])
    # print(temp)
df=pd.DataFrame()
df["model"]=m
df["scoring_method"]=sm
df["score"]=s
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>scoring_method</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wide_resnet50_2</td>
      <td>cohen_kappa</td>
      <td>0.319877</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wide_resnet50_2</td>
      <td>accuracy</td>
      <td>0.653061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>wide_resnet50_2</td>
      <td>F1</td>
      <td>0.652387</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wide_resnet50_2</td>
      <td>precision</td>
      <td>0.679003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wide_resnet101_2</td>
      <td>cohen_kappa</td>
      <td>0.406061</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>resnext101_32x8d</td>
      <td>precision</td>
      <td>0.702793</td>
    </tr>
    <tr>
      <th>96</th>
      <td>googlenet</td>
      <td>cohen_kappa</td>
      <td>0.368441</td>
    </tr>
    <tr>
      <th>97</th>
      <td>googlenet</td>
      <td>accuracy</td>
      <td>0.684932</td>
    </tr>
    <tr>
      <th>98</th>
      <td>googlenet</td>
      <td>F1</td>
      <td>0.684457</td>
    </tr>
    <tr>
      <th>99</th>
      <td>googlenet</td>
      <td>precision</td>
      <td>0.685014</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```
import seaborn as sns
import matplotlib.pyplot as plt
```


```
g = sns.catplot(x="model", y="score", hue="scoring_method", 
                data=df, kind="box", height=5, aspect=2);
g.savefig(PATH+"box_plot.pdf",bbox_inches ="tight", pad_inches = 1, transparent = True)
```


![boxplot](https://github.com/krishan57gupta/UPAHAR/blob/master/box_plot.png)



```

```


```

```


```

```
