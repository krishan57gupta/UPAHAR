# UPAHAR
Ultrasound Placental image texture Analysis for prediction of Hypertension during pregnancy using Artificial intelligence Research

```
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



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
cwd = os.getcwd()
cwd

```




    '/content/drive/My Drive/Deep_Learning_Altrasound_Images/results_5_CF_23/figures'




```
# initialization
selected_cases="CH" # CH for control and htn
selected_trimester="123" # 123 for all trimesters
image_folder="saved_5_"+selected_cases+"_"+selected_trimester
results_folder="results_5_"+selected_cases+"_"+selected_trimester
c_path_control='/content/drive/My Drive/Deep_Learning_Altrasound_Images/placenta_folder_filtered/placenta  folder/CONTROLS copy'
c_path_htn='/content/drive/My Drive/Deep_Learning_Altrasound_Images/placenta_folder_filtered/placenta  folder/cases /htn copy'
PATH="/content/drive/My Drive/Deep_Learning_Altrasound_Images/"
if not os.path.exists(PATH+image_folder):
      os.chdir(PATH)
      os.mkdir(image_folder)
if not os.path.exists(PATH+results_folder):
      os.chdir(PATH)
      os.mkdir(results_folder)
new_images_path=PATH+image_folder+"/"
PATH=PATH+results_folder+"/"
new_image_size_1=256
new_image_size_2=224
check_image=400
test_size=.20
entry_gate=False # means dividing images in train and val as well as labels wise from other source
batch_size=8
num_images=2
num_epochs=25
models_result=pd.DataFrame()
models_result_plot=pd.DataFrame()
models_score=pd.DataFrame(index=["cohen_kappa_score",
                                 "accuracy_score",
                                 "balanced_accuracy_score",
                                 "f1_score",
                                 "precision_score",
                                 "recall_score",
                                 "jaccard_score"])
pretrained_1='imagenet'
pretrained_2='imagenet'
pretrained_3=True
feature_extract = False # it makes three cases when pretrianed False, pretrianed True with feature_extract True, pretrianed True with feature_extract False
check="best_acc" # two options, best acc or best_loss
other = str(check)+"_"+str(pretrained_1)+"_"+str(feature_extract)+\
                   "_"+str(num_epochs)+"_"+str(batch_size)+\
                   "_"+str(new_image_size_1)+"_"+str(new_image_size_2)+"_"+str(check_image)+\
                   "_"+str(test_size)+"_"+str(selected_cases)+"_"+str(selected_trimester)
models_list=["resnet18","resnet34","resnet50","resnet101","resnet152",
             "densenet121","densenet161","densenet169","densenet201",
             "vgg11","vgg13","vgg16","vgg19",
             "vgg11_bn","vgg13_bn","vgg16_bn","vgg19_bn",
             "wide_resnet50_2","wide_resnet101_2",
             "resnext50_32x4d","resnext101_32x8d",
             "squeezenet1_0","squeezenet1_1",
             "googlenet",
             "alexnet",
             "shufflenet_v2_x1_0",
             "mobilenet_v2",
             "mnasnet1_0",
             "inception_v3"]
if os.path.exists(PATH+"models_score"+"_"+other+".csv"):
  models_score=pd.read_csv(PATH+"models_score"+"_"+other+".csv")
  print(models_score.columns)
if os.path.exists(PATH+"models_result"+"_"+other+".csv"):
  models_result=pd.read_csv(PATH+"models_result"+"_"+other+".csv")
if os.path.exists(PATH+"models_result_plot"+"_"+other+".csv"):
  models_result_plot=pd.read_csv(PATH+"models_result_plot"+"_"+other+".csv")
```

    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg11',
           'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
           'vgg19_bn', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d',
           'resnext101_32x8d', 'squeezenet1_0', 'squeezenet1_1', 'googlenet',
           'alexnet', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'mnasnet1_0',
           'inception_v3'],
          dtype='object')



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
    sub2.extend([folder_path+"/"+i+"/"+k  for k in sub_path if k.find("second")!=-1])
    sub3.extend([folder_path+"/"+i+"/"+k for k in sub_path if k.find("third")!=-1])
    temp=sorted([int(k)  for k in sub_path if k.find("first")==-1 and k.find("second")==-1 and k.find("third")==-1])
    if(len(temp)>0):
      sub1.extend([folder_path+"/"+i+"/"+str(temp[0])])
    if(len(temp)>1):
      sub2.extend([folder_path+"/"+i+"/"+str(temp[1])])
    if(len(temp)>2):
      sub3.extend([folder_path+"/"+i+"/"+str(temp[2])])
  """print(sub1)
  print(sub2)
  print(sub3)"""
  if selected_trimester =="123":
    selected_path=sub1
    selected_path.extend(sub2)
    selected_path.extend(sub3)
  print("selected_path: ",selected_path)
  print("selected_path_len: ",len(selected_path))
  return (selected_path)
```


```
if entry_gate:
  !pip install SimpleITK
  import SimpleITK as sitk
  count_image={'control':0,'htn':0}
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
  for i in range(len(image_path_with_control)):
    if(image_path_with_control[i].endswith("JPG")):
      sitk_t1 = sitk.ReadImage(image_path_with_control[i])
      sitk_t1=sitk.GetArrayFromImage(sitk_t1)
      if(sitk_t1.shape[0]<check_image or sitk_t1.shape[1]<check_image):
        count_image['control']+=1
        sitk_t2=sitk_t1
        print("control ",sitk_t2.shape)
        image_data_with_control.append(sitk_t2)

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
  for i in range(len(image_path_with_htn)):
    if(image_path_with_htn[i].endswith("JPG")):
      sitk_t1 = sitk.ReadImage(image_path_with_htn[i])
      sitk_t1=sitk.GetArrayFromImage(sitk_t1)
      if(sitk_t1.shape[0]<check_image or sitk_t1.shape[1]<check_image):
        count_image['htn']+=1
        sitk_t2=sitk_t1
        print("htn ",sitk_t2.shape)
        image_data_with_htn.append(sitk_t2)

  print("control_len: ",len(image_data_with_control))
  print("htn_len: ",len(image_data_with_htn))

  Image_data=[]
  Image_data.extend(image_data_with_control)
  Image_data.extend(image_data_with_htn)
  # Image_label = np.array(["control","htn"])
  Image_label = np.array([0,1])
  Image_label = np.repeat(Image_label, [len(image_data_with_control),len(image_data_with_htn)], axis=0)
  print("Image_data_len: ",len(Image_data))
  print("Image_data_label_len: ",len(Image_label))

  X_train, X_test, Y_train, Y_test = train_test_split(Image_data, Image_label, test_size=test_size, random_state=42)
  X_train_1, X_test_1, Y_train_1, Y_test_1=X_train, X_test, Y_train, Y_test
  print("Trian: ",len(X_train_1))
  print("Test: ",len(X_test_1))

  # saving images in corresponding folders
  if selected_cases=="CH":
    if os.path.exists(new_images_path+"/train"):
      shutil.rmtree("train")
    if os.path.exists(new_images_path+"/val"):
      shutil.rmtree("val")
    os.mkdir("train")
    os.mkdir("val")
    os.chdir(new_images_path+"/train")
    os.mkdir("control")
    os.mkdir("htn")
    os.chdir(new_images_path+"/val")
    os.mkdir("control")
    os.mkdir("htn")
    for i in range(len(Y_train_1)):
      if Y_train_1[i]==0:
        cv2.imwrite(new_images_path+"/train/control/image"+str(i)+".jpg",X_train_1[i]) 
      if Y_train_1[i]==2:
        cv2.imwrite(new_images_path+"/train/htn/image"+str(i)+".jpg",X_train_1[i]) 
    for i in range(len(Y_test_1)):
      if Y_test_1[i]==0:
        cv2.imwrite(new_images_path+"/val/control/image"+str(i)+".jpg",X_test_1[i])  
      if Y_test_1[i]==2:
        cv2.imwrite(new_images_path+"/val/htn/image"+str(i)+".jpg",X_test_1[i])
  print("sample sizes before augmentation: ",count_image)
```


```
data_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=new_image_size_1, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=1),
        transforms.ColorJitter(),
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
}
```

```
    "data_transforms = {\n    # Train uses data augmentation\n    'train':\n    transforms.Compose([\n        transforms.Resize(size=new_image_size_1),\n        transforms.CenterCrop(size=new_image_size_2),\n        transforms.ToTensor(),\n        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards\n    ]),\n    # Validation does not use augmentation\n    'val':\n    transforms.Compose([\n        transforms.Resize(size=new_image_size_1),\n        transforms.CenterCrop(size=new_image_size_2),\n        transforms.ToTensor(),\n        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \n    ]),\n}"




```
image_datasets = {x: datasets.ImageFolder(os.path.join(new_images_path, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```


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


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

```


![png](ultrasound_images_new_5_imp_files/ultrasound_images_new_5_imp_8_0.png)



```
def train_model(model, criterion, optimizer, scheduler, num_epochs, PATH, models_name):
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
        for phase in ['train', 'val']:
            if phase == 'train':
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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if(phase=="train"):
              train_loss.append(epoch_loss)
              train_acc.append(epoch_acc.cpu().numpy())
            if(phase=="val"):
              val_loss.append(epoch_loss)
              val_acc.append(epoch_acc.cpu().numpy())
            # deep copy the model
            if check=="best_acc":
              if phase == 'val' and epoch_acc > best_acc:
                  best_acc = epoch_acc
                  best_model_wts = copy.deepcopy(model.state_dict())
            if check=="best_loss":
              if phase == 'val' and epoch_loss < best_loss:
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
    torch.save(model, PATH)
    # model=torch.load(PATH)
    return model
```


```
def visualize_model(model, models_name, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        tl=[]
        pl=[]
        for i, (inputs, labels) in enumerate(dataloaders['val']):
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
        for i, (inputs, labels) in enumerate(dataloaders['val']):
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

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epochs,
                          PATH=PATH+"models/"+i+"_"+other, models_name=i)

    visualize_model(model_ft, models_name=i, num_images=num_images)

    # save the results immediately after execution for each model
    models_score.to_csv(PATH+"models_score"+"_"+other+".csv")
    models_result.to_csv(PATH+"models_result"+"_"+other+".csv")
    models_result_plot.to_csv(PATH+"models_result_plot"+"_"+other+".csv")
```

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
    densenet121
    densenet161
    densenet169
    densenet201
    vgg11
    vgg13
    vgg16
    vgg19
    vgg11_bn
    vgg13_bn
    vgg16_bn
    vgg19_bn
    wide_resnet50_2
    wide_resnet101_2
    resnext50_32x4d
    resnext101_32x8d
    squeezenet1_0
    squeezenet1_1
    googlenet
    alexnet
    shufflenet_v2_x1_0
    mobilenet_v2
    mnasnet1_0
    inception_v3



```
"""models_score.to_csv(PATH+"models_score"+"_"+other+".csv")
models_result.to_csv(PATH+"models_result"+"_"+other+".csv")
models_result_plot.to_csv(PATH+"models_result_plot"+"_"+other+".csv")
models_score=pd.read_csv(PATH+"models_score"+"_"+other+".csv")
models_result=pd.read_csv(PATH+"models_result"+"_"+other+".csv")
models_result_plot=pd.read_csv(PATH+"models_result_plot"+"_"+other+".csv")"""
models_score.transpose()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Unnamed: 0.1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Unnamed: 0.1.1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Unnamed: 0.1.1.1</th>
      <td>cohen_kappa_score</td>
      <td>accuracy_score</td>
      <td>balanced_accuracy_score</td>
      <td>f1_score</td>
      <td>precision_score</td>
      <td>recall_score</td>
      <td>jaccard_score</td>
    </tr>
    <tr>
      <th>resnet18</th>
      <td>0.176647</td>
      <td>0.715909</td>
      <td>0.568966</td>
      <td>0.633132</td>
      <td>0.80046</td>
      <td>0.715909</td>
      <td>0.516369</td>
    </tr>
    <tr>
      <th>resnet34</th>
      <td>0.110778</td>
      <td>0.693182</td>
      <td>0.54325</td>
      <td>0.603783</td>
      <td>0.710092</td>
      <td>0.693182</td>
      <td>0.490441</td>
    </tr>
    <tr>
      <th>resnet50</th>
      <td>0</td>
      <td>0.670455</td>
      <td>0.5</td>
      <td>0.538188</td>
      <td>0.449509</td>
      <td>0.670455</td>
      <td>0.449509</td>
    </tr>
    <tr>
      <th>resnet101</th>
      <td>0.0449102</td>
      <td>0.670455</td>
      <td>0.517534</td>
      <td>0.574433</td>
      <td>0.619724</td>
      <td>0.670455</td>
      <td>0.465632</td>
    </tr>
    <tr>
      <th>resnet152</th>
      <td>0.112469</td>
      <td>0.625</td>
      <td>0.55377</td>
      <td>0.614933</td>
      <td>0.608724</td>
      <td>0.625</td>
      <td>0.463439</td>
    </tr>
    <tr>
      <th>densenet121</th>
      <td>0</td>
      <td>0.670455</td>
      <td>0.5</td>
      <td>0.538188</td>
      <td>0.449509</td>
      <td>0.670455</td>
      <td>0.449509</td>
    </tr>
    <tr>
      <th>densenet161</th>
      <td>0.123626</td>
      <td>0.670455</td>
      <td>0.552601</td>
      <td>0.622883</td>
      <td>0.632327</td>
      <td>0.670455</td>
      <td>0.489836</td>
    </tr>
    <tr>
      <th>densenet169</th>
      <td>0.0631868</td>
      <td>0.647727</td>
      <td>0.526885</td>
      <td>0.596875</td>
      <td>0.596043</td>
      <td>0.647727</td>
      <td>0.465814</td>
    </tr>
    <tr>
      <th>densenet201</th>
      <td>0.382817</td>
      <td>0.727273</td>
      <td>0.691409</td>
      <td>0.727273</td>
      <td>0.727273</td>
      <td>0.727273</td>
      <td>0.580463</td>
    </tr>
    <tr>
      <th>vgg11</th>
      <td>0.107893</td>
      <td>0.681818</td>
      <td>0.543542</td>
      <td>0.609596</td>
      <td>0.651836</td>
      <td>0.681818</td>
      <td>0.488163</td>
    </tr>
    <tr>
      <th>vgg13</th>
      <td>0.0658858</td>
      <td>0.670455</td>
      <td>0.5263</td>
      <td>0.589053</td>
      <td>0.622644</td>
      <td>0.670455</td>
      <td>0.472606</td>
    </tr>
    <tr>
      <th>vgg16</th>
      <td>0</td>
      <td>0.670455</td>
      <td>0.5</td>
      <td>0.538188</td>
      <td>0.449509</td>
      <td>0.670455</td>
      <td>0.449509</td>
    </tr>
    <tr>
      <th>vgg19</th>
      <td>0</td>
      <td>0.670455</td>
      <td>0.5</td>
      <td>0.538188</td>
      <td>0.449509</td>
      <td>0.670455</td>
      <td>0.449509</td>
    </tr>
    <tr>
      <th>vgg11_bn</th>
      <td>0.0820996</td>
      <td>0.647727</td>
      <td>0.535652</td>
      <td>0.606149</td>
      <td>0.603304</td>
      <td>0.647727</td>
      <td>0.47043</td>
    </tr>
    <tr>
      <th>vgg13_bn</th>
      <td>0.0567787</td>
      <td>0.579545</td>
      <td>0.528638</td>
      <td>0.581312</td>
      <td>0.583216</td>
      <td>0.579545</td>
      <td>0.423809</td>
    </tr>
    <tr>
      <th>vgg16_bn</th>
      <td>-0.0224632</td>
      <td>0.659091</td>
      <td>0.491525</td>
      <td>0.53269</td>
      <td>0.44697</td>
      <td>0.659091</td>
      <td>0.44189</td>
    </tr>
    <tr>
      <th>vgg19_bn</th>
      <td>0.0619587</td>
      <td>0.636364</td>
      <td>0.527177</td>
      <td>0.597796</td>
      <td>0.591034</td>
      <td>0.636364</td>
      <td>0.460847</td>
    </tr>
    <tr>
      <th>wide_resnet50_2</th>
      <td>0.0771129</td>
      <td>0.613636</td>
      <td>0.536528</td>
      <td>0.600694</td>
      <td>0.593113</td>
      <td>0.613636</td>
      <td>0.450879</td>
    </tr>
    <tr>
      <th>wide_resnet101_2</th>
      <td>-0.042191</td>
      <td>0.636364</td>
      <td>0.483343</td>
      <td>0.538751</td>
      <td>0.510186</td>
      <td>0.636364</td>
      <td>0.433837</td>
    </tr>
    <tr>
      <th>resnext50_32x4d</th>
      <td>0.195755</td>
      <td>0.647727</td>
      <td>0.597019</td>
      <td>0.646117</td>
      <td>0.64467</td>
      <td>0.647727</td>
      <td>0.490699</td>
    </tr>
    <tr>
      <th>resnext101_32x8d</th>
      <td>0.0786802</td>
      <td>0.625</td>
      <td>0.536236</td>
      <td>0.603849</td>
      <td>0.595221</td>
      <td>0.625</td>
      <td>0.458194</td>
    </tr>
    <tr>
      <th>squeezenet1_0</th>
      <td>0.045701</td>
      <td>0.681818</td>
      <td>0.517241</td>
      <td>0.563844</td>
      <td>0.784222</td>
      <td>0.681818</td>
      <td>0.46604</td>
    </tr>
    <tr>
      <th>squeezenet1_1</th>
      <td>0</td>
      <td>0.670455</td>
      <td>0.5</td>
      <td>0.538188</td>
      <td>0.449509</td>
      <td>0.670455</td>
      <td>0.449509</td>
    </tr>
    <tr>
      <th>googlenet</th>
      <td>0.0820996</td>
      <td>0.647727</td>
      <td>0.535652</td>
      <td>0.606149</td>
      <td>0.603304</td>
      <td>0.647727</td>
      <td>0.47043</td>
    </tr>
    <tr>
      <th>alexnet</th>
      <td>0.0229709</td>
      <td>0.670455</td>
      <td>0.508767</td>
      <td>0.557625</td>
      <td>0.61694</td>
      <td>0.670455</td>
      <td>0.457955</td>
    </tr>
    <tr>
      <th>shufflenet_v2_x1_0</th>
      <td>0</td>
      <td>0.670455</td>
      <td>0.5</td>
      <td>0.538188</td>
      <td>0.449509</td>
      <td>0.670455</td>
      <td>0.449509</td>
    </tr>
    <tr>
      <th>mobilenet_v2</th>
      <td>0.0673732</td>
      <td>0.681818</td>
      <td>0.526008</td>
      <td>0.581282</td>
      <td>0.677184</td>
      <td>0.681818</td>
      <td>0.474137</td>
    </tr>
    <tr>
      <th>mnasnet1_0</th>
      <td>0.00146413</td>
      <td>0.647727</td>
      <td>0.500584</td>
      <td>0.560712</td>
      <td>0.559544</td>
      <td>0.647727</td>
      <td>0.448752</td>
    </tr>
    <tr>
      <th>inception_v3</th>
      <td>-0.0209581</td>
      <td>0.647727</td>
      <td>0.491818</td>
      <td>0.545084</td>
      <td>0.529356</td>
      <td>0.647727</td>
      <td>0.441855</td>
    </tr>
  </tbody>
</table>
</div>




```
models_result=pd.read_csv(PATH+"models_result"+"_"+other+".csv")
"""[confusion_matrix(models_result[i+'_truth'],models_result[i+'_pred']) for i in models_list[0:1]]"""
if selected_cases in ["CH"]:
  CM=pd.DataFrame(index=['Truth_controls_Pred_control','Truth_controls_Pred_hypertension','Truth_hypertension_Pred_control','Truth_hypertension_Pred_hypertension',
                       'Truth_controls','Truth_hypertension','Pred_controls','Pred_hypertension'])

for i in models_list:
  print(i)
  cm=confusion_matrix(models_result[i+'_truth'],models_result[i+'_pred'])
  if cm.shape[0]==2:
    CM[i]=np.append(cm.reshape(-1),[cm[0][0]+cm[0][1],cm[1][0]+cm[1][1],cm[0][0]+cm[1][0],cm[0][1]+cm[1][1]])
  if cm.shape[0]==3:
    CM[i]=np.append(cm.reshape(-1),[cm[0][0]+cm[0][1]+cm[0][2],cm[1][0]+cm[1][1]+cm[1][2],cm[2][0]+cm[2][1]+cm[2][2],
                                    cm[0][0]+cm[1][0]+cm[2][0],cm[0][0]+cm[1][1]+cm[2][1],cm[0][2]+cm[1][2]+cm[2][2]])
CM.to_csv(PATH+"models_confusion_matrix"+"_"+other+".csv")
CM
```

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
    densenet121
    densenet161
    densenet169
    densenet201
    vgg11
    vgg13
    vgg16
    vgg19
    vgg11_bn
    vgg13_bn
    vgg16_bn
    vgg19_bn
    wide_resnet50_2
    wide_resnet101_2
    resnext50_32x4d
    resnext101_32x8d
    squeezenet1_0
    squeezenet1_1
    googlenet
    alexnet
    shufflenet_v2_x1_0
    mobilenet_v2
    mnasnet1_0
    inception_v3





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
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnet101</th>
      <th>resnet152</th>
      <th>densenet121</th>
      <th>densenet161</th>
      <th>densenet169</th>
      <th>densenet201</th>
      <th>vgg11</th>
      <th>vgg13</th>
      <th>vgg16</th>
      <th>vgg19</th>
      <th>vgg11_bn</th>
      <th>vgg13_bn</th>
      <th>vgg16_bn</th>
      <th>vgg19_bn</th>
      <th>wide_resnet50_2</th>
      <th>wide_resnet101_2</th>
      <th>resnext50_32x4d</th>
      <th>resnext101_32x8d</th>
      <th>squeezenet1_0</th>
      <th>squeezenet1_1</th>
      <th>googlenet</th>
      <th>alexnet</th>
      <th>shufflenet_v2_x1_0</th>
      <th>mobilenet_v2</th>
      <th>mnasnet1_0</th>
      <th>inception_v3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Truth_controls_Pred_control</th>
      <td>59</td>
      <td>58</td>
      <td>59</td>
      <td>57</td>
      <td>45</td>
      <td>59</td>
      <td>53</td>
      <td>52</td>
      <td>47</td>
      <td>56</td>
      <td>56</td>
      <td>59</td>
      <td>59</td>
      <td>51</td>
      <td>40</td>
      <td>58</td>
      <td>50</td>
      <td>45</td>
      <td>55</td>
      <td>44</td>
      <td>47</td>
      <td>59</td>
      <td>59</td>
      <td>51</td>
      <td>58</td>
      <td>59</td>
      <td>58</td>
      <td>55</td>
      <td>56</td>
    </tr>
    <tr>
      <th>Truth_controls_Pred_fgr</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>12</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>19</td>
      <td>1</td>
      <td>9</td>
      <td>14</td>
      <td>4</td>
      <td>15</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Truth_fgr_Pred_control</th>
      <td>25</td>
      <td>26</td>
      <td>29</td>
      <td>27</td>
      <td>19</td>
      <td>29</td>
      <td>23</td>
      <td>24</td>
      <td>12</td>
      <td>25</td>
      <td>26</td>
      <td>29</td>
      <td>29</td>
      <td>23</td>
      <td>18</td>
      <td>29</td>
      <td>23</td>
      <td>20</td>
      <td>28</td>
      <td>16</td>
      <td>21</td>
      <td>28</td>
      <td>29</td>
      <td>23</td>
      <td>28</td>
      <td>29</td>
      <td>27</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Truth_fgr_Pred_fgr</th>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>17</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>11</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>1</td>
      <td>13</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Truth_controls</th>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
      <td>59</td>
    </tr>
    <tr>
      <th>Truth_fgr</th>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Pred_controls</th>
      <td>84</td>
      <td>84</td>
      <td>88</td>
      <td>84</td>
      <td>64</td>
      <td>88</td>
      <td>76</td>
      <td>76</td>
      <td>59</td>
      <td>81</td>
      <td>82</td>
      <td>88</td>
      <td>88</td>
      <td>74</td>
      <td>58</td>
      <td>87</td>
      <td>73</td>
      <td>65</td>
      <td>83</td>
      <td>60</td>
      <td>68</td>
      <td>87</td>
      <td>88</td>
      <td>74</td>
      <td>86</td>
      <td>88</td>
      <td>85</td>
      <td>82</td>
      <td>84</td>
    </tr>
    <tr>
      <th>Pred_fgr</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>24</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
      <td>29</td>
      <td>7</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>30</td>
      <td>1</td>
      <td>15</td>
      <td>23</td>
      <td>5</td>
      <td>28</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>14</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
    </tr>
  </t
```

```

```

```
