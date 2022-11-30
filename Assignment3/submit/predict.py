# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.



import os
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


##########################################
#### PREPROCESSING CODE ##################

#target dim
t_len=3
t_letters=24


letters={'ALPHA':0,'BETA':1,'CHI':2,'DELTA':3,'EPSILON':4,'ETA':5,'GAMMA':6,
         'IOTA':7,'KAPPA':8,'LAMDA':9,'MU':10,'NU':11,'OMEGA':12,'OMICRON':13,
         'PHI':14,'PI':15,'PSI':16,'RHO':17,'SIGMA':18,'TAU':19,'THETA':20,
         'UPSILON':21,'XI':22,'ZETA':23}


# background whitener
def bgwhitener(img):
    height,width=img.shape[:2]
    bgcolor=img[0][0].copy()
    for i in range(height):
        for j in range(width):
            if(img[i][j][0]==bgcolor[0] and img[i][j][1]==bgcolor[1] and img[i][j][2]==bgcolor[2]):
                img[i][j]=[255,255,255].copy()

                
# binarization of image (image still contains 3 characteres after this is done)
def preprocess1(filename):
    img=cv2.imread(filename)   # put in the path to training images in here plz
    height,width=img.shape[:2]
    # print(height,width)

    bgwhitener(img)

    #dilation
    kernel = np.ones((5, 5), np.uint8)
    dilate_img=cv2.dilate(img,kernel,iterations=1)

    binary=dilate_img.copy()

    # background black, others white
    for i in range(height):
        for j in range(width):
            if(dilate_img[i][j][0]!=255 or dilate_img[i][j][1]!=255 or dilate_img[i][j][2]!=255):
                binary[i][j]=[255,255,255].copy()
            else:
                binary[i][j]=[0,0,0].copy()

    return binary


# function to split the image into 3 parts containing individual letters 
def preprocess2(filenames):
    nsamples=len(filenames)
    fname = 0
    for i in range(nsamples):
        img=preprocess1(filenames[i])   # binarized image
        h = img.shape[0]
        w = img.shape[1]
        list = []
        stops = []
        for i in range(w):
            count = 0
            for j in range(h):
                if(img[j][i][0] >= 200):
                    count += 1
            if(count >= 10):
                stops.append(i)
        lis = []
        for i in range(len(stops)-1):
            if(stops[i+1]-stops[i] <= 10):
                lis.append(stops[i])
            else:
                if(len(lis)>=18):
                    list.append(lis)
                lis = []
        list.append(lis)
        img2 = img.copy()
        p1 = img2[:, max(0, list[0][0]-10):list[0][0]+110]
        p2 = img2[:, list[1][0]-10:list[1][0]+110]
        if(list[2][0]+90<499):
            p3 = img2[:, list[2][0]-10:min(499, list[2][0]+110)]
        else:
            p3 = img2[:, 379:499]

        
        cv2.imwrite("./ims1/"+str(fname)+".jpg", p1)
        fname += 1
        
        cv2.imwrite("./ims1/"+str(fname)+".jpg", p2)
        fname += 1
        
        cv2.imwrite("./ims1/"+str(fname)+".jpg", p3)
        fname += 1
        
#ims1 will contain all preprocessed images (3*no.of test cases images which contain individual letters in them)

        
############################ PREPROCESSING ENDS HERE ####################################
        
########################### PREDICTIONS CODE ############################################

#CUDA support if you are rich enough to afford a gpu like that
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model configurations 
config = dict(
	saved_path="random.pt",
	best_saved_path = "random_best.pt",
	lr=0.001, 
	EPOCHS = 3,
	BATCH_SIZE = 32,
	IMAGE_SIZE = 224,
	TRAIN_VALID_SPLIT = 0.2,
	device=device,
	SEED = 42,
	pin_memory=True,
	num_workers=2,
	USE_AMP = True,
	channels_last=False)


# more bt regarding cuda for poor ppl

# For custom operators, you might need to set python seed
random.seed(config['SEED'])
# If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG 
np.random.seed(config['SEED'])
# Prevent RNG for CPU and GPU using torch
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])

torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

data_transforms = {
	'test': transforms.Compose([
		transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

efficientnet = models.efficientnet_b0(pretrained = True)
efficientnet.classifier[1] = nn.Linear(in_features = 1280, out_features = 24, bias = True)
model = efficientnet
model.load_state_dict(torch.load('random_best.pt',map_location=torch.device('cpu')))
model = model.to(config['device'])

def pil_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')
        
req_dict = {0: 'ALPHA', 1: 'BETA', 2: 'CHI', 3: 'DELTA', 4: 'EPSILON', 5: 'ETA', 6: 'GAMMA', 7: 'IOTA', 8: 'KAPPA', 9: 'LAMDA', 10: 'MU', 11: 'NU', 12: 'OMEGA', 13: 'OMICRON', 14: 'PHI', 15: 'PI', 16: 'PSI', 17: 'RHO', 18: 'SIGMA', 19: 'TAU', 20: 'THETA', 21: 'UPSILON', 22: 'XI', 23: 'ZETA'}
def evalutate(model, path):
	model.eval()
	with torch.no_grad():
		image = pil_loader(path)
		x = data_transforms['test'](image)
		x = torch.Tensor(np.expand_dims(x,axis = 0))
		x = x.to(device)
		valid_logits = model(x)
		predict_prob = F.softmax(valid_logits)

		_,predictions = predict_prob.max(1)
		predictions = predictions.to('cpu')
		prediction = int(predictions[0])
		return req_dict[int(predictions[0])]


def decaptcha( filenames ):
	preprocess2(filenames)
	labels=[]
	ls=''
	cnt=1
	for i in range (3*len(filenames)):
		if(cnt%3==0):
			s=evalutate(model,'ims1/'+str(i)+'.jpg')
			ls+=s
			labels.append(ls)
			ls=''
		else:
			s=evalutate(model,'ims1/'+str(i)+'.jpg')
			ls=ls+s+','
		cnt+=1
	return labels
        

		