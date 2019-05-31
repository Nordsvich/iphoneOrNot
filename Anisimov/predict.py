#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
import pandas as pd
import argparse
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os


# In[29]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


# In[1]:


#create csv
def results(names,preds,out_file):
    print('Saving result-Start')
    final_out = pd.DataFrame()
    final_out['image_name'] = names
    final_out['iphone_probability'] = preds
    final_out.to_csv(out_file, index=False)
    print('Saving result-Done')


# In[31]:


def predict_proba(model,test):
    print('Model load-Start')
    #load the model
    model=load_checkpoint(model)
    print('Model load-Done')
    print('-------')
    print('Prediction process-Start')
    #Image convert
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #Define images loader
    image_datasets = ImageFolder(test,data_transforms)
    test_data_gen=torch.utils.data.DataLoader(image_datasets, batch_size=256,shuffle=False, num_workers=6)
    #Load images names
    names=[image_datasets.imgs[i][0].split('\\')[-1] for i in range(0,len(image_datasets.imgs))]
    #Predict probas
    preds=np.array([])
    for inputs, labels in test_data_gen:
        preds=np.append(preds,model(inputs).data.numpy()[:,1])
    print('Prediction process-Done')
    # save table
    results(names,preds,args.output)
    


# In[46]:


if __name__ == '__main__':
    # disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # parse arguments
    parser = argparse.ArgumentParser(description='Iphone detector')
    parser.add_argument('--model', type=str, default='model.hdf5', help='path to model')
    parser.add_argument('--input', type=str, default='test', help='Path to class folder directory')
    parser.add_argument('--output', type=str, default='predictions.csv', help='path to file with model output')
    args = parser.parse_args()
    # get predictions
    predict_proba(args.model,args.input)
    print()
    print('Done')

