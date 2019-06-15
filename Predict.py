
# coding: utf-8

# In[101]:


import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


# In[102]:


# creating function for calculating the results
def predict_proba(model_path, in_folder, out_file):

    print('Start image processing...')
    
# loading the model
model = load_model('second_try.h5')
batch_size=64


# In[103]:


# define ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    directory='final_dataset_2/test',
    target_size=(150, 150),
    shuffle=False,
    class_mode=None,
    batch_size=batch_size)


# In[104]:


# getting filenames
filenames = [filename.split('\\')[-1] for filename in
                 test_generator.filenames]


# In[105]:


# getting predictions
nb_samples = len(filenames)
test_generator.reset()
preds = model.predict_generator(test_generator,
                                steps=nb_samples / batch_size, verbose=1)


# In[94]:


# creating dataframe with filenames and iphone probabilities
def results(filenames, preds, out_file):
    output_data = pd.DataFrame()
    output_data['image_name'] = filenames
    output_data['iphone_probability'] = [pred[0] for pred in preds]
    
    # saving CSV with outputs
    output_data.to_csv(out_file, index=False)


# In[99]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iphone detector')
    parser.add_argument('--model', type=str, default='second_try.h5', help='path to model')
    parser.add_argument('--input', type=str, default='.final_dataset_2/test', help='path to folder with pictures')
    parser.add_argument('--output', type=str, default='predictions.csv', help='path to file with model output')

    args = parser.parse_args(args=[])
    print("model= {0} input_data= {1} output_data= {2}".format(args.model, args.input, args.output))


# In[107]:


predict_proba(args.model, args.input, args.output)
print()
print('Predictions file has been successfully created!')

