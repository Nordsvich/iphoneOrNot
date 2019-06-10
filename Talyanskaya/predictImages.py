#!/usr/bin/env 
import argparse
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from keras import backend as K
K.set_image_dim_ordering('th')

def AP(y_true, y_pred):
    import tensorflow as tf
    from sklearn.metrics import average_precision_score
    return tf.py_func(average_precision_score, (y_true, y_pred), tf.double)
    
localPath = os.getenv("PWD")
#os.chdir('/home/talyanskaya_marina/finalFolder')
#localPath = os.getcwd()

def makeYourPredictions(inPath, modelPath, outPath):

    model = keras.models.load_model(modelPath, custom_objects={'AP': AP})
    
    finalTest_datagen = ImageDataGenerator(rescale=1./255)

    finalTest_generator = finalTest_datagen.flow_from_directory(inPath,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False)
    
    inFiles = finalTest_generator.filenames;
    
    finalPredictions = model.predict_generator(finalTest_generator, steps=len(inFiles)/32).flatten();
      
    out_df = pd.DataFrame()
    out_df['image_name'] = inFiles
    out_df['iphone_probability'] = 1 - finalPredictions
    out_df.to_csv(outPath + 'yourPredictions.csv', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Iphone detector')
    parser.add_argument('-i', type=str, default=localPath + '/in_folder')#)default='/home/talyanskaya_marina/data main/data/final test/test')
    parser.add_argument('-m', type=str, default=localPath + '/model.hdf5') #default='/home/talyanskaya_marina/savedModel-82/model-82.hdf5')
    parser.add_argument('-o', type=str, default=localPath + '/')#default='/home/talyanskaya_marina/YourPredictionsOutput')
    parser.add_argument('-f', type=str, default='nthg')
    args = parser.parse_args()

    makeYourPredictions(args.i, args.m, args.o)
    
    
    #If we want to write down results in .txt file:
    #try:
    #    os.mkdir(outPath)
    #except OSError:
    #    shutil.rmtree(outPath)
    #    os.mkdir(outPath)
    #finalPath = outPath + '/yourPredictions.txt';

    #f = open(finalPath,'w+');
    
    #for i in range(finalPredictions.shape[0]):
    #    f.write('Probability of ' + inFiles[i] + ' img is iphone is ' + str(round(finalPredictions[i], 2)) + '\n');
        
    #close(f)
        