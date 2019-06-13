# Importing libraries
import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image

# Function for loading and rescaling images
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

# Function for calculating probabilities
def get_predictions(model_path, in_folder, out_file):

    print('Starting image processing...')
   
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print('Model has been successfully loaded!')
   
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set up some variables
    preds = []
    filenames = []
    counter = 0
   
    # Loop through test images and get predictions for each one
    for subdir, dir, files in os.walk(in_folder):
        for file in files:
            try:
                input_data = load_image(os.path.join(subdir, file))
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Append results to the array
                preds.append(output_data[0])
                filenames.append(file)

                # Track progress
                percentage = round(counter/len(files)*100, 2)
                print('Progress:',percentage,'%', end='\r')
                counter += 1
            except:
                print('Unsupported format:', file)

    # Generate dataframe with filenames and iphone probabilities
    print('Image processing has been completed')
    out_df = pd.DataFrame()
    out_df['image_name'] = filenames
    out_df['iphone_probability'] = [pred[0] for pred in preds]

    # Export dataframe to CSV
    out_df.to_csv(out_file, index=False)

if __name__ == '__main__':
    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Parse arguments
    parser = argparse.ArgumentParser(description='iPhone detector')
    
    parser.add_argument('--model', type=str, default='model.tflite', help='path to the model')
    parser.add_argument('--input', type=str, default='test', help='path to the folder containing pictures')
    parser.add_argument('--output', type=str, default='predictions.csv', help='output file path')
    args = parser.parse_args()
    print('model=',args.model,' input_data=',args.input,' output_data=',args.output)

    # Execute the predict function
    get_predictions(args.model, args.input, args.output)
    print('Predictions file', args.output,'has been successfully saved!')