# import libraries
import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image


# create function for loading and rescaling images
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


# create function for calculating probabilities
def predict_proba(model_path, in_folder, out_file):

    print('Start image processing...')

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Now lets walk through our validation set and get predictions for each photo
    preds = []
    filenames = []
    count = 0

    for subdir, dirs, files in os.walk(in_folder):
        for file in files:
            input_data = load_image(os.path.join(subdir, file))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            preds.append(output_data[0])
            filenames.append(file)
            count += 1
            if count % 100 == 0:
                print('First', str(count), 'images done!')

    # create dataframe with filenames and iphone probabilities
    out_df = pd.DataFrame()
    out_df['image_name'] = filenames
    out_df['iphone_probability'] = [pred[0] for pred in preds]

    # save dataframe with answer
    out_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    # disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # parse arguments
    parser = argparse.ArgumentParser(description='Iphone detector')
    parser.add_argument('--model', type=str, default='converted_model.tflite', help='path to model')
    parser.add_argument('--input', type=str, default='test', help='path to folder with pictures')
    parser.add_argument('--output', type=str, default='predictions.csv', help='path to file with model output')
    args = parser.parse_args()
    print("model= {0} input_data= {1} output_data= {2}".format(args.model, args.input, args.output))

    # get predictions
    predict_proba(args.model, args.input, args.output)
    print()
    print('Predictions file is created successfully!')
