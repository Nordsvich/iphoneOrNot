# import libraries
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os


def predict_proba(model_path, in_folder, out_file):

    print('Start image processing...')

    # load model
    model = load_model(model_path)

    # batch size
    batch_size = 32

    # define ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            directory=in_folder,
            target_size=(224, 224),
            shuffle=False,
            class_mode=None,
            batch_size=batch_size)

    # get filenames
    filenames = [filename.split('\\')[-1] for filename in test_generator.filenames]

    # get predictions
    nb_samples = len(filenames)
    test_generator.reset()
    preds = model.predict_generator(test_generator, steps=nb_samples / batch_size, verbose=1)

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
    parser.add_argument('--model', type=str, default='model.hdf5', help='path to model')
    parser.add_argument('--input', type=str, default='test', help='path to folder with pictures')
    parser.add_argument('--output', type=str, default='predictions.csv', help='path to file with model output')
    args = parser.parse_args()
    print("model= {0} input_data= {1} output_data= {2}".format(args.model, args.input, args.output))

    # get predictions
    predict_proba(args.model, args.input, args.output)
    print()
    print('Predictions file is created successfully!')
