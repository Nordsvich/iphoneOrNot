import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iphone detector')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--input', type=str, help='path to folder with pictures')
    parser.add_argument('--output', type=str, help='path to file with model output')

    args = parser.parse_args()

    print("model= {0} input_data= {1} output_data= {2}".format(args.model, args.input, args.output))



