# Developer: Waqas Alsubayee
# Date: 09/21/2018

import torch
import function as fu
import argparse


def main():
    parser = argparse.ArgumentParser(description='Predict category of flower')
    parser.add_argument('image_path', help='path of image to be analyzed')
    parser.add_argument('checkpoint_dir', help='directory containing /checkpoint.pth with pre-trained model to be used for prediction')
    parser.add_argument('--top_k', help='number of top K most likely classes', default=1, type=int)
    parser.add_argument('--category_names', help='Select JSON file')
    parser.add_argument('--gpu', help='Enable GPU', action='store_true')
    args = parser.parse_args()
    loaded_model, optimizer, criterion, epochs = fu.load_checkpoint(args.checkpoint_dir + '/checkpoint.pth')
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu') 
    print('Device is: ', device)
    probs, classes = fu.predict(args.image_path, loaded_model, args.top_k, device, args.category_names)
    print(probs)
    print(classes)

if __name__ == "__main__":
    main()