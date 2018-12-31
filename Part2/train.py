# Developer: Waqas Alsubayee
# Date: 09/21/2018

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import function as fu
import transformer as load_transform
import argparse
from torch.autograd import Variable
from PIL import Image

def main():
    allowed_models = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] 
    parser = argparse.ArgumentParser(description='Train NN')
    parser.add_argument('data_dir', help='directory containing sub-folders with data')
    parser.add_argument('--save_dir', help='directory for saving checkpoint', default='checkpoints')
    parser.add_argument('--arch', help='pre-trained model architecture', default='resnet18', choices=allowed_models)
    parser.add_argument('--learning_rate', help='learning rate during learning', type=float, default=0.01)
    parser.add_argument('--dropout', help='dropout during learning', type=float, default=0.05)
    parser.add_argument('--hidden_units', help='List of number of nodes in hidden layers', nargs='+', type=int, default=[256, 128])
    parser.add_argument('--epochs', help='Number of epochs for training', default=3, type=int)
    parser.add_argument('--gpu', help='Enable GPU', action='store_true')

    args = parser.parse_args()

    # Describe directories relative to working directory
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    save_dir = args.save_dir
    model_arch = args.arch
    model_hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    drop = args.dropout
    print('Data directory: ' + data_dir)
    print('hidden units: ' + str(args.hidden_units))
    print('Save directory: ' + save_dir)
    print('Architecture: ' + args.arch)
    fu.create_directory(save_dir)
    model = models.__getattribute__(model_arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = fu.Network(model.fc.in_features, 102, model_hidden_units, drop)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu') 
    print('device: ', device)

    epochs = args.epochs
    print_every = 50
    running_loss = 0
    steps = 0

    train_loader, test_loader, valid_loader, train_data, test_data, valid_data = load_transform.load_transform(data_dir, train_dir, valid_dir, test_dir)

    fu.train(device, model, epochs, criterion, optimizer, print_every, train_loader, test_loader, valid_loader)
    fu.save_checkpoint(model, model_arch, epochs, criterion, optimizer, train_data, save_dir)
    
    return model, test_loader, criterion

if __name__ == "__main__":
    main()