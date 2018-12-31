# Developer: Waqas Alsubayee
# Date: 09/21/2018

import torch
import function as fu
import train
import predict

original_model, test_loader, criterion = train.main()
loaded_model, optimizer, criterion, epochs = fu.load_checkpoint('/home/workspace/aipnd-project/checkpoints/checkpoint.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('Device is: ', device)

fu.compare_orig_vs_loaded(device, original_model, loaded_model, test_loader, criterion)