import numpy as np
from matplotlib import pyplot as plt
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Net
import TrainDataset

global scaling_factor
scaling_factor=1

def load_data(PATH):
    
    train_dataset = TrainDataset.TrainDataset(PATH)

    return train_dataset


def train_data(net, device, criterion, optimizer):
    
    training_loss = []
    epoch_loss = []
    num_epoch = 500

    print('Training...')

    for epoch in range(num_epoch):
        print('epoch #', epoch)

        for batch_idx, samples in enumerate(train_loader):
            inputs, labels = samples[0].to(device), samples[1].to(device)
    
            inputs = scaling_factor*inputs.view(1, 1, 9, 9)
#            print(inputs)
#            print(labels)

            #Reset the gradient info before calculating gradient
            optimizer.zero_grad()
            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            
            # Calculate gradient
            loss.backward()
            # Update weighting factors using the gradients?
            optimizer.step()
    
            running_loss = loss.item()
            training_loss.append(running_loss) 
        
        epoch_loss.append(np.mean(training_loss))
        training_loss = []

    print('Finish training')
    PATH = './cifar_net_cov_mat_'+ str(num_epoch) +'_sigmoid.pth'
    torch.save(net.state_dict(), PATH)

    return PATH, epoch_loss

def main():

    num_snr = 6

    ## GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    net = Net.Net()
    net.to(device)
     

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    PATH, epoch_loss_ = train_data(net, device, criterion, optimizer)
    net.load_state_dict(torch.load(PATH))
    
    plt.figure(1)
    step = np.arange(len(epoch_loss_))
    plt.plot(step, epoch_loss_)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()


if __name__ == '__main__':

    train_dataset = load_data('training_set')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=1, shuffle=True)
    
    main()

