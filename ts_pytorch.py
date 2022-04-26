import numpy as np
from matplotlib import pyplot as plt
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Net
import TrainDataset
import TestDataset

global scaling_factor
scaling_factor=1

def load_data():
    
    train_dataset = TrainDataset.TrainDataset()
    test_dataset  = TestDataset.TestDataset()

    return train_dataset, test_dataset


def train_data(net, device, criterion, optimizer):
    
    loss_arr = []

    print('Training...')
    for epoch in range(2):

        running_loss = 0.0

        for batch_idx, samples in enumerate(train_loader):
                             #For CUDA                        
            inputs, labels = samples[0].to(device), samples[1].to(device)
    
            inputs = scaling_factor*inputs.view(1, 1, 40, 40)
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
    
            running_loss += loss.item()

            loss_arr.append(running_loss)
            running_loss = 0.0

    print('Finish training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    return PATH, loss_arr

def test_data(net, device, num_snr):
    test_size = len(test_loader)
    g = np.zeros(test_size)

    g_size = int(test_size/2/num_snr)


    gamma_s = np.zeros((num_snr, g_size))
    gamma_n = np.zeros((num_snr, g_size))

    idx_s = 0
    idx_n = 0

    idx_row_s = 0
    idx_row_n = 0


    for batch_idx, samples in enumerate(test_loader):
        inputs, labels = samples[0].to(device), samples[1].to(device)
        inputs = scaling_factor*inputs.view(1, 1, 40, 40)
        
        print(labels)
        outputs = net(inputs)
        print(outputs)

        g[batch_idx] = np.abs((outputs[0][0]/outputs[0][1]).detach().cpu().numpy())

        if batch_idx < int(test_size/2):
            gamma_s[idx_row_s][idx_s] = np.abs((outputs[0][0]/outputs[0][1]).detach().cpu().numpy())
            idx_s = idx_s +1

            if (batch_idx + 1) % 40 == 0:
                idx_row_s = int((batch_idx + 1)/40)
                idx_s = 0

        else:
            gamma_n[idx_row_n][idx_n] = np.abs((outputs[0][0]/outputs[0][1]).detach().cpu().numpy())
            idx_n = idx_n +1

            if (batch_idx + 1) % 40 == 0:
                idx_row_n = int((batch_idx + 1)/40) - num_snr
                idx_n = 0
            

    idx = np.arange(1, test_size+1, 1)  
    plt.figure(1)
    plt.plot(idx, g)
    plt.title('Gamma')
    plt.grid()

    plt.show()

    print('Finish testing')

    return gamma_s, gamma_n
    
def main():

    num_snr = 5

    ## GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    net = Net.Net()
    net.to(device)
     

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    PATH, loss_arr_ = train_data(net, device, criterion, optimizer)
    net.load_state_dict(torch.load(PATH))
    
    gamma_s, gamma_n =test_data(net, device, num_snr)

    plt.figure(1)
    step = np.arange(len(loss_arr_))
    plt.plot(step, loss_arr_)
    plt.xticks([1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001])
    plt.title('Loss')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')

    ### Gamma ###
    num_bin = 10

    g_mean = np.zeros(num_snr)
    hist_arr_n = np.zeros((num_snr,num_bin))
    bin_arr_n = np.zeros((num_snr,num_bin+1))
    hist_arr_s = np.zeros((num_snr,num_bin))
    bin_arr_s = np.zeros((num_snr,num_bin+1))

    print('')
    print('')

    for i in range(num_snr):
        g_mean[i] = np.mean(gamma_n[i,:])
        gamma_n[i,:] = gamma_n[i,:]/g_mean[i]
        gamma_s[i,:] = gamma_s[i,:]/g_mean[i]


    ### histogram ###
    for i in range(num_snr):
        hist_arr_n[i,:], bin_arr_n[i,:] = np.histogram(gamma_n[i,:],bins=num_bin)
        hist_arr_s[i,:], bin_arr_s[i,:] = np.histogram(gamma_s[i,:],bins=num_bin)

    plt.figure(2)
    line1, = plt.plot(bin_arr_n[0,:][1:11],hist_arr_n[0,:])
    line2, = plt.plot(bin_arr_s[0,:][1:11], hist_arr_s[0,:])
    plt.legend(handles=(line2, line1),labels=('signal','noise'))
    plt.title('histogram of Gamma (SNR = 30dB)')

    plt.figure(3)
    line1, = plt.plot(bin_arr_n[1,:][1:11],hist_arr_n[1,:])
    line2, = plt.plot(bin_arr_s[1,:][1:11], hist_arr_s[1,:])
    plt.legend(handles=(line2, line1),labels=('signal','noise'))
    plt.title('histogram of Gamma (SNR = 25dB)')


    plt.show()


if __name__ == '__main__':

    train_dataset, test_dataset = load_data()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False)
    
    classes = ('[1,0]','[0,1]')
    main()

