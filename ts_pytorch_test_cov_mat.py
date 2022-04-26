import numpy as np
from matplotlib import pyplot as plt
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Net
import TestDataset

global scaling_factor
scaling_factor=1

def load_data(path):
    
    test_dataset  = TestDataset.TestDataset(path)

    return test_dataset


def test_data(net, device):
    test_size = len(test_loader)
    gamma = np.zeros(test_size)

    for batch_idx, samples in enumerate(test_loader):
        inputs, labels = samples[0].to(device), samples[1].to(device)
        inputs = scaling_factor*inputs.view(1, 1, 9, 9)
        
        outputs = net(inputs)
#        print(labels)        
#        print(outputs)
        
        gamma[batch_idx] = (outputs[0][0] - outputs[0][1]).detach().cpu().numpy()
#        print('gamma: ', gamma[batch_idx])       
    print('Finish testing')

    return gamma
    
def main():

    
    ## GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    net = Net.Net()
    net.to(device)
    
    net_path = './cifar_net_cov_mat_500_sigmoid.pth'
    net.load_state_dict(torch.load(net_path))
    
    gamma = test_data(net, device)


    SNR_list = [-10, -12, -13, -15, -18, -20, -6, 0]
    num_snr = len(SNR_list)    
    num_file = 50           #10
    num_set = 100*num_file
    s_cnt = 0
    n_cnt = 0

    ### Gamma ###
    # Split gamma to gamma_n & gamma_s
    gamma_n_ = gamma[0:num_snr*num_set]
    gamma_s_ = gamma[num_snr*num_set:len(gamma)]

    gamma_s = np.zeros((num_snr, num_set))
    gamma_n = np.zeros((num_snr, num_set))
    
    res_s = [gamma_s_[i*num_set: (i+1)*num_set] for i in range(num_snr)]
    res_n = [gamma_n_[i*num_set: (i+1)*num_set] for i in range(num_snr)]
    
    for i in range(num_snr):
        gamma_s[i,:] = res_s[i]
        gamma_n[i,:] = res_n[i]
    

    ### histogram ###
    num_bin = 500
    hist_n = np.zeros((num_snr,num_bin))
    bin_n = np.zeros((num_snr,num_bin+1))
    hist_s = np.zeros((num_snr,num_bin))
    bin_s = np.zeros((num_snr,num_bin+1))


    for i in range(num_snr):
        hist_n[i,:], bin_n[i,:] = np.histogram(gamma_n[i,:],bins=num_bin)
        hist_s[i,:], bin_s[i,:] = np.histogram(gamma_s[i,:],bins=num_bin)
#        print('hist_n[' + str(i) + '] :', hist_n[i,:])

    for i in range(num_snr):
        plt.figure(i+1)
        line1, = plt.plot(bin_n[i,:][1:num_bin+1],hist_n[i,:])
        line2, = plt.plot(bin_s[i,:][1:num_bin+1], hist_s[i,:])
        plt.legend(handles=(line2, line1),labels=('signal','noise'))
        plt.title('histogram of Gamma (SNR = '+ str(SNR_list[i])+ 'dB)')
#        plt.title('histogram of Gamma (Noise)')
        plt.ylabel('counts')
        plt.xlabel('bin value')
        plt.ylim(0, num_set+200)


    ### Find the threshold where P_fa is 0.05
    th_offset = bin_n[0,1] - bin_n[0,0]
    p_fa_goal = 0.05
    p_fa = np.zeros(num_snr)
    p_d  = np.zeros(num_snr)
    th   = np.zeros(num_snr)
    

    for i in range(num_snr):
        th[i] = 0
        p_fa[i] = 0
        cnt=0
        flag_limit = 0
        while (p_fa_goal-5e-3) > p_fa[i] or p_fa[i] > (p_fa_goal+5e-3):
            cnt = cnt+1
            ## Get p_fa
            for j in range(num_bin):
                if th[i] < bin_n[i][j]:
                    if j == 0:
                        print('[THRESHOLD] wrong init')
                    else:
                        p_fa[i] = sum(hist_n[i][j:num_bin])/num_set
                        break
                else:
                    if j == num_bin-2:
                        print('[THRESHOLD] limit')
                        flag_limit = 1
                        break
                    
            if flag_limit == 1:
                break
             
            ## Get threshold (0.01 may be wrong)
            if p_fa[i] < p_fa_goal:
                th[i] = th[i] - th_offset
            else:
                th[i] = th[i] + th_offset

    for i in range(num_snr):
        print('th[',i,'] :', th[i])
        print('p_fa[',i,'] :', p_fa[i]) 

    
        
    
    ### Get p_d
    for i in range(num_snr):
        for j in range(num_bin):
            if th[i] < bin_s[i][j]:
                p_d[i] = sum(hist_s[i][j:num_bin])/num_set
                break
    
    ### SNR vs. p_d plot
    tmp = p_d
    p_d[0] = tmp[num_snr-1]
    p_d[1] = tmp[num_snr-2]
    p_d[2:num_snr] = tmp[0:num_snr-2]
    p_d = np.flip(p_d)

    t = np.arange(num_snr)
    plt.figure(10)
    plt.plot(t, p_d)
    plt.title('SNR vs. p_d')
    plt.xlabel('SNR [dB]')
    plt.xticks(np.arange(num_snr),['-20', '-18', '-15', '-13', '-12', '-10', '-6', '0'])
    plt.ylabel('p_d')
    plt.grid()

    plt.show()


if __name__ == '__main__':

    test_dataset = load_data('test_set')

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, shuffle=False)
    
    main()

