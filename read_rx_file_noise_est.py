import struct
import numpy as np
from matplotlib import pyplot as plt

# The number of data*(800 + label)
num_samp= 3480

rx_data = np.zeros(num_samp,dtype= np.complex64)
noise = np.zeros(num_samp, dtype=np.complex64)

def cp_detect(x):
    l = len(x)
    cp_corr = np.zeros(l-79)

    for i in range(l-79):
        temp = np.dot(x[i:16+i], np.conj(x[64+i:80+i]))
        cp_corr[i] = abs(temp)

    return cp_corr

def main():
    ############################ NOISE ##########################
    #1. Get noise_data from file
    with open("noise_data.out", "rb") as f:
        for i in range(num_samp-1):
            real = struct.unpack('f', f.read(4))[0]
            imag = struct.unpack('f', f.read(4))[0]
            noise[i] = real + 1j*imag
    
    noise_power = np.var(noise[100:900])
    print('noise power(800 point) is ', noise_power)

    #3. Calculate noise_reference (cp corr)
    noise_cp_corr = cp_detect(noise)
    noise_corr_len = len(noise_cp_corr)

    noise_ref = np.mean(noise_cp_corr[0:501])
    print('noise_ref is ', noise_ref)

    '''
    #plot noise cp corr 
    plt.figure(2)
    tm = np.arange(1, len(cp_corr)+1, 1)
    plt.plot(tm, cp_corr)
    plt.title('noise cp correlation')
    '''
    
    ############################## RX DATA ######################
    
    filename = 'rx_data.out'
    print('filename: ', filename)

    #3. Get rx_data from file
    with open(filename, "rb") as f:
        for i in range(num_samp):
            real = struct.unpack('f', f.read(4))[0]
            imag = struct.unpack('f', f.read(4))[0]
            rx_data[i] = real + 1j*imag
    

    t = np.arange(1,num_samp+1,1)
    '''
    rx_data_fig, axs = plt.subplots(2,1, constrained_layout=True)
    
    axs[0].plot(t, np.real(rx_data))
    axs[0].set_title('real(rx_data)')
#    axs[0].set_ylim([-0.002, 0.0025])

    axs[1].plot(t, np.imag(rx_data))
    axs[1].set_title('imag(rx_data)')
#    axs[1].set_ylim([-0.0028, 0.0025])
    '''
    
    #4. Calculate signal power (SNR)
    sig_power = np.var(rx_data)
    print('signal power is ', sig_power)

    #5. Calculate SNR
    SNR = 10*np.log10(sig_power/noise_power)
    print('SNR is ', SNR)

    
    #6. cyclic prefix detection
    cp_corr = cp_detect(rx_data)
    corr_len = len(cp_corr)

    
    
    #7. Accumulate cyclic prefix auto-correlation (N = 10)
    N = 80*10

    acc_corr = np.zeros(corr_len)

    for i in range(corr_len):
        if i < N-1:
            acc_corr[i] = sum(cp_corr[0:i+1])
        else:
            acc_corr[i] = sum(cp_corr[i-(N-1):i+1])

    gamma = acc_corr/noise_ref 

    non_gamma = np.zeros(len(gamma))
    j = 0

    for i in range(len(gamma)):
        if gamma[i] != 0:
            non_gamma[j] = gamma[i]
            j = j+1
    
    '''
    gamma_min = np.argmin(non_gamma[0:j])
    print('gamma size:', len(gamma))
    print('non_gamma size:', j)
    print('min value of gamma:', non_gamma[gamma_min])
    print('gamma')
    print(gamma)
    '''
    
    cp_start_idx = 2000 + np.argmax(cp_corr[2000:2080])

    cp_arr = rx_data[cp_start_idx:cp_start_idx+800]

    # calculate average power
    cp_arr = cp_arr - np.mean(cp_arr)
    average_p = np.var(cp_arr)
    
    temp=0.0
    for i in range(10):
        temp = temp + np.dot(cp_arr[80*i:80*(i+1)], np.conj(cp_arr[80*i:80*(i+1)]))

    average_p_ = np.abs(temp)/800
    print('average_p')
    print(average_p)

    print('average_p_')
    print(average_p_)


    temp = 0.0
    # calculate signal power 
    for i in range(10):
        temp = temp + np.dot(cp_arr[80*i:80*i+16], np.conj(cp_arr[80*(i+1)-16:80*(i+1)]))

    
    signal_p = np.abs(temp)/(16*10)
    print('signal_p')
    print(signal_p)
    
    noise_p = average_p - signal_p

    print('calculated noise power')
    print(noise_p)


    #plot cp corr
    tm = np.arange(0, corr_len, 1)

    plt.figure(1)
    plt.plot(tm, cp_corr)
    plt.title('cyclic prefix auto-correlation')
#    plt.ylim(0, 1.62e-5)
    plt.grid()
    '''
    plt.figure(2)
    plt.plot(tm, acc_corr)
    plt.title('acc corr')
#    plt.ylim(0, 0.0038)
    '''
    plt.figure(2)
    plt.plot(tm, gamma)
    plt.title('Gamma ('+ str(filename) + ')')
#    plt.ylim(0, 3200)
    plt.grid()

    '''
    ## No signal reference 
    if SNR > 14.0:
        no_signal_ref = 8.22e-4
    elif SNR > 13.0:
        no_signal_ref = 8.2e-4
    elif SNR > 9.5:
        no_signal_ref = 4e-4
    else:
        no_signal_ref = 2e-4
    
   
    no_signal_ref = 3.37e-6

    #7. Detect symbol index
    no_sig_idx = 0
    no_sig_start_idx = 0
    no_sig_end_idx = 0
    no_sig_flag = 0
    cnt = 0

    for i in range(corr_len):
        if (no_sig_flag == 1) and (cnt == 0):
            ##0. No signal interval is end
            no_sig_end_idx = i-2
            
            # For now, we assume there is one no signal interval
            # So we break
            break

        elif cnt >= 300:
    #        print('No signal detected')
            no_sig_flag = 1
            no_sig_start_idx = (i-1) -cnt
            
            if cp_corr[i] < no_signal_ref:
                if (no_sig_idx + 1) == i:
                    cnt = cnt + 1
                else:
                    cnt = 0
                no_sig_idx = i

            else:
                cnt = 0

        else:
            if cp_corr[i] < no_signal_ref:
                if (no_sig_idx + 1) == i:
                    cnt = cnt + 1
                else:
                    cnt = 0
                no_sig_idx = i

            else:
                cnt = 0
    
    if (no_sig_flag == 1) and (cnt == (corr_len-no_sig_start_idx-1)):
        no_sig_end_idx = corr_len-1


    print('corr len: ', corr_len)
    ## Now, you can develop the noise estimation
    print('no signal start idx : ', no_sig_start_idx)
    print('no signal end idx   : ', no_sig_end_idx)

    
    for i in range(no_sig_end_idx - no_sig_start_idx+1):
        print(cp_corr[no_sig_start_idx+i])
    
    
    # 8. Start detect symbol from no signal interval
    
    #Assume TX total symbol num is 27 (But, sometimes it false detects over 27)
    sym_num = 12
    sym_idx = np.zeros(sym_num)
        
    ## 1) Before no signal interval
    # I think we use 2) rather than 1). It is not real time...
    if SNR < 8.0:
        interval = 160 #Because, it always detects wrong the first symbol before no signal
    else:
        interval = 80

    if (no_sig_start_idx-interval) >= 0:
        idx = np.argmax(cp_corr[no_sig_start_idx-interval:no_sig_start_idx])
        i=0

        sym_idx[i]=(no_sig_start_idx -interval)+idx
        print(sym_idx[i])

        while (sym_idx[i]-80) >= 0:
            i = i+1
            sym_idx[i] = sym_idx[i-1]-80
            print(sym_idx[i])
    else:
        i=0
        print('We cannot sure there is a symbol')

    
    ## 2) After no signal interval
    
    if no_sig_end_idx+interval < corr_len:
        idx = np.argmax(cp_corr[no_sig_end_idx+1:no_sig_end_idx+interval])
        i=i+1
        sym_idx[i] = (no_sig_end_idx+1) + idx
        print(sym_idx[i])
        while (sym_idx[i]+80) < corr_len:
            i = i+1
            sym_idx[i] = sym_idx[i-1]+80
            print(sym_idx[i])

    elif no_sig_end_idx+1 == corr_len:
        print('There is no symbol after no signal interval')

    else: #no_sig_end_idx+40 exceeds the corr_len
        itv = corr_len-no_sig_end_idx
    
        idx = np.argmax(cp_corr[no_sig_end_idx+1:no_sig_end_idx+itv+1])
        i=0
        sym_idx[i] = (no_sig_end_idx+1) + idx

        while (sym_idx[i]+80) < corr_len:
            i = i+1
            sym_idx[i] = sym_idx[i-1]+80

    for i in range(sym_num):
        print('sym_idx[',i,']:', sym_idx[i])
        print('sym_idx[',i,']:', cp_corr[int(sym_idx[i])])

    
    ## 1) Get signal ratio (cp_corr/noise_ref)
    ratio_s = cp_corr/noise_ref

    
    ## 2) Get noise ratio (cp_noise_corr/noise_ref)
    ## This value would be about 1    
    ratio_n = noise_cp_corr/noise_ref
   
    ratio_s_sym=np.zeros(sym_num)
    
    for i in range(sym_num):
        ratio_s_sym[i]= ratio_s[int(sym_idx[i])]
        print('ratio_s_sym[',i,']:', ratio_s_sym[i])


    ## 3) Set threshold 
    if SNR < 8.0:
        detect_th = 7.0
    else:
        detect_th = 10.0
    
    new_ratio_s_sym = np.zeros(sym_num)

    for i in range(sym_num):
        new_ratio_s_sym[i] = ratio_s_sym[i]

    print('threshold: ', detect_th)
    
    
    ## 4) Print result
    print('no_signal_ref: ', no_signal_ref)
    print('')
    print('SNR: ', SNR)
    print('')
    print('noise_ref: ', noise_ref)
    print('')
    print('')
    print('min(ratio_s)', new_ratio_s_sym[np.argmin(new_ratio_s_sym)])
    print('max(ratio_s)', new_ratio_s_sym[np.argmax(new_ratio_s_sym)])
    print('mean(ratio_s)', np.mean(new_ratio_s_sym))    
    print('')
    print('')
    print('min(ratio_n)', ratio_n[np.argmin(ratio_n)])
    print('max(ratio_n)', ratio_n[np.argmax(ratio_n)])
    print('mean(ratio_n)', np.mean(ratio_n))
    print('')

    '''
    plt.show()
  

if __name__ == "__main__":
    main()
