import numpy as np
import time
import uhd
import threading
import math
import signal
import struct
from os.path import getsize
from matplotlib import pyplot as plt

stop_signal_called = False
repeat = True

tx_file = 'tx_filtered_data.bin'
rx_num_samp = 150000*2 #3300000
tx_num_samp = getsize(tx_file)//8

rate    = 40e6
freq    = 2.6e9
tx_gain = 60
rx_gain = 29
tx_bw   = 20e6
rx_bw   = 18e6

tx_lpf_cutoff = 11e6

txusrp=uhd.usrp.MultiUSRP("name=MyB200")
rxusrp=uhd.usrp.MultiUSRP("name=MyB200_1")


global recv_buffer
recv_buffer = np.zeros(rx_num_samp, dtype=np.complex64)


def write_file(write_data, file_name, d_type):
    print('file_name : ', file_name)

    out_file = open(file_name, 'wb')

    if d_type == 'complex':
        for i in range(len(write_data)):
           data = struct.pack('ff', np.real(write_data[i]), np.imag(write_data[i])) 
           out_file.write(data)
        out_file.close()

    elif d_type == 'abs':
        for i in range(len(write_data)):
           data = struct.pack('f', write_data[i])
           out_file.write(data)
        out_file.close()
    
    print('File write end')


def recv_to_file(rx_stream, timer_elapsed_event):

    metadata = uhd.types.RXMetadata()

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_stream.issue_stream_cmd(stream_cmd)

    num_rx_samps= 0
    
#    while not timer_elapsed_event.is_set():
#        num_rx_samps += rx_stream.recv(recv_buffer, metadata)
#        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
#            print(metadata.strerror())
    num_rx_samps = rx_stream.recv(recv_buffer, metadata)    

    print('num_rx_samps= ', num_rx_samps)
    
        

def send_from_file(tx_data, tx_stream, timer_elapsed_event):

    md=uhd.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = False
    timeout = 0.1

    num_tx_samps = 0

#    while not timer_elapsed_event.is_set():
#        num_tx_samps += tx_stream.send(tx_data, md, timeout)
    num_tx_samps = tx_stream.send(tx_data, md, timeout)

    print('num_tx_samps= ', num_tx_samps) 


    
def main():
    tune_req = uhd.types.TuneRequest(freq)

    #Setting
    txusrp.set_tx_rate(rate)
    txusrp.set_tx_freq(tune_req)
    txusrp.set_tx_gain(tx_gain)
#    txusrp.set_tx_bandwidth(tx_bw)
    txusrp.set_tx_antenna("TX/RX")

    rxusrp.set_rx_rate(rate)
    rxusrp.set_rx_freq(tune_req)
    rxusrp.set_rx_gain(rx_gain)
#    rxusrp.set_rx_bandwidth(rx_bw)
    rxusrp.set_rx_antenna("RX2")

    #make tx streamer
    cpu_format = "fc32"
    otw_format = "sc16"
    stream_args = uhd.usrp.StreamArgs(cpu_format, otw_format)
    tx_stream = txusrp.get_tx_stream(stream_args)

    #make rx streamer
    rx_stream = rxusrp.get_rx_stream(stream_args)
    
    '''    
    #print('filters :', txusrp.get_filter_names())
    #filters = txusrp.get_filter_names()

    # get Low Pass Filter objects
    tx_lpf1 = txusrp.get_filter('/mboards/0/dboards/A/tx_frontends/A/filters/LPF_BB')
    tx_lpf2 = txusrp.get_filter('/mboards/0/dboards/A/tx_frontends/A/filters/LPF_SECONDARY')
    rx_lpf1 = rxusrp.get_filter('/mboards/0/dboards/A/rx_frontends/A/filters/LPF_BB')
    rx_lpf2 = rxusrp.get_filter('/mboards/0/dboards/A/rx_frontends/A/filters/LPF_TIA')
   
    #tx_lpf1.set_cutoff(tx_lpf_cutoff)
    #tx_lpf2.set_cutoff(tx_lpf_cutoff)
    #rx_lpf1.set_cutoff(tx_lpf_cutoff)
    #rx_lpf2.set_cutoff(tx_lpf_cutoff)
    
    print('cutoff freq. (tx_lpf1)  :', tx_lpf1.get_cutoff())
    print('rolloff freq. (tx_lpf1) :', tx_lpf1.get_rolloff())
    print('cutoff freq. (tx_lpf2)  :', tx_lpf2.get_cutoff())
    print('rolloff freq. (tx_lpf2) :', tx_lpf2.get_rolloff())
    print('cutoff freq. (rx_lpf1)  :', rx_lpf1.get_cutoff())
    print('rolloff freq. (rx_lpf1) :', rx_lpf1.get_rolloff())
    print('cutoff freq. (rx_lpf2)  :', rx_lpf2.get_cutoff())
    print('rolloff freq. (rx_lpf2) :', rx_lpf2.get_rolloff())

    print('TX bandwidth : ', txusrp.get_tx_bandwidth())
    print('RX bandwidth : ', rxusrp.get_rx_bandwidth())
    '''

    # Read tx data from files
    data_real = np.zeros(tx_num_samp)
    data_imag = np.zeros(tx_num_samp)
    
    with open(tx_file, "rb") as f:
        for i in range(tx_num_samp):
            data_real[i] = struct.unpack('f', f.read(4))[0]
            data_imag[i] = struct.unpack('f', f.read(4))[0]
    

    tx_data = np.complex64(data_real + 1j*data_imag)
    
    ## Thread
    threads = []
    quit_event = threading.Event()

    #RX thread
    rx_thread = threading.Thread(target=recv_to_file, 
                                 args=(rx_stream, quit_event))
    threads.append(rx_thread)
    

    #TX thread
    tx_thread = threading.Thread(target=send_from_file, 
                                 args=(tx_data, tx_stream, quit_event))

    threads.append(tx_thread)


    #Transmit & Receive
    rx_thread.start()
    time.sleep(0.002)
    tx_thread.start()


#    time.sleep(1)
#    time.sleep(300)
    quit_event.set()

   
    for thr in threads:
        thr.join()

    print('')
    print('tx_num_samp:', tx_num_samp)

    max_tx_real = np.max(np.abs(np.real(tx_data)))
    max_tx_imag = np.max(np.abs(np.imag(tx_data)))
    print('')
    print('max_tx_real:', max_tx_real)
    print('max_tx_imag:', max_tx_imag)


    max_rx_real = np.max(np.abs(np.real(recv_buffer)))
    max_rx_imag = np.max(np.abs(np.imag(recv_buffer)))

    print('')
    print('max_rx_real:', max_rx_real)
    print('max_rx_imag:', max_rx_imag)


    ####### Downsampling #####################
    recv_buffer_ds =  recv_buffer[0::2]
    print('len(recv_buffer_ds): ', len(recv_buffer_ds))
    
    ####### Covariance matrix ################
    set_size = 90000
    N_samp   = 32000 #32000
    cov_size = 81
    norm_offset = 5

    cov_mat_set = np.zeros((set_size, cov_size))
    cov_mat_norm = np.zeros((set_size, cov_size))
    cov_max_idx = np.zeros(set_size)

    '''
    ### Original ver. ###
    
    for k in range(set_size):
        l = k*N_samp
        for i in range(cov_size):
            cov_mat_set[k][i] = np.abs(np.dot(recv_buffer_ds[l:l+N_samp] , np.conj(recv_buffer_ds[i+l:i+l+N_samp])))

        cov_mat_abs = cov_mat_set[k,:]

        cov_max = np.max(cov_mat_abs[norm_offset:len(cov_mat_abs)])
        cov_max_idx = np.argmax(cov_mat_abs[norm_offset:len(cov_mat_abs)]) + norm_offset

        cov_min = np.min(cov_mat_abs[norm_offset:len(cov_mat_abs)])

        cov_mat_norm[k,:] = (cov_mat_abs - cov_min)/(cov_max - cov_min)
        cov_mat_norm[k][0] = 1

        print('cov_max_idx[', str(k), '] :', cov_max_idx)
    '''

    ### Real-time ver. ###
    for k in range(set_size):
        for i in range(cov_size):
            cov_mat_set[k][i] = np.abs(np.dot(recv_buffer_ds[k:k+N_samp] , np.conj(recv_buffer_ds[i+k:i+k+N_samp])))

        cov_mat_abs = cov_mat_set[k,:]

        cov_max = np.max(cov_mat_abs[norm_offset:len(cov_mat_abs)])
        cov_max_idx[k] = np.argmax(cov_mat_abs[norm_offset:len(cov_mat_abs)]) + norm_offset

        cov_min = np.min(cov_mat_abs[norm_offset:len(cov_mat_abs)])

        cov_mat_norm[k,:] = (cov_mat_abs - cov_min)/(cov_max - cov_min)
        cov_mat_norm[k][0] = 1
        
        if k % 1000 == 0:
            print('k :', k)
#        print('cov_max_idx[', str(k), '] :', cov_max_idx)
    
    
    plt.figure(20)
    t = np.arange(set_size)
    plt.plot(t, cov_max_idx)

    plt.figure(21)
    t = np.arange(cov_size)
    plt.plot(t, cov_mat_norm[0,:])

    plt.show()

    
    ####### Write file #######################
    cov_mat_norm_reshape = cov_mat_norm.reshape(-1, cov_mat_norm.size)[0]
    write_file(cov_mat_norm_reshape, 'S_test_cov.out', 'abs')
    
    write_file(recv_buffer_ds, 'rx_data_after_downsampling.out', 'complex')
    
    ####### Spectrogram ######################
    N_sp = 35
    N = 4096
    N2 = int(N/2)

    sp = np.zeros((N_sp, N), dtype=np.complex64)
    SP = np.zeros((N_sp, N), dtype=np.complex64)
    
    hanning = np.hanning(N)

    # You should detect the starting point to plot spectrogram! It is not a good value
    k = 50000 

    for i in range(N_sp):
        sp[i] = recv_buffer[k:k+N]

        for j in range(N):
            sp[i][j] = sp[i][j]*hanning[j]

        SP[i] = np.fft.fftshift(np.fft.fft(sp[i]))
        k += N2
    
    ST = np.mean(np.abs(SP)**2, axis=0)
    log_ST = 10*np.log10(ST)


    ####### Spectrogram after down-sampling ######
    sp_ds = np.zeros((N_sp, N), dtype=np.complex64)
    SP_ds = np.zeros((N_sp, N), dtype=np.complex64)

    for i in range(N_sp):
        sp_ds[i] = recv_buffer_ds[(k//2):(k//2)+N]

        for j in range(N):
            sp_ds[i][j] = sp_ds[i][j]*hanning[j]

        SP_ds[i] = np.fft.fftshift(np.fft.fft(sp_ds[i]))
        k += N2

    ST_ds = np.mean(np.abs(SP_ds)**2, axis=0)
    log_ST_ds = 10*np.log10(ST_ds)
    
    
    tx_t = np.arange(0, tx_num_samp, 1)
    rx_t = np.arange(0, rx_num_samp, 1)
    rx_ds_t = np.arange(0, len(recv_buffer_ds) , 1)

    tx_data_fig, axis = plt.subplots(2,1, constrained_layout=True)

    axis[0].plot(tx_t, np.real(tx_data))
    axis[0].set_title('real(tx_data)')

    axis[1].plot(tx_t, np.imag(tx_data))
    axis[1].set_title('imag(tx_data)')

    
    rx_data_fig, axs_ = plt.subplots(2,1, constrained_layout=True)

    axs_[0].plot(rx_t, np.real(recv_buffer))
    axs_[0].set_title('real(recv_buffer)')

    axs_[1].plot(rx_t, np.imag(recv_buffer))
    axs_[1].set_title('imag(recv_buffer)')

    rx_data_ds_fig, axs_ = plt.subplots(2,1, constrained_layout=True)

    axs_[0].plot(rx_ds_t, np.real(recv_buffer_ds))
    axs_[0].set_title('real(recv_buffer_ds)')

    axs_[1].plot(rx_ds_t, np.imag(recv_buffer_ds))
    axs_[1].set_title('imag(recv_buffer_ds)')
    
    
    wfr = rate/N*np.arange(-N2, N2, 1)
    
    plt.figure(10)
    plt.plot(wfr, log_ST)
    plt.ylim(top=30)
    plt.title('Power spectral density')
    plt.xlabel('Hz')
    plt.ylabel('dB')
    plt.grid()

    wfr_ds = (rate/2)/N*np.arange(-N2, N2, 1)

    plt.figure(11)
    plt.plot(wfr_ds, log_ST_ds)
    plt.ylim(top=30)
    plt.title('Power spectral density (After down-sampling)')
    plt.xlabel('Hz')
    plt.ylabel('dB')
    plt.grid()
    

    plt.show()
    
    print('end')

if __name__ == "__main__":
    main()

