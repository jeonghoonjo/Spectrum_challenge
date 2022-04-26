import numpy as np
import time
import uhd
import threading
import math
import signal
import struct
from matplotlib import pyplot as plt

stop_signal_called = False
repeat = True
#matlab tx_time_data size
num_samp = 3480 #80000

rate=5e6
freq=2.5e9
tx_gain=30 
rx_gain=30 

myusrp=uhd.usrp.MultiUSRP("name=MyB200_1")


global recv_buffer
recv_buffer = np.zeros(num_samp, dtype=np.complex64)

global noise_buffer
noise_buffer = np.zeros(num_samp, dtype=np.complex64)


def write_file(recv_buffer_, noise_buffer_):
    noise_file = 'noise_data.out'
    data_file  = 'rx_data.out'
#    noise_file = 'test_noise_norm_03_.out'
#    data_file  = 'test_set_norm_03_.out'

    print('noise file: ', noise_file)
    print('data file : ', data_file)

    out_file = open(noise_file, 'wb')
    
    for i in range(num_samp):
        '''
        #Put label
        label = [0, 1]

        if (i != 0) and (i % 800 == 0):
            data = struct.pack('ff', label[0], label[1])
            out_file.write(data)
        '''
        data = struct.pack('ff', np.real(noise_buffer_[i]),
                                 np.imag(noise_buffer_[i]))
        out_file.write(data)
    ''' 
    data = struct.pack('ff', label[0], label[1])
    out_file.write(data)
    '''
    out_file.close()


    out_file = open(data_file, 'wb')

    for i in range(num_samp):
        '''
        #Put label
        label = [1, 0]

        if (i != 0) and (i % 800 == 0):
            data = struct.pack('ff', label[0], label[1])
            out_file.write(data)
        '''
        data = struct.pack('ff', np.real(recv_buffer_[i]),
                                 np.imag(recv_buffer_[i]))
        out_file.write(data)
    '''  
    data = struct.pack('ff', label[0], label[1])
    out_file.write(data)
    '''
    out_file.close()

    print('File write end')

def normalization(recv_buffer_, noise_buffer_):
    rssi_len = 800
    rssi = np.zeros(rssi_len)
    N = 100

    for i in range(rssi_len-N-1):
        rssi[i+N] = 10*np.log10((np.var(recv_buffer_[i:i+N])*1e3)) #dBm

    rssi_val = np.mean(rssi)
    print('')
    print('rssi:', rssi_val, 'dBm')

    max_sig = np.max(recv_buffer_)
    min_sig = np.min(recv_buffer_)
    max_noise = np.max(noise_buffer_)
    min_noise = np.min(noise_buffer_)

    print('max_signal')
    print(max_sig)
    print('max_noise')
    print(max_noise)
    print('')
    
    recv_buffer_  = (recv_buffer_ - min_sig)/(max_sig - min_sig)
    noise_buffer_ = (noise_buffer_ - min_noise)(max_noise - min_noise)

    return recv_buffer_, noise_buffer_

def normalize(recv_buffer_, noise_buffer_):
    
    rssi_len = 800
    rssi = np.zeros(rssi_len)
    N = 100

    for i in range(rssi_len-N-1):
        rssi[i+N] = 10*np.log10((np.var(recv_buffer_[i:i+N])*1e3)) #dBm
        
    rssi_val = np.mean(rssi)
    print('')
    print('rssi:', rssi_val, 'dBm')

    max_sig = np.max(np.abs(recv_buffer_))
    max_noise = np.max(np.abs(noise_buffer_))

    print('max_signal') 
    print(max_sig)
    print('max_noise')
    print(max_noise)
    print('')

    recv_buffer_  = recv_buffer_/max_sig
    noise_buffer_ = noise_buffer_/max_noise


    return recv_buffer_, noise_buffer_

def standardization(recv_buffer_, noise_buffer_):
    rx_mean    = np.mean(recv_buffer_)
    noise_mean = np.mean(noise_buffer_)

    rx_std    = np.std(recv_buffer_)
    noise_std = np.std(noise_buffer_)

    recv_buffer_  = (recv_buffer_ - rx_mean)/rx_std
    noise_buffer_ = (noise_buffer_ - noise_mean)/noise_std

    return recv_buffer_, noise_buffer_


#Only Receive & get noise data
def cal_noise(rx_stream):

    metadata = uhd.types.RXMetadata()
    max_samps_per_packet = rx_stream.get_max_num_samps()
    
    #You should put these three lines
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_stream.issue_stream_cmd(stream_cmd)

    
    num_rx_samps=0
    
    num_rx_samps += rx_stream.recv(noise_buffer, metadata)


    if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
        print(metadata.strerror())

    print('num_rx_samps= ', num_rx_samps)



def recv_to_file(rx_stream, timer_elapsed_event):

    metadata = uhd.types.RXMetadata()
    max_samps_per_packet = rx_stream.get_max_num_samps()


    #You should put these three lines
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_stream.issue_stream_cmd(stream_cmd)

    num_rx_samps= 0
    
    while not timer_elapsed_event.is_set():
        num_rx_samps += rx_stream.recv(recv_buffer, metadata)
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(metadata.strerror())
    

    print('num_rx_samps= ', num_rx_samps)
    
        

def send_from_file(tx_data, tx_stream, timer_elapsed_event):

    md=uhd.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = False
    timeout = 0.1

    num_tx_samps = 0

    while not timer_elapsed_event.is_set():
        num_tx_samps += tx_stream.send(tx_data, md, timeout)

    print('num_tx_samps= ', num_tx_samps) 


    
def main():
    tune_req = uhd.types.TuneRequest(freq)

    #Setting
    myusrp.set_tx_rate(rate)
    myusrp.set_tx_freq(tune_req)
    myusrp.set_tx_gain(tx_gain)
    #myusrp.set_tx_bandwidth(tx_bw)
    myusrp.set_tx_antenna("TX/RX")

    myusrp.set_rx_rate(rate)
    myusrp.set_rx_freq(tune_req)
    myusrp.set_rx_gain(rx_gain)
    #myusrp.set_rx_bandwidth(rx_bw)
    myusrp.set_rx_antenna("RX2")

    #make tx streamer
    cpu_format = "fc32"
    otw_format = "sc16"
    stream_args = uhd.usrp.StreamArgs(cpu_format, otw_format)
    tx_stream = myusrp.get_tx_stream(stream_args)

    #make rx streamer
    rx_stream = myusrp.get_rx_stream(stream_args)
    
    # Read tx data from files
    data_real = np.zeros(num_samp)
    data_imag = np.zeros(num_samp)
    
    with open("tx_data_.bin", "rb") as f:
        for i in range(num_samp):
            data_real[i] = struct.unpack('f', f.read(4))[0]
            data_imag[i] = struct.unpack('f', f.read(4))[0]
    

    tx_data = np.complex64(data_real + 1j*data_imag)
    
    ## Thread
    threads = []
    quit_event = threading.Event()

    #Noise thread
    noise_thread = threading.Thread(target=cal_noise, 
                                    args=(rx_stream,))

    #RX thread
    rx_thread = threading.Thread(target=recv_to_file, 
                                 args=(rx_stream, quit_event))
    threads.append(rx_thread)
    

    #TX thread
    tx_thread = threading.Thread(target=send_from_file, 
                                 args=(tx_data, tx_stream, quit_event))

    threads.append(tx_thread)

    #Only Receive
    noise_thread.start()
    noise_thread.join()

    #Transmit & Receive
    tx_thread.start()

    time.sleep(0.5)
    rx_thread.start()

    time.sleep(0.5)
    quit_event.set()
    
    for thr in threads:
        thr.join()
    
    #Data normalization
    #recv_buffer_, noise_buffer_ = normalization(recv_buffer, noise_buffer)

    
    #Write files
    write_file(recv_buffer, noise_buffer)

    
    ''' 
    t = np.arange(0, num_samp, 1)

    tx_data_fig, axis = plt.subplots(2,1, constrained_layout=True)

    axis[0].plot(t, np.real(tx_data))
    axis[0].set_title('real(tx_data)')

    axis[1].plot(t, np.imag(tx_data))
    axis[1].set_title('imag(tx_data)')

    
    rx_data_fig, axs_ = plt.subplots(2,1, constrained_layout=True)

    axs_[0].plot(t, np.real(recv_buffer))
    axs_[0].set_title('real(recv_buffer)')

    axs_[1].plot(t, np.imag(recv_buffer))
    axs_[1].set_title('imag(recv_buffer)')

    noise_data_fig, ax = plt.subplots(2,1, constrained_layout=True)

    ax[0].plot(t, np.real(noise_buffer))
    ax[0].set_title('real(noise_buffer)')

    ax[1].plot(t, np.imag(noise_buffer))
    ax[1].set_title('imag(noise_buffer)')

    rx_data_fig_, axs = plt.subplots(2,1, constrained_layout=True)

    axs[0].plot(t, np.real(recv_buffer_))
    axs[0].set_title('real(normalized recv_buffer)')

    axs[1].plot(t, np.imag(recv_buffer_))
    axs[1].set_title('imag(normalized recv_buffer)')
    
    noise_data_fig_, axss = plt.subplots(2,1, constrained_layout=True)

    axss[0].plot(t, np.real(noise_buffer_))
    axss[0].set_title('real(normalized noise_buffer)')

    axss[1].plot(t, np.imag(noise_buffer_))
    axss[1].set_title('imag(normalized noise_buffer)')
    '''
    plt.show()
    
    print('end')

if __name__ == "__main__":
    main()

