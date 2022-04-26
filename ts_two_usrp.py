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
#matlab tx_time_data size
tx_file = 'test.bin'
rx_num_samp = 160000
tx_num_samp = getsize(tx_file)//8

rate    = 40e6
freq    = 2.6e9
tx_gain = 60
rx_gain = 25
tx_bw   = 20e6
rx_bw   = 18e6

tx_lpf_cutoff = 11e6

txusrp=uhd.usrp.MultiUSRP("name=MyB200")
rxusrp=uhd.usrp.MultiUSRP("name=MyB200_1")


global recv_buffer
recv_buffer = np.zeros(rx_num_samp, dtype=np.complex64)

global noise_buffer
noise_buffer = np.zeros(rx_num_samp, dtype=np.complex64)


def write_file(recv_buffer_, noise_buffer_):
    noise_file = 'noise_data.out'
    data_file  = 'rx_data.out'
    
    print('noise file: ', noise_file)
    print('data file : ', data_file)

    out_file = open(noise_file, 'wb')
    
    for i in range(num_samp):
        data = struct.pack('ff', np.real(noise_buffer_[i]), np.imag(noise_buffer_[i]))
        out_file.write(data)
    out_file.close()


    out_file = open(data_file, 'wb')

    for i in range(num_samp):
       data = struct.pack('ff', np.real(recv_buffer_[i]), np.imag(recv_buffer_[i]))
       out_file.write(data)
    out_file.close()

    print('File write end')


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
#    num_rx_samps = rx_stream.recv(recv_buffer, metadata)    

    print('num_rx_samps= ', num_rx_samps)
    
        

def send_from_file(tx_data, tx_stream, timer_elapsed_event):

    md=uhd.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst = False
    timeout = 0.1

    num_tx_samps = 0

    while not timer_elapsed_event.is_set():
        num_tx_samps += tx_stream.send(tx_data, md, timeout)
#    num_tx_samps = tx_stream.send(tx_data, md, timeout)
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
    rx_thread.start()

    time.sleep(0.5)
#    time.sleep(300)
    quit_event.set()
   
    for thr in threads:
        thr.join()

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

    max_n_real = np.max(np.abs(np.real(noise_buffer)))
    max_n_imag = np.max(np.abs(np.imag(noise_buffer)))
    print('')
    print('max_no_real:', max_n_real)
    print('max_no_imag:', max_n_imag)

    print('')
    p_sig = np.var(recv_buffer)
    print('P_signal   :', p_sig)

    p_noise = np.var(noise_buffer)
    print('P_noise    :', p_noise)

    snr = 10*np.log10(p_sig/p_noise)
    
    print('SNR        :', snr)
    print('')
    
    
    #Write files
#    write_file(recv_buffer, noise_buffer)
    
    ####### Spectrogram ######################
    N_sp = 35
    N = 4096
    N2 = int(N/2)

    sp = np.zeros((N_sp, N), dtype=np.complex64)
    SP = np.zeros((N_sp, N), dtype=np.complex64)
    
    hanning = np.hanning(N)

    k = 0 

    for i in range(N_sp):
        sp[i] = recv_buffer[k:k+N]

        for j in range(N):
            sp[i][j] = sp[i][j]*hanning[j]

        SP[i] = np.fft.fftshift(np.fft.fft(sp[i]))
        k += N2
    
    ST = np.mean(np.abs(SP)**2, axis=0)
    log_ST = np.log10(ST)


    tx_t = np.arange(0, tx_num_samp, 1)
    rx_t = np.arange(0, rx_num_samp, 1)

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

    plt.figure(10)
    wfr = rate/N*np.arange(N)
    plt.plot(wfr, log_ST) #[0:N2]
    plt.xscale('log')
#    plt.ylim(top=1)
    plt.title('Spectrogram of rx data(log)')

    plt.figure(11)
    plt.plot(wfr, log_ST) #[0:N2]
    plt.ylim(top=3)
    plt.title('Power spectral density')
    plt.xlabel('Hz')
    plt.ylabel('dB')
    plt.grid()

    plt.show()
    
    print('end')

if __name__ == "__main__":
    main()

