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
num_samp = 20000 #80000

rate=10e6
freq=90e6
tx_gain=55 #55
rx_gain=15.5

myusrp=uhd.usrp.MultiUSRP("name=MyB200")


global recv_buffer
recv_buffer = np.zeros(num_samp, dtype=np.complex64)

global noise_buffer
noise_buffer = np.zeros(num_samp, dtype=np.complex64)


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

    
    #tx signal
    wfreq = 50e3
    tx_amplitude = 0.9

    n = np.arange(int(np.floor(rate/wfreq)))
    n = np.arange(1000)


    tx_data = tx_amplitude*np.complex64(np.exp(2j*np.pi*wfreq*n/rate)) # n/rate : 1/fs 

    tx_data_fig, axs_ = plt.subplots(2,1, constrained_layout=True)

    
    max_val = np.max(np.abs(tx_data))
    print('max tx_val', max_val)
    
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
    
    max_rx_real = np.max(np.abs(np.real(recv_buffer)))
    max_rx_imag = np.max(np.abs(np.imag(recv_buffer)))

    print('max_rx_real :', max_rx_real)
    print('max_rx_imag :', max_rx_imag)


    ####### Spectrogram  #############################

    #Fix the RBW at 4.88kHz
    rbw = 10e6/2048
    N = int(rate/rbw)
    N2 = int(N/2)

    N_sp = int((num_samp*2/N +1)/2)

    sp = np.zeros((N_sp, N), dtype=np.complex64)
    SP = np.zeros((N_sp, N), dtype=np.complex64)

    hanning = np.hanning(N)

    k = 0

    for i in range(N_sp):
        sp[i] = recv_buffer[k:k+N]

        for j in range(N):
            sp[i][j] = sp[i][j]*hanning[j]

        SP[i] = np.fft.fft(sp[i])
        k += N2

    ST = np.mean(np.abs(SP)**2, axis=0)   # power
    log_ST = 10*np.log10(ST)              # [dB]

    ####### Spectrogram of noise ###############

    sp_noise = np.zeros((N_sp, N), dtype=np.complex64)
    SP_noise = np.zeros((N_sp, N), dtype=np.complex64)

    k = 0

    for i in range(N_sp):
        sp_noise[i] = noise_buffer[k:k+N]

        for j in range(N):
            sp_noise[i][j] = sp_noise[i][j]*hanning[j]

        SP_noise[i] = np.fft.fft(sp_noise[i])
        k += N2

    ST_noise = np.mean(np.abs(SP_noise)**2, axis=0)
    log_ST_noise = 10*np.log10(ST_noise)


    ####### Get power ###########################
    evt_var = np.zeros(N_sp, dtype=np.complex64)
    evt_noise_var = np.zeros(N_sp, dtype=np.complex64)


    for i in range(N_sp):
        evt_var[i] = np.var(sp[i])
        evt_noise_var[i] = np.var(sp_noise[i])

    sig_power   = 10*np.log10(np.mean(np.abs(evt_var), axis=0)*1e3)       # [dBm]
    noise_power = 10*np.log10(np.mean(np.abs(evt_noise_var), axis=0)*1e3) # [dBm]
    SNR = sig_power - noise_power

    print('')
    print('Signal power :', sig_power, '[dBm]')
    print('Noise power  :', noise_power, '[dBm]')
    print('SNR          :', SNR, '[dB]')
    print('')

    tx_data_fig, axs_t = plt.subplots(2,1, constrained_layout=True)


    axs_t[0].plot(n, np.real(tx_data))
    axs_t[0].set_title('real(tx_data)')

    axs_t[1].plot(n, np.imag(tx_data))
    axs_t[1].set_title('imag(tx_data)')

    rx_data_fig, axs_r = plt.subplots(2,1, constrained_layout=True)

    t = np.arange(0, num_samp, 1)

    axs_r[0].plot(t, np.real(recv_buffer))
    axs_r[0].set_title('real(recv_buffer)')
#    axs_r[0].set_ylim([-1, 1])

    axs_r[1].plot(t, np.imag(recv_buffer))
    axs_r[1].set_title('imag(recv_buffer)')
#    axs_r[1].set_ylim([-1, 1])

    plt.figure(10)
    wfr = rate/N*np.arange(N2)
    plt.plot(wfr, log_ST[0:N2]) #[0:N2]
#    plt.xscale('log')
    plt.ylim(top=80)
    plt.ylabel('dB')
    plt.xlabel('Hz')
    plt.title('Power Spectral Density')
    plt.grid()


    max_idx = np.argmax(np.abs(log_ST[0:N2])) #[0:N2]
    print('max spectrum freq.:', wfr[max_idx])
    print('')
    print('@10MHz :', log_ST[int(N2/2)])
    print('wfr(10MHz) :', wfr[int(N2/2)])
    print('@20MHz :', log_ST[N2-1])
    print('wfr(20MHz) :', wfr[N2-1])
    print('diff   :', log_ST[int(N2/2)]-log_ST[N2-1], 'dB')

    plt.show()


if __name__ == "__main__":
    main()

