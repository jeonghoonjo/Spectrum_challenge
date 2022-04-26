% ofdm lab - TXRX  
% by Youngsik Kim @2013.11.11
%

%clearing
close all;
clear all;
clc;

% Setting
N = 12000000; % number of bits of the source data 800000
% N = 150000;
tx_mod = 4; % 1 for bpsk, 2 for qpsk, 3 for 16qam, 4 for 64 qam

T_symbol_interval = 4e-6; % symbol interval = 4us
T_fft_interval = 3.2e-6; % FFT interval = 3.2us
T_sampling = T_fft_interval/64; % 50ns
F_sampling_freq=1/T_sampling; %sampling clock 20MHz base band


%Short Tr
% @4 -1-j, @8 -1-j,     @12 1+j,  @16 1+j,   @20 1+j,  @24 1+j 
%@-24 1+j, @-20 -1 -j,  @-16 1+j, @-12 -1-j, @-8 -1-j  @-4 1+j
freq_short=sqrt(13/6)*[0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+j, 0,0, ...
0,0,0,0,0,0,0,0,0,0,0,...
0, 0, 1+1j, 0, 0, 0,-1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0];

%Long Tr
freq_long=[0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, ...
0,0,0,0,0,0,0,0,0,0,0,...
1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1];

time_short = ifft(freq_short); 
time_long = ifft(freq_long);

N_samp = 32000;
cov_size = 81;
set_size = 100;
% set_size = 1;

cov_mat_0_set = zeros(set_size, cov_size);
cov_mat_norm = zeros(set_size, cov_size);

offset = 5;
SNR_list = [0, -6, -10, -12, -13, -15, -18, -20]; %[0, -6, -10, -12, -15, -20];

is_sig = 0;
is_training = 0;
size_st = 3333280;
% size_st = 41600;

if is_sig
    if is_training
        cnt_num = 10;
        file_head = '~/py_workspace/training_set_mp/S_training_';
    else
        cnt_num = 50;
        file_head = '~/py_workspace/test_set_mp/S_test_';
    end
else
    if is_training
        cnt_num = 10;
        file_head = '~/py_workspace/training_set_mp/N_training_';
    else
        cnt_num = 50;
        file_head = '~/py_workspace/test_set_mp/N_test_';
    end
end



for cnt= 1:cnt_num
    disp(strcat('set', int2str(cnt)))
    
    for SNR = SNR_list
        rng shuffle;
        if is_sig
            %---- TX Begin
            tx_data = round(rand(1,N)); %base-data - supplied from MAC layer
            %1. Modulation
            tx_mapped_data = tx_mapping(tx_data, tx_mod);

            %2. Pilot Generation
            %Pilot polarity generation
            pilot = scramble(zeros(1,127),ones(1,7));
            % 0->1, 1->-1 conversion
            pilot_polarity = 1-2*pilot;

            %3. symbol assemble
            symbol_length = floor(length(tx_mapped_data)/48);


            %pre-allocate symbol data
            tx_symbol_assembled_data = complex(zeros(1,symbol_length*64));
            tx_time_domain_data = complex(zeros(1,symbol_length*64));
            tx_cp_added_data = complex(zeros(1,symbol_length*80));


            for n=1:symbol_length
                pn = pilot_polarity(mod(n-1,127)+1); %pilot channel polarity
                px = tx_mapped_data(n*48-47:n*48);
                py(1)=0; % DC channel
                py(2:7)=px(1:6); % data channel from 1 to 6 ( 6 channels )
                py(8) = pn; % pilot channel @7
                py(9:21)=px(7:19); % data channel from 8 to 20( 13 channels )
                py(22) = -pn; %pilot channel @21
                py(23:27) = px(20:24); %data channel from 22 to 26( 5 channels )
                py(28:38) = 0; % Gaurd band 27 to 31 and -32 to -27 channels
                py(39:43) = px(25:29); % data channel from -26 to -22
                py(44) = pn; % pilot channel -21
                py(45:57) = px(30:42); %data channel from -20 to -8 ( 11 channels )
                py(58) = pn; % pilot channel -7
                py(59:64) = px(43:48); %data channel from -6 to -1 ( 6 channel )

                %add assmebled data to tx symbol package
                tx_symbol_assembled_data(64*n-63:n*64) = py;
                % convert the symbol to the time domain data
            %     tx_time_domain_data(64*n-63:n*64) = fftshift(ifft(py));
                tx_time_domain_data(64*n-63:n*64) = ifft(fftshift(py));
                % add cyclic prefix
                tx_cp_added_data(n*80-79:n*80) = [tx_time_domain_data(n*64-15:n*64) tx_time_domain_data(n*64-63:n*64)];

            end

            st = tx_cp_added_data/sqrt(var(tx_cp_added_data))*10^(SNR/20);
            
            channel = [1, 0.1*exp(-1j*pi/10), 0.01*exp(-1j*pi/20)];
            st = conv(channel, st);
            
            cfo = 1.3e3;
            Nin = length(st);
            ts  = 0:1:Nin-1;
            
            st = st(1:length(ts)).*exp(2j*pi*cfo.*ts);
            
            nt = randn(size(st))+1j*randn(size(st));
%             nt = randn([1,size_st])+1j*randn([1,size_st]);
            nt = nt/sqrt(2);
            
            rt = st+nt;
            
           
        else
            nt = randn([1,size_st])+1j*randn([1,size_st]);
%             nt = randn(size(st))+1j*randn(size(st));
            nt = nt/sqrt(2);
            rt = nt;
        end

    
        rt = rt(500 : end);

        for k = 1:set_size
            l = (k-1)*N_samp;
            for i = 1:cov_size
                cov_mat_0_set(k,i) = dot(rt(1+l:N_samp+l), rt(i+l:N_samp+(i-1)+l));
            end
            cov_mat_abs = abs(cov_mat_0_set(k,:));

            % Min-max normalization (revision ver.)
            [cov_max, idx] = max(cov_mat_abs(offset:end));
            idx_arr(k) = idx + offset -1;
            cov_min = min(cov_mat_abs(offset:end));

            cov_mat_norm(k,:) = (cov_mat_abs - cov_min)/(cov_max - cov_min); 
            cov_mat_norm(k,1) = 1;

        end
        
        
%         figure(index);
%         cov_mat_norm_ = uint8(255) - uint8(255*cov_mat_norm);
%         cov_mat_image = reshape(cov_mat_norm_, [9,9]);
%         imshow(cov_mat_image);
%         title(strcat('cp-corr norm (',int2str(SNR), 'dB)'));
%         
%         index = index +1;
        
        fid = fopen(strcat(file_head, int2str(SNR),'_', int2str(cnt), '.bin'), 'w');
        for j=1:set_size
            fwrite(fid, cov_mat_norm(j,:), 'float');
        end
        fclose(fid);
    end
    
%     figure(1);
%     cov_mat_norm_ = uint8(255) - uint8(255*cov_mat_norm);
%     cov_mat_image = reshape(cov_mat_norm_, [9,9]);
%     imshow(cov_mat_image);
%     title('cp-corr norm (noise)');
    
    
end

% diag_cov_mat = ones([1, cov_size]);
% cov_mat_norm = (cov_mat_abs - cov_min)/(cov_max - cov_min); 
% cov_mat_norm = spdiags(diag_cov_mat', 0, cov_mat_norm);
% cov_mat_norm = full(cov_mat_norm);

% Min-max normalization (original ver.)
% cov_max = max(cov_mat_abs);
% cov_min = min(cov_mat_abs);
% cov_mat_norm_orig = (cov_mat_abs - cov_min)/(cov_max - cov_min);

% tm = 1:1:cov_size;
% figure(1);
% subplot(211);plot(tm, cov_mat_abs);
% ylim([0 6e4]);
% title('cov mat abs (noise)');
% subplot(212);
% plot(tm, cov_mat_norm);
% title('cov mat norm (noise)');
% 
% figure(2);
% cov_mat_norm_ = uint8(255) - uint8(255*cov_mat_norm(1,:));
% cov_mat_image = reshape(cov_mat_norm_, [9,9]);
% imshow(cov_mat_image);
% title('cp-corr norm');
% 
% figure(3);
% cov_mat_norm_orig_ = uint8(255) - uint8(255*cov_mat_norm_orig);
% cov_mat_image_orig = reshape(cov_mat_norm_orig_, [9,9]);
% imshow(cov_mat_image_orig);
% title('orig norm');










