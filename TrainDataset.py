import struct
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset():
    def __init__(self):
        super(TrainDataset, self).__init__()

        self.x_data = []
        self.y_data = []
        data        = []
        label       = []

        num_data = 100
        num_samp = 801
    
        
        filename = ['training_set_00.out','training_set_01.out',
                    'training_set_02.out','training_set_03.out',
                    'training_set_04.out','training_set_05.out',
                    'training_set_06.out','training_set_07.out',
                    'training_set_08.out',
                    'training_noise_00.out','training_noise_01.out',
                    'training_noise_02.out','training_noise_03.out',
                    'training_noise_04.out','training_noise_05.out',
                    'training_noise_06.out','training_noise_07.out',
                    'training_noise_08.out']
        '''
        filename = ['training_set_norm_00.out','training_set_norm_01.out',
                    'training_set_norm_02.out','training_set_norm_03.out',
                    'training_set_norm_04.out','training_set_norm_05.out',
                    'training_set_norm_06.out','training_set_norm_07.out',
                    'training_set_norm_08.out',
                    'training_noise_norm_00.out','training_noise_norm_01.out',
                    'training_noise_norm_02.out','training_noise_norm_03.out',
                    'training_noise_norm_04.out','training_noise_norm_05.out',
                    'training_noise_norm_06.out','training_noise_norm_07.out',
                    'training_noise_norm_08.out']
        '''
        num_file = len(filename)

        for k in range(num_file):
            with open(filename[k], 'rb') as f:
                print('filename:', filename[k])

                for i in range(num_data):
                    data  = []
                    label = []
                    for j in range((num_samp-1)*2):
                        data_ = struct.unpack('f', f.read(4))[0]
                        data.append(data_)

                    data_ = struct.unpack('f', f.read(4))[0]
                    label.append(data_)
                    data_ = struct.unpack('f', f.read(4))[0]
                    label.append(data_)
            
                    self.x_data.append(data)
                    self.y_data.append(label)
    
       
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        
        x = torch.cuda.FloatTensor(self.x_data[idx])
        y = torch.cuda.FloatTensor(self.y_data[idx])
        return x, y
