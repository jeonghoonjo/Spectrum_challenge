import struct
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TestDataset():
    def __init__(self):
        super(TestDataset, self).__init__()

        self.x_data = []
        self.y_data = []
        data        = []
        label       = []

        num_data = 10
        num_samp = 801
                
        filename = ['test_set_00.out','test_set_01.out',
                    'test_set_02.out','test_set_03.out',
                    'test_set_04.out','test_set_05.out',
                    'test_set_06.out','test_set_07.out',
                    'test_set_08.out',
                    'test_noise_00.out','test_noise_01.out',
                    'test_noise_02.out','test_noise_03.out',
                    'test_noise_04.out','test_noise_05.out',
                    'test_noise_06.out','test_noise_07.out',
                    'test_noise_08.out']
        '''
        filename = ['test_set_norm_00.out','test_set_norm_01.out',
                    'test_set_norm_02.out','test_set_norm_03.out',
                    'test_set_norm_04.out','test_set_norm_05.out',
                    'test_set_norm_06.out','test_set_norm_07.out',
                    'test_set_norm_08.out',
                    'test_noise_norm_00.out','test_noise_norm_01.out',
                    'test_noise_norm_02.out','test_noise_norm_03.out',
                    'test_noise_norm_04.out','test_noise_norm_05.out',
                    'test_noise_norm_06.out','test_noise_norm_07.out',
                    'test_noise_norm_08.out']
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
