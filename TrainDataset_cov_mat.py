import os
import struct
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset(Dataset):
    def __init__(self, dir_path):
        self.x_data = []
        self.y_data = []

        num_set = 100 
        num_samp = 81
        S_label = [1,0]
        N_label = [0,1]

        files = os.listdir(dir_path)
        files.sort()

        for F in files:
            print('filename :', F)
            if F.startswith('S'):
                with open(os.path.join(dir_path, F), 'rb') as f:
                    for i in range(num_set):
                        d=[]
                        for j in range(num_samp):
                            d.append(struct.unpack('f', f.read(4))[0])

                        self.x_data.append(d)
                        self.y_data.append(S_label)

            if F.startswith('N'):
                with open(os.path.join(dir_path, F), 'rb') as f:
                    for i in range(num_set):
                        d=[]
                        for j in range(num_samp):
                            d.append(struct.unpack('f', f.read(4))[0])

                        self.x_data.append(d)
                        self.y_data.append(N_label)
            
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        
        x = torch.cuda.FloatTensor(self.x_data[idx])
        y = torch.cuda.FloatTensor(self.y_data[idx])
        return x, y
