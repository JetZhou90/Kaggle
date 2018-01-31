from sklearn.model_selection import train_test_split
import numpy as np

class dataload:

    def __init__(self,path):
        self.cam_img=np.load(path)
        self.cam_img=np.asarray(self.cam_img,dtype='float32')
        self.batch_index=0
        self.batch_size=0
        
    def train_next_batch(self,batch_size):
        self.batch_size=batch_size
        batch_x = self.cam_img[self.batch_index * self.batch_size: self.batch_index * self.batch_size + self.batch_size]
        self.batch_index += 1
        if self.batch_index > self.cam_img.shape[0] // self.batch_size:
            self.batch_index=0
        return batch_x


