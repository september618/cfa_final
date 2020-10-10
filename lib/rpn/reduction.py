import caffe
import numpy as np

class reductionlayer(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        # check input dimensions match
        top[0].reshape(bottom[0].data.shape[0],1,14,14)
        
    def forward(self, bottom, top):
        top[0].data[...] = np.mean(bottom[0], axis=1)

    def backward(self, top, propagate_down, bottom):
        pass
