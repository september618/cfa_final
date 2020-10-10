import caffe
import numpy as np

class DivLossLayer(caffe.Layer):
    """
    Compute div loss from ICCV2017 `Learning Multi-Attention Convolutional Neural 
    Network for Fine-Grained Image Recognition`
    """
    def setup(self, bottom, top):
        # check input
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance")

    def reshape(self, bottom, top):
        # check input dimensions match        
        if bottom[0].data.shape[1] != 784:
            raise Exception("Bottom 1 must have the 784.")
        if bottom[1].data.shape[1] != 784:
            raise Exception("Bottom 2 must have the 784.")
        if bottom[2].data.shape[1] != 784:
            raise Exception("Bottom 3 must have the 784.")
#        if bottom[3].data.shape[1] != 784:
#            raise Exception("Bottom 4 must have the 784.")
     
        # difference is shape of inputs
        self.diff0 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff1 = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff2 = np.zeros_like(bottom[2].data, dtype=np.float32)
 #       self.diff3 = np.zeros_like(bottom[3].data, dtype=np.float32)
 
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        margin = 0.0

        self.diff0[...] = np.max(np.vstack([[bottom[1].data], [bottom[2].data]]), axis=0)
        self.diff1[...] = np.max(np.vstack([[bottom[0].data], [bottom[2].data]]), axis=0)
        self.diff2[...] = np.max(np.vstack([[bottom[0].data], [bottom[1].data]]), axis=0)
     #  self.diff3[...] = np.max(np.vstack([[bottom[0].data], [bottom[1].data], [bottom[2].data]]), axis=0) - margin

        top[0].data[...] =  (np.sum(bottom[0].data*self.diff0) + np.sum(bottom[1].data*self.diff1) +
                            np.sum(bottom[2].data*self.diff2) ) / bottom[0].num

        

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] =  self.diff0 / bottom[0].num
        bottom[1].diff[...] =  self.diff1 / bottom[1].num
        bottom[2].diff[...] =  self.diff2 / bottom[2].num
#       bottom[3].diff[...] =  self.diff3 / bottom[3].num
