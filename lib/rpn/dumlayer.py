import caffe
import numpy as np

class dumLayer(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        # check input dimensions match
        top[0].reshape(bottom[0].data.shape[0],1,14,14)
        top[1].reshape(bottom[0].data.shape[0],512,14,14)
        
    def forward(self, bottom, top):
        #print(bottom[0].data)
        dot_p = bottom[0].data[:,:,:,None] * bottom[1].data
        
        top[0].data[...] = np.mean(dot_p, axis=1, keepdims=True)
        top[1].data[...] = dot_p
    def backward(self, top, propagate_down, bottom):
        pass
