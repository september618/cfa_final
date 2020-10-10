import caffe
import numpy as np

class AwareLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.lr=0.5
        self.alpha=0.5
        self.init=True

    def reshape(self, bottom, top):

        # difference is shape of inputs
        self.diff0 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff1 = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff2 = np.zeros_like(bottom[2].data, dtype=np.float32)
        self.diff3 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff4 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff5 = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff6 = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff7 = np.zeros_like(bottom[2].data, dtype=np.float32)
        self.diff8 = np.zeros_like(bottom[2].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)


    def forward(self, bottom, top):
        if self.init==True:   # self.init==True denotes we should init the previous memory of three masks
            self.memory0=bottom[0].data
            self.memory1=bottom[1].data
            self.memory2=bottom[2].data
            self.init=False

        self.diff0[...] = bottom[0].data-self.memory0
        self.diff1[...] = bottom[1].data-self.memory1
        self.diff2[...] = bottom[2].data-self.memory2

        self.diff3[...] = bottom[0].data-self.memory1
        self.diff4[...] = bottom[0].data-self.memory2
        self.diff5[...] = bottom[1].data-self.memory0
        self.diff6[...] = bottom[1].data-self.memory2
        self.diff7[...] = bottom[2].data-self.memory0
        self.diff8[...] = bottom[2].data-self.memory1

        #alpha is to avoid the loss is always negative
        top[0].data[...] = (np.sum(self.diff0**2)-(np.sum(self.diff3**2)+np.sum(self.diff4**2))*self.alpha)/2.0/ bottom[0].num
        top[1].data[...] = (np.sum(self.diff1**2)-(np.sum(self.diff5**2)+np.sum(self.diff6**2))*self.alpha)/2.0/ bottom[1].num
        top[2].data[...] = (np.sum(self.diff2**2)-(np.sum(self.diff7**2)+np.sum(self.diff8**2))*self.alpha)/2.0/ bottom[2].num

        self.memory0 = bottom[0].data * self.lr + self.memory0*(1 - self.lr)
        self.memory1 = bottom[1].data * self.lr + self.memory1*(1 - self.lr)
        self.memory2 = bottom[2].data * self.lr + self.memory2*(1 - self.lr)


    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = (self.diff0-(self.diff3+self.diff4)*self.alpha) / bottom[0].num
        bottom[1].diff[...] = (self.diff1-(self.diff5+self.diff6)*self.alpha) / bottom[1].num
        bottom[2].diff[...] = (self.diff2-(self.diff7+self.diff8)*self.alpha) / bottom[2].num

