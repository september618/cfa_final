import caffe


class dummyLayer(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        # check input dimensions match

        pass
    def forward(self, bottom, top):
        pass
        #print("label",bottom[0].data)
        #print("predict",bottom[1].data)


    def backward(self, top, propagate_down, bottom):
        pass

