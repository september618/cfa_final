import caffe
import numpy as np


class stat(caffe.Layer):  # Another version of edited_cls_aware to avoid overfitting
    def __init(self):
        self.alpha = 0.05  # alpha is to avoid the loss is always negative
        self.num_cls = 21

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):

        # 0 cls, 1 bbox
        pass

    def forward(self, bottom, top):
        self.__init()
        bbox = bottom[1].data
        cls_score = bottom[0].data  # b,18,h,w
        bbox = bbox.reshape(16, 4, -1)
        cls_score = cls_score.reshape(16, 2, -1)
        count_fore = 0
        proposal_mean = 0
        total_samples = np.size(bbox).sum([0, 2])

        for ind in range(16):

            fore = cls_score[ind, 0, :]
            back = cls_score[ind, 1, :]
            total_l = np.size(fore)[0]


            for i in range(total_l):
                self.count_sample += 1
                if fore[i] >= back[i]:
                    count_fore += 1
                    bbox_ind = bbox[ind, :, i]
                    count_proposal_size = abs((bbox_ind[0] - bbox_ind[2]) * (bbox_ind[1] - bbox_ind[3]))
                    proposal_mean += count_proposal_size

        self.count_sample[1] = (self.count_sample[1] * (self.count_sample - total_samples) + total_samples * proposal_mean)/(self.count_sample)
        self.count_sample[0] += count_fore
        print(self.count_sample)  # debug

    def backward(self, top, propagate_down, bottom):
        pass