import caffe
import numpy as np


class stat(caffe.Layer):  # Another version of edited_cls_aware to avoid overfitting
    def __init(self):
        self.alpha = 0.05  # alpha is to avoid the loss is always negative
        self.num_cls = 21
        self.count_sample = 0

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):

        # 0 cls, 1 bbox
#        top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1]) 
        pass

    def change_pos(self, bottom):
        f=open("./cd.txt", 'r+')
        line=f.readline()
        if line.split()[0] == 'te':
            ff=open("./cd.txt", 'w') 
            ff.write("zsa")
            print("initializing")
            self.stat = [0,0]

    def forward(self, bottom, top):
        self.__init()
        self.change_pos(bottom)
        bbox = bottom[1].data
        cls_score = bottom[0].data  # b,18,h,w
        #bbox = bbox.reshape(16, 4, -1)
        #cls_score = cls_score.reshape(16, 2, -1)
        bbox = bbox.reshape(1, 4, -1)
        cls_score = cls_score.reshape(1, 2, -1)
        count_fore = 0
        proposal_total = 0
        #total_samples = np.size(bbox).sum([0, 2])
        total_samples = np.size(bbox)/4
        for ind in range(1):
            fore = cls_score[ind, 0, :]
            back = cls_score[ind, 1, :]
            #total_l = np.size(fore)[0]
            total_l = fore.shape[0]
            for i in range(total_l):
                if fore[i] >= back[i]:
                    self.count_sample += 1
                    count_fore += 1
                    bbox_ind = bbox[ind, :, i]
                    count_proposal_size = abs((bbox_ind[0] - bbox_ind[2]) * (bbox_ind[1] - bbox_ind[3]))
                    proposal_total += count_proposal_size

        self.stat[1] = (self.stat[1] * self.stat[0] + proposal_total)/(self.stat[0]+count_fore)  # fore avg size
        self.stat[0] += count_fore  # num of fore
        print(self.stat[1], "Fore Average Size")
        print(self.stat[0], "Total Fore Number")
    def backward(self, top, propagate_down, bottom):
        pass
