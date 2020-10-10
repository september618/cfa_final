import caffe
import numpy as np
import cv2


class WeightslabelLayer(caffe.Layer):  # Another version of edited_cls_aware to avoid overfitting
    # bottom[0]:"pool5"
    # bottom[1]: "labels"
    def __init(self):
        self.alpha = 0.05  # alpha is to avoid the loss is always negative
        self.num_cls = 21

    def change_pos(self, bottom):
        #if self._start != False:
        #    self._start = False
        #    print("initializing")
        self.memory = [np.zeros_like(bottom[0].data[0]) for _ in range(self.num_cls)]
        self.count = [0 for _ in range(self.num_cls)]

    def set_start(self):
        self._start = True

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):

        # label output
        top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1])  # score shape: num of channels


    def forward(self, bottom, top):
        self.__init()
        self.change_pos(bottom)

        labels = bottom[1].data
        pool5_feature_pool = bottom[0].data
        score = np.zeros_like(pool5_feature_pool, dtype=np.float32)
        for ind in range(16):
            label = int(labels[ind])
            if label == 0:
                continue
            # print("label",label)
            # print("memory", self.memory[3])
            for i in list(range(1, label)) + list(range(label + 1, self.num_cls)):
                score[ind] += self.alpha * ((pool5_feature_pool[ind] - self.memory[i]) ** 2)
            score[ind] -= (pool5_feature_pool[ind] - self.memory[label]) ** 2

            #print(pool5_feature[ind])

            self.memory[label] = (self.memory[label] * self.count[label] + pool5_feature_pool[ind]) / (self.count[label] + 1)
            self.count[label] += 1

            '''
            if label==7:
                #print("feature",pool5_feature[ind][127])
                #print("memory",self.memory[7][127])
                print("arg7",np.argsort(np.sum(score[ind],axis=1)))
                print("score7",np.sort(np.sum(score[ind],axis=1)))

                class1_feature = np.sum(score[ind],axis=1)
                class1_feature -= np.min(class1_feature)
                class1_feature /= (np.max(class1_feature) - np.min(class1_feature))
                class1_feature *= 255
                feature = class1_feature.reshape(512, 1, 1)
                blank = 1
                side_length = 23 * 1 + 23 * blank
                merge_image = np.zeros((side_length, side_length), dtype=np.uint8)
                for i in range(23):
                    for j in range(23):
                        if 23 * i + j >= 512:
                            break
                        merge_image[1 * i + blank * i:1 * (i + 1) + blank * i,
                        1 * j + blank * j:1 * (j + 1) + blank * j] = \
                            feature[23 * i + j, :, :]
                # merge_image*=255
                merge_image = cv2.resize(merge_image, (side_length * 35, side_length * 35))
                cv2.imwrite("../single.jpg", merge_image)
                #print("score7",np.sort(np.sum(score[ind],axis=1)))
                #print("labelself",np.argsort(np.sum((pool5_feature[ind] - self.memory[label]) ** 2,axis=1)))
                #print("labelself_DAXIAO", np.sort(np.sum((pool5_feature[ind] - self.memory[label]) ** 2, axis=1)))
                '''

        #print("ori",channel_score)
        #print(channel_score.shape)
        #channel_score-=channel_score.mean(axis=1,keepdims=True)
        #print("channel_score",channel_score)
        score=np.sum(score,axis=2)
        score-=score.mean(axis=1,keepdims=True)
        top[0].data[...] = score/40
        #print("top0",top[0].data)

        '''
        if self.count[12] >= 30:
            class1_feature = self.memory[12]
            class1_feature -= np.min(class1_feature)
            class1_feature /= (np.max(class1_feature) - np.min(class1_feature))
            class1_feature *= 255
            feature = class1_feature.reshape(512, 1, 1)
            blank = 1
            side_length=23 * 1 + 23 * blank
            merge_image = np.zeros((side_length, side_length), dtype=np.uint8)
            for i in range(23):
                for j in range(23):
                    if 23 * i + j >= 512:
                        break
                    merge_image[1 * i + blank * i:1 * (i + 1) + blank * i, 1 * j + blank * j:1 * (j + 1) + blank * j] = \
                    feature[23 * i + j,:,:]
            #merge_image*=255
            merge_image=cv2.resize(merge_image,(side_length*35,side_length*35))
            cv2.imwrite("../mergedog.jpg",merge_image)
        '''

        '''
        if self.count[7] >= 1000:
            class1_feature = self.memory[7]
            class1_feature -= np.min(class1_feature)
            class1_feature /= (np.max(class1_feature) - np.min(class1_feature))
            class1_feature *= 255
            feature = class1_feature.reshape(256, 6, 6)
            blank = 1
            side_length=16 * 6 + 16 * blank
            merge_image = np.zeros((side_length, side_length), dtype=np.uint8)
            for i in range(16):
                for j in range(16):
                    if 16 * i + j >= 256:
                        break
                    merge_image[6 * i + blank * i:6 * (i + 1) + blank * i, 6 * j + blank * j:6 * (j + 1) + blank * j] = \
                    feature[16 * i + j,:,:]
            #merge_image*=255
            merge_image=cv2.resize(merge_image,(side_length*5,side_length*5))
            cv2.imwrite("../merge.jpg",merge_image)
        '''

    def backward(self, top, propagate_down, bottom):
        pass
