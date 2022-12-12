import os
import config
import numpy as np
import matplotlib.pyplot as plt
from locally_binary_pattern import LocalBinaryPatterns
from sift_features import SIFTFeatures
from skimage.feature import SIFT, match_descriptors, plot_matches
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import filters
from skimage.measure import find_contours
from PIL import ImageOps
# import cv2 as cv

class DataLoader:

    def __init__(self):
        self.desc = LocalBinaryPatterns(24, 8)

    def read_dataset(self):
        self.dataset = []
        for folder in os.listdir(config.DATASET_PATH):
            vid = []
            for file in os.listdir(os.path.join(config.DATASET_PATH, folder)):
                vid.append(os.path.join(config.DATASET_PATH, folder, file))
            vid.sort()
            self.dataset.append([(vid[i], vid[i+1], vid[i+2], vid[i+3], vid[i+4], ) for i in range(0, len(vid)-1, 5)])
        
    
    def shuffle_dataset(self):
        import random
        blocksize = 10
        for video in self.dataset:
            # shuffle blocks
            # Create blocks
            blocks = [video[i:i+blocksize] for i in range(0,len(video),blocksize)]
            # shuffle the blocks
            random.shuffle(blocks)
            # concatenate the shuffled blocks
            video[:] = [b for bs in blocks for b in bs]


    def read_image(self, path):
        return plt.imread(path)
    
    def read_2d_skeleton(self, path):
        skel_file = open(path)
        points_x = []
        points_y = []
        for line in skel_file.readlines():
            line = line.split(',')
            if line[0] == '0.0000':
                break
            points_x.append(float(line[4]))
            points_y.append(float(line[5]))
        skel_file.close()
        return np.array([points_x, points_y]) * 2

    def read_3d_skeleton(self, path):
        skel_file = open(path)
        points_x = []
        points_y = []
        points_z = []
        for line in skel_file.readlines():
            line = line.split(',')
            if line[0] == '0.0000':
                break
            points_x.append(float(line[4]))
            points_y.append(float(line[5]))
            points_z.append(float(line[6]))
        skel_file.close()
        return np.array([points_x, points_y, points_z]) * 2
    

    def crop_face(self, img, skel_2d):
        if skel_2d.shape[1] < 1:
            return img
        x = skel_2d[0][2]
        y = skel_2d[1][2]
        return img[int(y) - 100:int(y) + 30, int(x)-70:int(x) + 70, :]
    
    def extract_bbox(self, usmp): 
        contours = find_contours(usmp, 0.8)
        if len(contours) > 0:
            contour = contours[0]
            Xmin = np.min(contour[:,0]) * 2
            Xmax = np.max(contour[:,0]) * 2
            Ymin = np.min(contour[:,1]) * 2
            Ymax = np.max(contour[:,1]) * 2
            return int(Xmin), int(Xmax), int(Ymin), int(Ymax)
        else:
            return 0, usmp.shape[0], 0, usmp.shape[1] # Xmin, Xmax, Ymin, Ymax

    def read_ldp(self, usmp_path, img):
        usmp = self.read_image(usmp_path)
        usmp = np.pad(usmp, [(10, 0), (10, 0)], mode='constant')
        person = rgb2gray(img) * resize(usmp, (img.shape[0], img.shape[1]))
        bbox = self.extract_bbox(usmp)
        return self.desc.describe(person[bbox[0]:bbox[1], bbox[2]:bbox[3]])

    def read_frame(self, frame):
        img = self.read_image(frame[0])
        skel_2d = self.read_2d_skeleton(frame[3])
        return img, skel_2d, self.read_3d_skeleton(frame[3]), self.read_ldp(frame[2], img), self.crop_face(img, skel_2d)
    
    def iterate(self):
        for video in self.dataset:
            for frame in video:
                yield self.read_frame(frame)


    def display_dataset(self):
        img = None
        lbp_img = None
        rects = None
        face = None

        figure, axis = plt.subplots(1, 4)
        points, = axis[0].plot([], [], 'ro')

        for im, skel_2d, skel_3d, lbp, face_crop in d.iterate():
            points.set_data(skel_2d[0], skel_2d[1])
            if img is None:
                img = axis[0].imshow(im)
                lbp_img = axis[1].imshow(lbp[0])
                face = axis[2].imshow(face_crop)
            else:
                img.set_data(im)
                lbp_img.set_data(lbp[0])
                face.set_data(face_crop)
            hist = lbp[1]
            if rects is None:
                rects = axis[3].bar([_ for _ in range(len(hist))], hist)
            else:
                for rect,h in zip(rects,hist):
                    rect.set_height(h)
            plt.pause(.01)
            plt.draw()
    
if __name__ == "__main__":
    d = DataLoader()
    d.read_dataset()
    d.shuffle_dataset()
    d.display_dataset()