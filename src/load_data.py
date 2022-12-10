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
from PIL import ImageOps
# import cv2 as cv

# read folters
def read_dataset():
    dataset = []
    for folder in os.listdir(config.DATASET_PATH):
        vid = []
        for file in os.listdir(os.path.join(config.DATASET_PATH, folder)):
            vid.append(os.path.join(config.DATASET_PATH, folder, file))
        vid.sort()

        dataset.append([(vid[i], vid[i+1], vid[i+2], vid[i+3], vid[i+4], ) for i in range(0, len(vid)-1, 5)])
    
    return dataset

def shuffle_dataset(dataset):
    import random
    blocksize = 10
    for video in dataset:
        # shuffle blocks
        # Create blocks
        blocks = [video[i:i+blocksize] for i in range(0,len(video),blocksize)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        video[:] = [b for bs in blocks for b in bs]
    return dataset

def read_image(path):
    return plt.imread(os.path.join(config.DATASET_PATH, path))


def display_dataset(dataset, desc, sift):
    # create two subplots
    figure, axis = plt.subplots(1, 6)
    # for each video
    points, = axis[0].plot([], [], 'ro')
    features, = axis[5].plot([], [], 'bo')

    c = 0
    print(len(dataset))
    for video in dataset:
        if c < 10:
            c += 1
            continue
        img = None
        depth = None
        user_map = None
        ldp = None
        rects = None
        suf_img = None
        # for each frame
        for frame in video:
            # read skeleton, image, depth and user_map
            skel_file = open(frame[3])
            points_x = []
            points_y = []
            for line in skel_file.readlines():
                line = line.split(',')
                if line[0] == '0.0000':
                    break
                points_x.append(float(line[4]) * 2)
                points_y.append(float(line[5]) * 2)
            skel_file.close()
            points.set_data(points_x, points_y)

            im = plt.imread(frame[0])
            if img is None:
                img = axis[0].imshow(im)
            else:
                img.set_data(im)
            
            dep = plt.imread(frame[1])
            # mean normalization for depth
            dep = (dep - dep.min()) / (dep.max() - dep.min())
            if depth is None:
                depth = axis[1].imshow(dep)
            else:
                depth.set_data(dep)
            
            usmp = plt.imread(frame[2])
            if user_map is None:
                user_map = axis[2].imshow(usmp)
            else:
                user_map.set_data(usmp)



            
            print(im.shape)
            print(usmp.shape)
            gray_im = rgb2gray(im)
            usmp = np.pad(usmp, [(10, 0), (10, 0)], mode='constant')
            person = gray_im * resize(usmp, (im.shape[0], im.shape[1]))

            feature = desc.describe(person)
            if ldp is None:
                ldp = axis[3].imshow(feature)
            else:
                ldp.set_data(feature)
            
            hist = desc.histogram(person)
            if rects is None:
                rects: plt.BarContainer = axis[4].bar([_ for _ in range(len(hist))], hist)
            else:
                for rect,h in zip(rects,hist):
                    rect.set_height(h)
            
            if suf_img is None:
                suf_img = axis[5].imshow(person)
            else:
                suf_img.set_data(person)



                        # SIFT is very slow. I removed it

            # if suf_img is None:
            #     suf_img = axis[5].imshow(filters.gaussian(feature, 20))
            # else:
            #     suf_img.set_data(filters.gaussian(feature, 20))
            
            # keypoints_surf, descriptors = sift.extract(filters.gaussian(feature, 20))
            # print(keypoints_surf.shape)
            # features.set_data(keypoints_surf[:,1], keypoints_surf[:,0])

            
            plt.pause(.01)
            plt.draw()


d = read_dataset()
d = shuffle_dataset(d)
desc = LocalBinaryPatterns(10, 10)
sift = SIFTFeatures()
display_dataset(d, desc, sift)