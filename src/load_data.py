import os
import config
import numpy as np
import matplotlib.pyplot as plt
from locally_binary_pattern import LocalBinaryPatterns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sift_features import SIFTFeatures
from skimage.feature import SIFT, match_descriptors, plot_matches
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray
from skimage.transform import resize, integral_image
from skimage import filters
from skimage.measure import find_contours
from skimage.feature import haar_like_feature, haar_like_feature_coord, draw_haar_like_feature, hog
from PIL import ImageOps
import dlib
import cv2
#import face_cascade

class DataLoader:

    def __init__(self):
        self.desc = LocalBinaryPatterns(config.NUM_POINTS_LBP, config.RADIUS)

    def read_dataset(self):
        self.dataset = []
        dirs =  os.listdir(config.DATASET_PATH)
        dirs.sort()
        for folder in dirs:
            vid = []
            for file in os.listdir(os.path.join(config.DATASET_PATH, folder)):
                vid.append(os.path.join(config.DATASET_PATH, folder, file))
            vid.sort()
            self.dataset.append([(vid[i], vid[i+1], vid[i+2], vid[i+3], vid[i+4], ) for i in range(0, len(vid)-1, 5)])
        # print(self.dataset)
        
    def shuffle_videos(self):
        import random
        for video in self.dataset:
            # shuffle blocks
            # Create blocks
            blocks = [video[i:i+config.BLOCKSIZE] for i in range(0,len(video),config.BLOCKSIZE)]
            # shuffle the blocks
            random.shuffle(blocks)
            # concatenate the shuffled blocks
            video[:] = [b for bs in blocks for b in bs]
    
    def sample_video(self, video=0, seed = 42, return_images = False, num_samples=50):
        # I have to take one sample from a person and different samples from the others
        # choose a random video:
        images = []
        features = np.zeros((num_samples, 9 + config.NUM_POINTS_LBP + 12))
        import random
        random.seed(seed)
        if video >= len(self.dataset):
            video = random.randint(0, len(self.dataset) - 1)
        i = 0
        while i < num_samples:
            j = random.randint(0, len(self.dataset[video]) - 1)      
            frame = self.dataset[video][j]
            feat = self.frame_to_features(frame)
            if feat is not None:
                features[i,:] = feat
                if return_images:
                    images.append(self.read_image(frame[0]))
                i += 1
                print(i)
        return features, images


    # Todo: funzione che fa un sample random da un video dato in input

    def sample_dataset(self, video=None, seed = 42, return_images = False, file=None):
        # I have to take one sample from a person and different samples from the others
        # choose a random video:
        images = []
        features = np.zeros((config.BLOCKSIZE + config.BLOCKSIZE * config.NUM_SAMPLES, 9 + config.NUM_POINTS_LBP + 12))
        y = np.asarray([1] * config.BLOCKSIZE + [0] * config.BLOCKSIZE * config.NUM_SAMPLES, dtype=np.int)
        import random
        random.seed(seed)
        if video == None or video >= len(self.dataset):
            video = random.randint(0, len(self.dataset) - 1)
        others = []
        i = 0
        while i < config.NUM_SAMPLES * config.BLOCKSIZE:
            v = random.randint(0, len(self.dataset) - 1)
            if v == video:
                v = v + 1 if v < len(self.dataset[v]) - 1 else v - 1

            count = 0
            while count < config.BLOCKSIZE:
                j = random.randint(0, len(self.dataset[v]) - 1)
                frame = self.dataset[v][j]
                feat = self.frame_to_features(frame)
                if feat is not None:
                    features[i + config.BLOCKSIZE,:] = feat
                    if return_images:
                        images.append(self.read_image(frame[0]))
                    i += 1
                    print(i, frame[0])
                    if i + config.BLOCKSIZE >= config.NUM_SAMPLES * config.BLOCKSIZE - 1:
                        break
                    count += 1
                
        i = 0
        while i < config.BLOCKSIZE:
            j = random.randint(0, len(self.dataset[video]) - 1)      
            
            # ret = [self.dataset[video][j:j+config.BLOCKSIZE]] + others
            frame = self.dataset[video][j]
            feat = self.frame_to_features(frame)
            if feat is not None:
                features[i,:] = feat
                if return_images:
                    images.insert(i, self.read_image(frame[0]))
                i += 1
                print(i, "chosen frame", j, "from", frame[0])
        if file is not None:
            np.save(config.SAMPLED_PATH + file + "_X.npy", features)
            np.save(config.SAMPLED_PATH + file + "_y.npy", y)
        return features, y, images

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

    def process_3d_skeleton(self, skel):
        '''
        [[x,x,x,x,x]
         [y,y,y,y,y]
         [z,z,z,z,z]]
        '''
        bust = np.linalg.norm([np.abs(skel[0,0] - skel[0,2]), np.abs(skel[1,0] - skel[1,2]), np.abs(skel[2,0] - skel[2,2])])
        right_upper_arm = np.linalg.norm([np.abs(skel[0,4] - skel[0,5]), np.abs(skel[1,4] - skel[1,5]), np.abs(skel[2,4] - skel[2,5])])
        right_lower_arm = np.linalg.norm([np.abs(skel[0,6] - skel[0,5]), np.abs(skel[1,6] - skel[1,5]), np.abs(skel[2,6] - skel[2,5])])
        left_upper_arm = np.linalg.norm([np.abs(skel[0,8] - skel[0,9]), np.abs(skel[1,8] - skel[1,9]), np.abs(skel[2,8] - skel[2,9])])
        left_lower_arm = np.linalg.norm([np.abs(skel[0,10] - skel[0,9]), np.abs(skel[1,10] - skel[1,9]), np.abs(skel[2,10] - skel[2,9])])
        right_upper_leg = np.linalg.norm([np.abs(skel[0,12] - skel[0,13]), np.abs(skel[1,12] - skel[1,13]), np.abs(skel[2,12] - skel[2,13])])
        right_lower_leg = np.linalg.norm([np.abs(skel[0,14] - skel[0,13]), np.abs(skel[1,14] - skel[1,13]), np.abs(skel[2,14] - skel[2,13])])
        left_upper_leg = np.linalg.norm([np.abs(skel[0,16] - skel[0,17]), np.abs(skel[1,16] - skel[1,17]), np.abs(skel[2,16] - skel[2,17])])
        left_lower_leg = np.linalg.norm([np.abs(skel[0,17] - skel[0,18]), np.abs(skel[1,17] - skel[1,18]), np.abs(skel[2,17] - skel[2,18])])

        return [bust, right_upper_arm, right_lower_arm, left_upper_arm, left_lower_arm, right_upper_leg, right_lower_leg, left_upper_leg, left_lower_leg]


    def frame_to_features(self, frame):
        '''use this to get the features from a frame'''
        features = np.zeros((9 + config.NUM_POINTS_LBP + 12))
        img = self.read_image(frame[0])
        
        skel_2d = self.read_2d_skeleton(frame[3])
        if len(skel_2d[0]) <= 0:
            return None
        face, success = self.crop_face(img, skel_2d)
        face_features =  [None]*12
        if success:
            # face_lbp = self.desc.describe(rgb2gray(face))
            try:
                landmarks = self.extract_landmark_features(face)
                if landmarks is not None:
                    face_features = self.process_face(landmarks)
            except cv2.error as e:
                pass
        else:
            return None
        features[0:9] = self.process_3d_skeleton(self.read_3d_skeleton(frame[3]))
        features[9:9 + config.NUM_POINTS_LBP] = self.read_lbp(frame[2], img)[1]
        features[9 + config.NUM_POINTS_LBP:] = face_features
        return features
    

    def crop_face(self, img, skel_2d):
        if skel_2d.shape[1] < 1 or skel_2d.shape[0] < 1:
            return img, False
        x = skel_2d[0][2]
        y = skel_2d[1][2]
        return img[int(y) - 100:int(y) + 30, int(x)-70:int(x) + 70, :], True
    
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

    def read_lbp(self, usmp_path, img):
        usmp = self.read_image(usmp_path)
        usmp = np.pad(usmp, [(10, 0), (10, 0)], mode='constant')
        person = rgb2gray(img) * resize(usmp, (img.shape[0], img.shape[1]))
        bbox = self.extract_bbox(usmp)
        return self.desc.describe(person[bbox[0]:bbox[1], bbox[2]:bbox[3]])

    def extract_feature_image(self, img, feature_type, feature_coord=None):
        ii = integral_image(img)
        return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                                feature_type=feature_type,
                                feature_coord=feature_coord)
        
    def read_frame(self, frame):
        img = self.read_image(frame[0])
        skel_2d = self.read_2d_skeleton(frame[3])
        face, success = self.crop_face(img, skel_2d)
        ld_features = self.extract_landmark_features(face)
        return img, skel_2d, self.read_3d_skeleton(frame[3]), self.read_lbp(frame[2], img), face, ld_features
    
    def iterate(self):
        return self.read_frame(self.dataset[40][random.randint(0, len(self.dataset[40]) - 1)])
        for video in self.dataset:
            for frame in video:
                yield self.read_frame(frame)


    def display_dataset(self, limit = 1):
        img = None
        lbp_img = None
        rects = None
        face = None
        face_features = []

        figure, axis = plt.subplots(1, 4)
        figure.set_size_inches(18.5, 10.5)
        points, = axis[0].plot([], [], 'ro')
        face_points, = axis[3].plot([], [], 'ro')
        count = 0
        for im, skel_2d, skel_3d, lbp, face_crop, ld in self.iterate():
            if count >= limit:
                break
            points.set_data(skel_2d[0], skel_2d[1])
            if ld is not None:
                face_points.set_data(ld[:, 0], ld[:, 1])
            for i, txt in enumerate(skel_2d[0]):
                axis[0].annotate(i, (skel_2d[0][i], skel_2d[1][i]))
            if img is None:
                img = axis[0].imshow(im)
                lbp_img = axis[1].imshow(lbp[0])
                face = axis[3].imshow(face_crop)
            else:
                img.set_data(im)
                lbp_img.set_data(lbp[0])
                face.set_data(face_crop)
            hist = lbp[1]
            if rects is None:
                rects = axis[2].bar([_ for _ in range(len(hist))], hist)
            else:
                for rect,h in zip(rects,hist):
                    rect.set_height(h)
            plt.draw()
            plt.pause(0.01)  
            count += 1
    

    def extract_landmark_features(self, img):
        # Import the necessary libraries
        
        # Load the pre-trained facial landmark detection model
        predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

        # Load the input image and convert it to grayscale
     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # Use the predictor to detect the facial landmarks in the grayscale image
        dets = dlib.get_frontal_face_detector()
        

        faces = dets(gray)
        if len(faces) <= 0:
            return None
        points_x = []   
        points_y = []    

        # Loop through the detected faces and extract the landmark features
        shape_np = np.zeros((68, 2), dtype="int")
        for face in faces:
            shape = predictor(gray, face)
            #use the np array?
            for i in range(0, 68):
                 shape_np[i] = (shape.part(i).x, shape.part(i).y)
        
        return shape_np

    def process_face(self, face):

        eyes_lenght = np.linalg.norm(face[36] - face[39])
        eyes_width = np.linalg.norm(face[37] - face[38])
        nose_heigth = np.linalg.norm(face[30] - face[33])
        nose_width = np.linalg.norm(face[31] - face[35])
        mouth_width = np.linalg.norm(face[48] - face[54])
        mouth_height = np.linalg.norm(face[51] - face[57])
        chin= np.linalg.norm(face[8] - face[33])
        face_width = np.linalg.norm(face[0] - face[16])
        face_height = np.linalg.norm(face[8] - face[27])
        distance_nose_mouth = np.linalg.norm(face[33] - face[51])
        distance_mouth_chin = np.linalg.norm(face[57] - face[8])
        distance_between_eyes = np.linalg.norm(face[39] - face[42])

        return [eyes_lenght, eyes_width, nose_heigth, nose_width, mouth_width, mouth_height, chin, face_width, face_height, distance_nose_mouth, distance_mouth_chin, distance_between_eyes]



if __name__ == "__main__":
    d = DataLoader()
    d.read_dataset()
    d.shuffle_videos()
    d.display_dataset()