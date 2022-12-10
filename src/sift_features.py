from skimage.feature import SIFT
import numpy as np


class SIFTFeatures:
	def __init__(self):
		# store the number of points and radius
		self.descriptor = SIFT()

	def extract(self, image):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		self.descriptor.detect_and_extract(image)
		return self.descriptor.keypoints, self.descriptor.descriptors

	