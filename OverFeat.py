import overfeat
import numpy
from scipy.ndimage import imread
from scipy.misc import imresize
import pdb

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse

# read image
class OverFeat(object):
	def __init__(self, height, width, channels, network, debug):
		self.height = height
		self.width = width
		self.channels = channels
		self.allFeatures = []
		if (network == "fast"):
			self.network = 0
		elif (network == "accurate"):
			self.network = 1
		self.debug = debug

	def initialize(self):
		# initialize overfeat. Note that this takes time, so do it only once if possible
		if self.debug == True:
			pdb.set_trace()
		overfeat.init('../../data/default/net_weight_'+str(self.network), self.network)

	def runForImage(self, input_img):
		image = imread(input_img)
		if len(image.shape) == 2:
			return 0
		h0 = image.shape[0]
		w0 = image.shape[1]
		d0 = float(min(h0, w0))
		if len(image.shape) == 2:
			pdb.set_trace()
			image = image[int(round((h0-d0)/2.)):int(round((h0-d0)/2.)+d0),
		              int(round((w0-d0)/2.)):int(round((w0-d0)/2.)+d0)]
		elif len(image.shape)== 3:
			image = image[int(round((h0-d0)/2.)):int(round((h0-d0)/2.)+d0),
		              int(round((w0-d0)/2.)):int(round((w0-d0)/2.)+d0),:]
		image = imresize(image, (self.height, self.width)).astype(numpy.float32)
		# numpy loads image with colors as last dimension, transpose tensor
		h = image.shape[0]
		w = image.shape[1]
		if len(image.shape) == 2:
			c = 1
			return
		elif len(image.shape)== 3:
			#pdb.set_trace()
			c = image.shape[2]	
		image = image.reshape(w*h, c)
		image = image.transpose()
		image = image.reshape(c, h, w)
		#print "Image size :", image.shape
		# run overfeat on the image
		self.outputAtLastLayer = overfeat.fprop(image)
		return 1
# resize and crop into a 231x231 image
	
	def getFeatures(self, layer_number):
		self.featuresAtLayer = overfeat.get_output(layer_number)
		self.allFeatures.append(self.featuresAtLayer)
		#pdb.set_trace()
		if self.debug == True:
			pdb.set_trace()
		#print "featuresAtLayer size :", self.featuresAtLayer.shape

	def getTopClasses(self, numTopClasses):
		self.outputAtLastLayer = self.outputAtLastLayer.flatten()
		top = [(self.outputAtLastLayer[i], i) for i in xrange(len(self.outputAtLastLayer))]
		#pdb.set_trace()
		top.sort()
		topNew = []
		#print "\nTop classes :"
		for i in xrange(numTopClasses):
			topNew.append(top[-(i+1)][1])
			#print(overfeat.get_class_name(top[-(i+1)][1]))
		#pdb.set_trace()
		return topNew

	def getTopClassesNames(self, numTopClasses):
		self.outputAtLastLayer = self.outputAtLastLayer.flatten()
		top = [(self.outputAtLastLayer[i], i) for i in xrange(len(self.outputAtLastLayer))]
		#pdb.set_trace()
		top.sort()
		#topNew = []
		print "\nTop classes :"
		for i in xrange(numTopClasses):
			#topNew.append(top[i])
			print(overfeat.get_class_name(top[-(i+1)][1])) + '/t' + str(top[-(i+1)][1])
		#pdb.set_trace()
		#return topNew


	#overfeatObj.sampleLayer(overfeatObj.)
	#overfeatObj.buildNetwork(1)
