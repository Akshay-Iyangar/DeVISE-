import overfeat
import numpy
from scipy.ndimage import imread
from scipy.misc import imresize
import pdb

#import theano
#import theano.tensor as T
#from theano.tensor.signal import downsample
#from theano.tensor.nnet import conv
#from theano.ifelse import ifelse

import ConvNeuNet
import OverFeat
from datetime import datetime
import scipy.io as sio

overfeatObj = OverFeat.OverFeat(	height 			= 256,
									width 			= 256,
									channels		= 3,
									network 		= "accurate",
									debug 			= False)
overfeatObj.initialize()
output = []
features = []
tstart = datetime.now()
numOfImagesToConsider = 500000
numOfImagesCovered = 0
tstart = datetime.now()
text_file = open("/mnt/parent/vijetha/cuda-convnet2-master/make-data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt", "r")
lines = text_file.read().split(',')
#print lines
#print len(lines)
#pdb.set_trace()
text_file.close()
#pdb.set_trace()
linesNew = []
labelstoconsider = [712,362,379,405,384,331,326,319,320,295,265,238,237,229,230,231,232,261,542,183,138,220,221,225,224,233,228,264,324,322,323,450,502,499,516,531,532,533,535,556,560,550,569,582,585,587,591,594,631,632,634,653,694,730,749,755,789,799,820,824,870,879,884,939,944,936,984,997,890,839,827,504,420,270,508,526,547,565,580,599,661,685,686,706,711,704,705,695,640,639,647,667,668,738,737,757,729,675,676,677]
for i in xrange(numOfImagesToConsider):
	#print str(i)
	#pdb.set_trace()
	if int(lines[i]) in labelstoconsider:
		overfeatObj = OverFeat.OverFeat(	height 			= 256,
									width 			= 256,
									channels		= 3,
									network 		= "accurate",
									debug 			= False)
		overfeatObj.initialize()
		ret = overfeatObj.runForImage(input_img 	= '/mnt/parent/vijetha/cuda-convnet2-master/make-data/ILSVRC2012_img_val/ILSVRC2012_val_000'+'{:0>5d}'.format(i+1)+'.JPEG')	
		if ret == 1:
			overfeatObj.getFeatures(21)
			features.append(overfeatObj.featuresAtLayer)
			#pdb.set_trace()
			numOfImagesCovered  = numOfImagesCovered + 1
			linesNew.append(lines[i])
			print "image number " + str(i)

sio.savemat('featuresimagenet_'+str(i)+'.mat', {'featuresimagenet':features})
sio.savemat('labelsimagenet_'+str(i)+'.mat',{'labelsimagenet':linesNew})

