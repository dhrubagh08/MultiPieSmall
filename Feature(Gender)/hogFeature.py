import sys
import os,pickle
from skimage.feature import hog
from skimage import io
from skimage import data,color,exposure

def hogx(fileName):
	ri = io.imread(fileName)
	ii = color.rgb2gray(ri)
	fd = hog(ii)
	return fd

'''
try:
  from cPickle import dumps, loads
except ImportError:
  from pickle import dumps, loads




def s_dump(iterable_to_pickle, file_obj):
  dump contents of an iterable iterable_to_pickle to file_obj, a file
  opened in write mode
  for elt in iterable_to_pickle:
    s_dump_elt(elt, file_obj)


def s_dump_elt(elt_to_pickle, file_obj):
  dumps one element to file_obj, a file opened in write mode
  pickled_elt_str = dumps(elt_to_pickle)
  file_obj.write(pickled_elt_str)
  # record separator is a blank line
  # (since pickled_elt_str might contain its own newlines)
  file_obj.write('\n\n')
'''

dirPath = './SampleImages'
#ff = open('MultiPie_Testing6.spkl','w')
imgList = os.listdir(dirPath)
facesList=[]

for imgName in imgList:
	#print imgName
	faceUnit = []
	faceUnit.append(imgName)
	imgPath = os.path.join(dirPath,imgName)
	faceUnit.append(imgPath)
	fea = hogx(imgPath)
	faceUnit.append(fea)
	facesList.append(faceUnit)	
	print 'Finish %f '%(imgList.index(imgName)*1.0/len(imgList))

'''
count = 0	
for i in range(len(imgList)):
	try:
		faceUnit = []
		faceUnit.append(imgList[i])
		#print imgList[i]
		imgPath = os.path.join(dirPath,imgList[i])
		faceUnit.append(imgPath)
		fea = hogx(imgPath)
		faceUnit.append(fea)
		facesList.append(faceUnit)		
		if i%1000 ==0:
			s_dump(facesList,ff)
			facesList=[]
		print 'Finish %f '%(i*1.0/len(imgList))
	except:
		pass
		

s_dump(facesList,ff)
'''
#ff = open('MultiPie_Validation.pkl','wb')
ff = open('MultiPie_Hog_Small_Train_Memory.pkl','wb')
pickle.dump(facesList,ff)
ff.close()

