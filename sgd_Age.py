import numpy as np

import time
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from scipy.spatial import distance
import csv
from scipy import interp
from itertools import cycle
import warnings
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV


warnings.filterwarnings("ignore")
try:
  from cPickle import dumps, loads
except ImportError:
  from pickle import dumps, loads


def s_load(file_obj):
  '''load contents from file_obj, returning a generator that yields one
  element at a time'''
  cur_elt = []
  for line in file_obj:
    cur_elt.append(line)

    if line == '\n':
      pickled_elt_str = ''.join(cur_elt)
      elt = loads(pickled_elt_str)
      cur_elt = []
      yield elt

def s_dump(iterable_to_pickle, file_obj):
  '''dump contents of an iterable iterable_to_pickle to file_obj, a file
  opened in write mode'''
  for elt in iterable_to_pickle:
    s_dump_elt(elt, file_obj)


def s_dump_elt(elt_to_pickle, file_obj):
  '''dumps one element to file_obj, a file opened in write mode'''
  pickled_elt_str = dumps(elt_to_pickle)
  file_obj.write(pickled_elt_str)
  # record separator is a blank line
  # (since pickled_elt_str might contain its own newlines)
  file_obj.write('\n\n')



def genderTest():
	
	#X,y = s_load(open('MultiPie_Validation_XY.spkl','rb'))
	f1 = open('SGDCompareResExpressionMultiPie.txt','w+')

	dfp = 'MultiPie_Expression_Validation_XY.spkl'
	count = 0
	#print y
	
	clf = SGDClassifier(loss="log", penalty="l2")
	for fu in s_load(open(dfp)):
		#print fu
		print 'len fu'
		print len(fu)
		#for i in range(len(fu)):
		if count <=30:	
			trainX=fu[0]
			trainY=fu[1]
			print trainY
			clf.partial_fit(trainX,trainY,[0,1])
		if count >30:
			break
		count +=1
		#count = 0
	
	joblib.dump(clf,open('expression_multi_pie_sgd_log.p','wb'))
	
	
	'''
	clf = joblib.load(open('gender_multi_pie_sgd_log.p', 'rb'))
	preY = []
	testX =[]
	preY_all = []
	testY_all = []
	count = 0
	for fu in s_load(open(dfp)):
		print count
		if count >=30:
			testX=fu[0]
			testY=fu[1]		
			#probX = clf.predict_proba(testX)
			preY= clf.predict(testX)		
			temp_prec = sum(preY == testY)*1.0/len(preY)
			print 
			#print>>f1,probX
			print>>f1,testY 
			print>>f1,temp_prec
			preY_all.extend(preY)
			testY_all.extend(fu[1])
			
		count +=1
		if count >= 50:
			break
		
		

	print>>f1,testY_all
	print>>f1,preY_all
	prec = sum(preY_all== testY_all)*1.0/len(preY_all)

	print>>f1,prec
	'''
		
	'''	
	tn = time.time()
	probX = clf.predict_proba(testX)
	preY = clf.predict(testX)
	et = time.time() - tn
	print probX
	
	#testY, probX[:,1]
	
	#storing the images along with probability values in the validation dataset.
	imageNumbers = range(0,len(testY))
	print imageNumbers
	with open('sgd_imageProbabilities.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(imageNumbers,probX[:,1])
		for row in rows:
			writer.writerow(row)
	
	#calculation and plot of roc_auc for male
	totalMale = testY.count(1)
	totalNotMale = testY.count(0)
	print totalMale
	print totalNotMale
	#totalMale = sum(testY==1)*1.0
	#totalNotMale = sum(testY==0)*1.0
	roc_auc = dict()
	fpr, tpr, thresholds = roc_curve(testY, probX[:,1], pos_label=1)
	roc_auc = auc(fpr, tpr)	
	print>>f1,roc_auc
	print>>f1,fpr
	print>>f1,tpr
	print>>f1,'thresholds'
	print>>f1,thresholds
	
	
	#storing the threshold value along with the tpr and fpr values.
	with open('sgd_threhsolds.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(thresholds,tpr,fpr)
		for row in rows:
			writer.writerow(row)
	   
	#pickle.dump([fpr,tpr,thresholds],open('rf_threshold.p','wb'))
	
	print>>f1,'total detection'
	print>>f1,fpr+tpr
		
		
	#choosing the best threshold on roc curve	
	mindist = 100;
	minI=0;
	for i in range(len(fpr)):
		a = np.array((fpr[i],tpr[i]))
		b = np.array((0,1))
		dist_a_b = distance.euclidean(a,b)
		if dist_a_b < mindist:
			mindist = dist_a_b
			minI =i
			minX = fpr[minI]
			minY = tpr[minI]
			threshold = thresholds[minI]
			
	print>>f1, 'minX :%f,  minY: %f, mindist:%f, Threshold = %f, minI =%d , fpr[min]=%f, tpr[min]=%f, false detection value= %f, true detection value =%f '%(minX,minY,mindist,threshold,minI, fpr[minI],tpr[minI], fpr[minI]*totalNotMale,tpr[minI]*totalMale)

	#storing the selectivity and accuracy values 
	
	prec = sum(preY == testY)*1.0/len(preY)
	select_1 = sum(preY==1)*1.0/len(preY) #Male
	select_2 = sum(preY==0)*1.0/len(preY) #Female
	f1.write('Gender time: %f acc. %f select_1 %f preY:%f testY:%f testX:%f select_2 %f\n'%\
	(et/len(preY),prec,select_1,len(preY),len(testY),len(testX),select_2))
	list1, list2 = (list(x) for x in zip(*sorted(zip(probX[:,1], testY), key=lambda pair: pair[0])))
	
	print list1
	print list2
	# lower correct is for low threshold
	lowerCorrect = (list2.index(1))
	higherCorrect = ((list2[::-1].index(0) + 1) -1)
	yesAccuracy = float(higherCorrect)/ len(list2)
	noAccuracy = float(lowerCorrect) / len(list2)
	
	print 'higherCorrect : %d, yesAccuracy : %f,  lowerCorrect: %d, noAccuracy: %f'%(higherCorrect,yesAccuracy,lowerCorrect,noAccuracy)  
	print 'lower threshold: %f ,   upperThreshold: %f'%(list1[lowerCorrect],list1[len(list1)-higherCorrect])
	print>>f1,'higherCorrect : %d, yesAccuracy : %f,  lowerCorrect: %d, noAccuracy: %f'%(higherCorrect,yesAccuracy,lowerCorrect,noAccuracy)  
	print>>f1,'lower threshold: %f ,   upperThreshold: %f'%(list1[lowerCorrect],list1[len(list1)-higherCorrect])
	
	for prob in (probX):
		print>>f1, prob
	print>>f1, 'predicted value'
	for pred in preY :
		print>>f1, pred
	print>>f1, 'True value'
	for truth in testY :
		print>>f1, truth
		
	for sortedProb in list1 :
		print>>f1, sortedProb
	for sortedTruth in list2 :
		print>>f1, sortedTruth
	
	'''

	f1.flush()
		

if __name__ =='__main__':

	genderTest()
	#f1.close()