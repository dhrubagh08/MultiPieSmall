import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier



import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial import distance

import csv
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from scipy import interp

from sklearn.calibration import CalibratedClassifierCV




f1 = open('ExtraTreeCompareRes.txt','w+')


	
def glassTest():
	
	trainX,trainY = pickle.load(open('MultiPieTrainGender_XY.p','rb'))
	testX,testY = pickle.load(open('MultiPieValidationGender_XY.p','rb'))
	#print trainY
	#print testY
	f1 = open('ExtraTreeCompareResGlass.txt','w+')	
	
	'''
	clf_uncalibrated = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0)
	clf_uncalibrated = clf_uncalibrated.fit(trainX,trainY)
	
	clf = CalibratedClassifierCV(clf_uncalibrated, cv=3, method='sigmoid')
	clf.fit(trainX, trainY)
	pickle.dump(clf,open('glass_muct_et_calibrated.p','wb'))
	'''
	
	clf=pickle.load(open('glass_muct_et_calibrated.p', 'rb'))

	tn = time.time()
	preY = clf.predict(testX)
	probX = clf.predict_proba(testX)
	et = time.time() - tn
	
	
	#storing the images along with probability values in the validation dataset.
	imageNumbers = range(0,len(testY))
	print imageNumbers
	with open('et_glass_imageProbabilities.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(imageNumbers,probX[:,1])
		for row in rows:
			writer.writerow(row)
	
	#calculation and plot of roc_auc
	
	roc_auc = dict()
	fpr, tpr, thresholds = roc_curve(testY, probX[:,1], pos_label=1)
	roc_auc = auc(fpr, tpr)	
	print>>f1,roc_auc
	print>>f1,fpr
	print>>f1,tpr
	print>>f1,'thresholds'
	print>>f1,thresholds	
	
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('rocGlassET.png')
	plt.show()
	
	
	#choosing the best threshold on roc curve	
	mindist = 100;
	for i in range(len(fpr)):
		a = np.array((fpr[i],tpr[i]))
		b = np.array((0,1))
		dist_a_b = distance.euclidean(a,b)
		if dist_a_b < mindist:
			mindist = dist_a_b
			minX = fpr[i]
			minY = tpr[i]
			threshold = thresholds[i]
			
	print>>f1, 'minX :%f,  minY: %f, mindist:%f, Threshold = %f '%(minX,minY,mindist,threshold)

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
	
	'''
	print 'higherCorrect : %d, yesAccuracy : %f,  lowerCorrect: %d, noAccuracy: %f'%(higherCorrect,yesAccuracy,lowerCorrect,noAccuracy)  
	print 'lower threshold: %f ,   upperThreshold: %f'%(list1[lowerCorrect],list1[len(list1)-higherCorrect])
	print>>f1,'higherCorrect : %d, yesAccuracy : %f,  lowerCorrect: %d, noAccuracy: %f'%(higherCorrect,yesAccuracy,lowerCorrect,noAccuracy)  
	print>>f1,'lower threshold: %f ,   upperThreshold: %f'%(list1[lowerCorrect],list1[len(list1)-higherCorrect])
	'''
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
	
	f1.flush()



	

def genderTest():
	
	trainX,trainY = pickle.load(open('MultiPieTrainGender_XY.p','rb'))
	testX,testY = pickle.load(open('MultiPieValidationGender_XY.p','rb'))
	f1 = open('ExtraTreeCompareResGender.txt','w+')
	
	
	clf_uncalibrated = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0)
	clf_uncalibrated = clf_uncalibrated.fit(trainX,trainY)
	
	clf = CalibratedClassifierCV(clf_uncalibrated, cv=3, method='sigmoid')
	clf.fit(trainX, trainY)
	pickle.dump(clf,open('gender_multi_pie_et_calibrated.p','wb'))

	tn = time.time()
	preY = clf.predict(testX)
	probX = clf.predict_proba(testX)
	et = time.time() - tn
	
	
	#storing the images along with probability values in the validation dataset.
	imageNumbers = range(0,len(testY))
	print imageNumbers
	with open('et_imageProbabilities.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(imageNumbers,probX[:,1])
		for row in rows:
			writer.writerow(row)
	
	#calculation and plot of roc_auc
	
	roc_auc = dict()
	fpr, tpr, thresholds = roc_curve(testY, probX[:,1], pos_label=1)
	roc_auc = auc(fpr, tpr)	
	print>>f1,roc_auc
	print>>f1,fpr
	print>>f1,tpr
	print>>f1,'thresholds'
	print>>f1,thresholds

	#storing the threshold value along with the tpr and fpr values.
	with open('et_threhsolds.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(thresholds,tpr,fpr)
		for row in rows:
			writer.writerow(row)	
	
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('rocGenderET.png')	
	plt.show()
	
	#choosing the best threshold on roc curve	
	mindist = 100;
	for i in range(len(fpr)):
		a = np.array((fpr[i],tpr[i]))
		b = np.array((0,1))
		dist_a_b = distance.euclidean(a,b)
		if dist_a_b < mindist:
			mindist = dist_a_b
			minX = fpr[i]
			minY = tpr[i]
			threshold = thresholds[i]
			
	print>>f1, 'minX :%f,  minY: %f, mindist:%f, Threshold = %f '%(minX,minY,mindist,threshold)
	
	#choosing threshold depending on Youden's index
	
	maxYoudenIndex = 0;
	for i in range(len(fpr)):
		J =tpr[i] -fpr[i]  #J= sensitivity + specificity -1
		if J > maxYoudenIndex:
			maxYoudenIndex = J
			maxX = fpr[i]
			maxY = tpr[i]
			thresholdYouden = thresholds[i]
			
	print>>f1, 'maxX :%f,  maxY: %f, maxYoudenIndex:%f, ThresholdYouden = %f '%(maxX,maxY,maxYoudenIndex,thresholdYouden)
	
	#choosing threshold depending on IsoCost index
	
	cost = 0;  #averageExpectedCost
	print testY
	proportionPositive = sum(testY)/float(len(testY))
	print proportionPositive
	yIntercept = 0.99;
	flag =0;
	maxIntercept = yIntercept
	thresholdIso =0
	maxX= 0 
	maxY=0
	for j in range(100):
		for i in range(len(fpr)):
			if((tpr[i] - proportionPositive * fpr[i] - yIntercept) ==0):
				print 'found it'
				maxIntercept = yIntercept
				maxX = fpr[i]
				maxY = tpr[i]
				thresholdIso = thresholds[i]
		
		yIntercept = yIntercept - j*0.01
			
	print>>f1, 'maxX :%f,  maxY: %f, maxIntercept:%f, thresholdIso = %f '%(maxX,maxY,maxIntercept,thresholdIso)
	

	#storing the selectivity and accuracy values 
	
	prec = sum(preY == testY)*1.0/len(preY)
	select_1 = sum(preY==1)*1.0/len(preY) #Male
	select_2 = sum(preY==0)*1.0/len(preY) #Female
	f1.write('Gender time: %f acc. %f select_1 %f preY:%f testY:%f testX:%f select_2 %f\n'%\
	(et/len(preY),prec,select_1,len(preY),len(testY),len(testX),select_2))
	list1, list2 = (list(x) for x in zip(*sorted(zip(probX[:,1], testY), key=lambda pair: pair[0])))
	
	#print list1
	#print list2
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
	
	f1.flush()

if __name__ =='__main__':

	#glassTest()
	genderTest()
	#genderTestCV()
	#expresTest()
	f1.close()
