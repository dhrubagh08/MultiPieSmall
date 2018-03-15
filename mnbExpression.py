import time
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial import distance
import csv
from scipy import interp
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")


from sklearn.calibration import CalibratedClassifierCV

	
def expressionTest():
	
	trainX,trainY = pickle.load(open('MultiPieTrainExpression_XY.p','rb'))
	testX,testY = pickle.load(open('MultiPieValidationExpression_XY.p','rb'))
	
	f1 = open('GnbCompareResExpressionMultiPie.txt','w+')		
	
	
	
	clf_uncalibrated = MultinomialNB()
	clf_uncalibrated = clf_uncalibrated.fit(trainX,trainY)
	
	clf = CalibratedClassifierCV(clf_uncalibrated, cv=3, method='sigmoid')
	clf.fit(trainX, trainY)
	#prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
	pickle.dump(clf,open('expression_multipie_mnb_calibrated.p','wb'))
	
	
	
	tn = time.time()
	preY = clf.predict(testX)
	probX = clf.predict_proba(testX)
	et = time.time() - tn
	print probX
	
	#storing the images along with probability values in the validation dataset.
	imageNumbers = range(0,len(testY))
	print imageNumbers
	with open('mnb_expression_imageProbabilities.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(imageNumbers,probX[:,1])
		for row in rows:
			writer.writerow(row)
	
	#calculation and plot of roc_auc for male
	totalMale = testY.count(1)
	totalNotMale = testY.count(0)
	print totalMale
	print totalNotMale
	roc_auc = dict()
	fpr, tpr, thresholds = roc_curve(testY, probX[:,1], pos_label=1)
	roc_auc = auc(fpr, tpr)	
	print>>f1,roc_auc
	print>>f1,fpr
	print>>f1,tpr
	print>>f1,'thresholds'
	print>>f1,thresholds
	
	
	#storing the threshold value along with the tpr and fpr values.
	with open('mnb_expression_thresholds.csv', 'wb') as f:
		writer = csv.writer(f)	
		rows = zip(thresholds,tpr,fpr)
		for row in rows:
			writer.writerow(row)
	   
	#pickle.dump([fpr,tpr,thresholds],open('rf_threshold.p','wb'))
	'''
	print>>f1,'total detection'
	print>>f1,fpr+tpr
	'''
	
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('rocExpressionMNB.png')	
	plt.show()
	
	
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
	
	f1.flush()
	

if __name__ =='__main__':

	#glassTest()
	expressionTest()
	#f1.close()
