import numpy as np

import time
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial import distance
import csv
from scipy import interp
from itertools import cycle
import warnings
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

def genderTest():
	f1 = open('DTCompareResGenderMultipie.txt','w+')
	
	trainX,trainY = pickle.load(open('MultiPieTrainGender_XY.p','rb'))
	testX,testY = pickle.load(open('MultiPieValidationGender_XY.p','rb'))
	
		
	
	# initiate PCA and classifier
	pca = PCA()
	
	clf_uncalibrated = tree.DecisionTreeClassifier()
	
	#X_transformed = pca.fit_transform(trainX)
	clf_uncalibrated = clf_uncalibrated.fit(trainX,trainY)
	
	#dl_transformed = pca.transform(dl)
	#clf = clf_uncalibrated
	
	clf = CalibratedClassifierCV(clf_uncalibrated, cv=3, method='sigmoid')
	clf.fit(trainX, trainY)

	joblib.dump(clf,open('gender_multipie_dt_calibrated.p','wb'))
	
	
	
	#clf = joblib.load(open('gender_muct_dt_calibrated.p', 'rb'))
	
	#testX = pca.fit_transform(testX)
	
	tn = time.time()
	preY = clf.predict(testX)
	probX = clf.predict_proba(testX)
	et = time.time() - tn
	print probX
	
	#storing the images along with probability values in the validation dataset.
	imageNumbers = range(0,len(testY))
	print imageNumbers
	with open('dt_gender_imageProbabilities.csv', 'wb') as f:
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
	
	   
	#pickle.dump([fpr,tpr,thresholds],open('rf_threshold.p','wb'))
	
	print>>f1,'total detection'
	print>>f1,fpr+tpr
	
	
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('rocGenderDTMuct.png')	
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
	
	
	prec = sum(preY == testY)*1.0/len(preY)
	select_1 = sum(preY==1)*1.0/len(preY) #Male
	select_2 = sum(preY==0)*1.0/len(preY) #Female
	f1.write('Gender time: %f acc. %f select_1 %f preY:%f testY:%f testX:%f select_2 %f\n'%\
	(et/len(preY),prec,select_1,len(preY),len(testY),len(testX),select_2))
	list1, list2 = (list(x) for x in zip(*sorted(zip(probX[:,1], testY), key=lambda pair: pair[0])))
	
	print list1
	print list2

	for prob in (probX):
		print>>f1, prob
	print>>f1, 'predicted value'
	for pred in preY :
		print>>f1, pred
	print>>f1, 'True value'
	for truth in testY :
		print>>f1, truth
		
	
	f1.flush()
	

if __name__ =='__main__':

	#glassTest()
	genderTest()
	#f1.close()