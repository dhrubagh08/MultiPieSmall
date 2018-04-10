import sys,re
import time,pickle
import numpy as np
import heapq
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import random
from operator import truediv,mul,sub
import csv
import math
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import tree
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import copy
import operator
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

f1 = open('QueryExecutionResult.txt','w+')

#dl,nl = pickle.load(open('MuctTestGender6_XY.p','rb'))
'''
dl,nl = pickle.load(open('MultiPieTestGender8_XY.p','rb'))
dl2,nl2 = pickle.load(open('MultiPieTestExpression8_XY.p','rb'))
dl3,nl3 = pickle.load(open('MultiPieTestAge8_XY.p','rb'))
'''

dl4,nl4 = pickle.load(open('MultiPieTestGender8_XY.p','rb'))
dl3,nl3 = pickle.load(open('MultiPieTestExpression8_XY.p','rb'))
dl5,nl5 = pickle.load(open('MultiPieTestAge8_XY.p','rb'))


dl,nl = [],[]

#print>>f1,"gender objects: {}".format(nl)
#print>>f1,"expression objects: {}".format(nl2)

#dl2=[]
dl3=[]
dl5=[]
sys.setrecursionlimit(1500)
print 'loading finished'

gender_gnb = pickle.load(open('gender_multipie_gnb_calibrated.p', 'rb'))
#gender_extraTree = pickle.load(open('gender_multipie_et_calibrated.p', 'rb'))
gender_rf = pickle.load(open('gender_multipie_rf_calibrated.p', 'rb'))
#gender_svm = joblib.load(open('gender_multipie_svm_calibrated.p', 'rb'))


gender_dt = joblib.load(open('gender_multipie_dt_calibrated.p', 'rb'))
gender_knn = joblib.load(open('gender_multipie_knn_calibrated.p', 'rb'))


'''
expression_gnb = pickle.load(open('expression_multipie_gnb_calibrated.p', 'rb'))
#expression_extraTree = pickle.load(open('expression_multipie_et_calibrated.p', 'rb'))
expression_rf = pickle.load(open('expression_multipie_rf_calibrated.p', 'rb'))
#expression_svm = joblib.load(open('expression_multipie_svm_calibrated.p', 'rb'))


expression_dt = joblib.load(open('expression_multipie_dt_calibrated.p', 'rb'))
expression_knn = joblib.load(open('expression_multipie_knn_calibrated.p', 'rb'))
'''


expression_gnb = joblib.load(open('expression_multi_pie_bnb.p', 'rb'))
expression_rf = joblib.load(open('expression_multi_pie_mnb.p', 'rb'))
expression_knn = joblib.load(open('expression_multi_pie_mlp.p', 'rb'))
expression_dt = joblib.load(open('expression_multi_pie_sgd_log.p', 'rb'))



age_dt = joblib.load(open('age_multipie_dt_calibrated.p', 'rb'))
age_gnb = pickle.load(open('age_multipie_gnb_calibrated.p', 'rb'))
age_rf = pickle.load(open('age_multipie_rf_calibrated.p', 'rb'))
age_knn = joblib.load(open('age_multipie_knn_calibrated.p', 'rb'))


rf_thresholds, gnb_thresholds, et_thresholds,  svm_thresholds = [], [], [] , []
rf_tprs, gnb_tprs, et_tprs,  svm_tprs = [], [] ,[], []
rf_fprs, gnb_fprs, et_fprs, svm_fprs  = [], [] ,[], []
rf_probabilities, gnb_probabilities, et_probabilities, svm_probabilities = [], [], [], []

'''
dl = np.array(dl)
nl = np.array(nl)
nl2= np.array(nl2)
'''
#listInitial,list0,list1,list2,list3,list01,list02,list03,list12,list13,list23,list012,list013,list023,list123=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
listInitial_f1,list0_f1,list1_f1,list2_f1,list3_f1,list01_f1,list02_f1,list03_f1,list12_f1,list13_f1,list23_f1,list012_f1,list013_f1,list023_f1,list123_f1=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
listInitial_f2,list0_f2,list1_f2,list2_f2,list3_f2,list01_f2,list02_f2,list03_f2,list12_f2,list13_f2,list23_f2,list012_f2,list013_f2,list023_f2,list123_f2=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
listInitial_f3,list0_f3,list1_f3,list2_f3,list3_f3,list01_f3,list02_f3,list03_f3,list12_f3,list13_f3,list23_f3,list012_f3,list013_f3,list023_f3,list123_f3=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]



def genderPredicate1(rl):
	gProb = gender_gnb.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile

'''	
def genderPredicate2(rl):
	gProb = gender_extraTree.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
'''	
	
def genderPredicate3(rl):
	gProb = gender_rf.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
'''
def genderPredicate4(rl):
	gProb = gender_svm.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
'''	
def genderPredicate6(rl):
	gProb = gender_dt.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate7(rl):
	gProb = gender_knn.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
###### Expression Classifiers############	
def expressionPredicate1(rl):
	gProb = expression_gnb.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
'''	
def expressionPredicate2(rl):
	gProb = expression_extraTree.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
'''	
	
def expressionPredicate3(rl):
	gProb = expression_rf.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
'''
def expressionPredicate4(rl):
	gProb = expression_svm.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass
'''	
	
def expressionPredicate6(rl):
	gProb = expression_dt.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass

def expressionPredicate7(rl):
	gProb = expression_knn.predict_proba(rl)
	gProbGlass = gProb[:,1]
	return gProbGlass


###################################################

####### Age classifiers ###########################	

	
def agePredicate1(rl):
	gProb = age_gnb.predict_proba(rl)
	gProbAge = gProb[:,1]
	return gProbAge

	
def agePredicate3(rl):
	gProb = age_rf.predict_proba(rl)
	gProbAge = gProb[:,1]
	return gProbAge

	
def agePredicate6(rl):
	gProb = age_dt.predict_proba(rl)
	gProbAge = gProb[:,1]
	return gProbAge
	
def agePredicate7(rl):
	gProb = age_knn.predict_proba(rl)
	gProbAge = gProb[:,1]
	return gProbAge
	
	



	
def setupFeature1():
	included_cols = [0]
	skipRow= 0	
	with open('UncertaintyExperiments/Feature1/listInitialDetails.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 1
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				listInitial_f1.append(temp1)
			rowNum = rowNum+1
	
	
	with open('UncertaintyExperiments/Feature1/list0Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list0_f1.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature1/list1Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list1_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list2Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list2_f1.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature1/list3Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list3_f1.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature1/list01Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list01_f1.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature1/list02Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list02_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list03Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list03_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list12Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list12_f1.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature1/list13Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list13_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list23Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list23_f1.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature1/list012Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list012_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list013Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list013_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list023Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list023_f1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list123Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list123_f1.append(temp1)
			rowNum = rowNum+1
			
			
	
	print listInitial_f1	
	print list0_f1
	print list1_f1
	print list2_f1
	print list3_f1
	print list01_f1
	print list02_f1
	print list03_f1
	print list12_f1
	print list13_f1
	print list23_f1
	print list012_f1
	print list013_f1
	print list023_f1
	print list123_f1
	
		
	
def setupFeature2():
	included_cols = [0]
	skipRow= 0	
	with open('UncertaintyExperiments/Feature2/listInitialDetails.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 1
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				listInitial_f2.append(temp1)
			rowNum = rowNum+1
	
	
	with open('UncertaintyExperiments/Feature2/list0Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list0_f2.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature2/list1Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list1_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list2Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list2_f2.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature2/list3Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list3_f2.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature2/list01Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list01_f2.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature2/list02Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list02_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list03Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list03_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list12Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list12_f2.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature2/list13Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list13_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list23Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list23_f2.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature2/list012Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list012_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list013Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list013_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list023Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list023_f2.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature2/list123Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list123_f2.append(temp1)
			rowNum = rowNum+1


def setupFeature3():
	included_cols = [0]
	skipRow= 0	
	with open('UncertaintyExperiments/Feature3/listInitialDetails.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 1
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				listInitial_f3.append(temp1)
			rowNum = rowNum+1
	
	
	with open('UncertaintyExperiments/Feature3/list0Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list0_f3.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature3/list1Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list1_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list2Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list2_f3.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature3/list3Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list3_f3.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature3/list01Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list01_f3.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature3/list02Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list02_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list03Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list03_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list12Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list12_f3.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature3/list13Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list13_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list23Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list23_f3.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature3/list012Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list012_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list013Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list013_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list023Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list023_f3.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature3/list123Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list123_f3.append(temp1)
			rowNum = rowNum+1			

	
def chooseNextBest(prevClassifier,uncertainty,feature):
	#print currentProbability
	#print prevClassifier
	#print uncertainty
	noOfClassifiers = len(prevClassifier)
	uncertaintyList = []
	
	#print prevClassifier
	nextClassifier = -1 
	
	# for objects gone through zero classifiers. This is the initialization stage.
	
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = listInitial_f1
	
	
	# for objects only gone through one classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list0_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list1_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list2_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list3_f1
	
	# for objects gone through two classifiers
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list01_f1
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list02_f1
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list03_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list12_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list13_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list23_f1
	
	# for objects gone through three classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==1) :
		uncertaintyList = list012_f1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list123_f1
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list023_f1
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==1) :
		uncertaintyList = list013_f1
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==1) :
		return ['NA',0]
		
	
	
	# For feature 2	
	# for objects gone through zero classifiers. This is the initialization stage.
	
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = listInitial_f2
	
	# for objects only gone through one classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list0_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list1_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list2_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list3_f2
	
	# for objects gone through two classifiers
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list01_f2
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list02_f2
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list03_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list12_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list13_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list23_f2
	
	# for objects gone through three classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==2) :
		uncertaintyList = list012_f2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list123_f2
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list023_f2
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==2) :
		uncertaintyList = list013_f2
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==2) :
		return ['NA',0]
		
		
	# For Feature 3
	# for objects gone through zero classifiers. This is the initialization stage.
	
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = listInitial_f3
	
	# for objects only gone through one classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list0_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list1_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list2_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list3_f3
	
	# for objects gone through two classifiers
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list01_f3
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list02_f3
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list03_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list12_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list13_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list23_f3
	
	# for objects gone through three classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 and feature ==3) :
		uncertaintyList = list012_f3
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list123_f3
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list023_f3
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 and feature ==3) :
		uncertaintyList = list013_f3
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 and feature ==3) :
		return ['NA',0]	
		
		
	#print 'uncertaintyList'
	#print uncertaintyList
	[nextClassifier,deltaU] = chooseBestBasedOnUncertainty(uncertaintyList, uncertainty)
		
			
	return [nextClassifier,deltaU]		

	

	
def convertEntropyToProb(entropy):
	#print 'entropy: %f'%(entropy)
	for i in range(50):
		f= -0.01*i * np.log2(0.01*i) -(1-0.01*i)*np.log2(1-0.01*i)
		#print f
		if abs(f-entropy) < 0.02:
			#print 0.01*i
			break
	#print 'entropy found: %f'%(0.01*i)
	return 0.01*i
	
	
	

def chooseBestBasedOnUncertainty(uncertaintyList, uncertainty):
	bestClassifier = -1
	index = 0
	#print 'current uncertainty:%f'%(uncertainty)
	#print 'uncertaintyList'
	#print uncertaintyList
	for i in range(len(uncertaintyList)):
		element = uncertaintyList[i]
		if float(element[0]) >= float(uncertainty) :
			index = i
			break
	uncertaintyListElement =  uncertaintyList[index]
	bestClassifier = uncertaintyListElement[1]
	#print bestClassifier
	
	deltaUncertainty = uncertaintyListElement[2]
	#print deltaUncertainty
	
	return [bestClassifier,deltaUncertainty]
	
	
def chooseNextBestBasedOnBlocking(prevClassifier,currentUncertainty,currentProbability):
	miniBlock= []
	print 'inside chooseNextBestBasedOnBlocking'
	# first collecting objects which  are in the same state
	state = 'init'
	stateCollection =[]
	featureVector = []
	#print 'prevClassifier'
	#print prevClassifier
	#print("CurrentProbability: {} ".format(currentProbability))
	
	for i in range(len(prevClassifier)):
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = 'init'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = '0'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = '1'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '2'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '3'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = '01'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '02'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '03'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '12'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '13'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = '23'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '012'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '013'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = '023'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = '123'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = 'NA'
		
		stateCollection.append(state)
	
	
	print("stateCollection: {} ".format(stateCollection))
	

	block0,block1,block2,block3,maxBlock =[],[],[],[],[] # These three variables store information about best block and next best classifier for that block.
	maxNextBestClassifier =''
	deltaUncertainty0, deltaUncertainty1, deltaUncertainty2, deltaUncertainty3, maxDeltaUncertainty = 100,100,100,100,100
	size0,size1,size2,size3 =0,0,0,0
	valMax = sys.float_info.max
	flag = 0
	
	strSet = ['init','0','1','2','3','01','02','03','12','13','23','012','013','023','123']
	for k in range(len(strSet)):
		str = strSet[k]
		subCollection = [i for i, j in enumerate(stateCollection) if j == str]  # it will contain the index of images which have gone through classifier 0 and 1.
		#print subCollection
		#print("state: {} ".format(str))
		#print("subcollection: {} ".format(subCollection))
		if len(subCollection)>0:
			for i in range(len(subCollection)):
				#featureValue = [currentProbability.get(subCollection[i])[0],currentProbability.get(subCollection[i])[1]]
				featureValue = [combineProbability (currentProbability.get(subCollection[i]))]
				#probList = currentProbability.get(subCollection[i])
				#featureValue = [p for p in probList[0] if p !=-1]
				#featureValue = [currentUncertainty[i]]
				#print("featureValue: {} ".format(featureValue))
				featureVector.append(featureValue)
			
			unique_data = [list(x) for x in set(tuple(x) for x in featureVector)]
			uniqueValues = len(unique_data)
			#print("uniqueValues: {} ".format(uniqueValues))
			flag = 1
			
			if uniqueValues >=4:
				#kmeans = KMeans(n_clusters=4, random_state=0).fit(featureVector)
				aggClustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
				aggClustering.fit(featureVector)
				
				#print kmeans.labels_
				#print aggClustering.labels_
				
				
				block0Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 0]
				block1Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 1]
				block2Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 2]
				block3Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 3]
				
				
				'''
				block0Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0]
				block1Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1]
				block2Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 2]
				block3Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 3]
				'''
				
				
				#print("block0Index: {} ".format(block0Index))
				#print("block1Index: {} ".format(block1Index))
				#print("block2Index: {} ".format(block2Index))
				
				block0= [subCollection[x] for x in block0Index]
				block1 = [subCollection[x] for x in block1Index]
				block2 = [subCollection[x] for x in block2Index]
				block3 = [subCollection[x] for x in block3Index]
				
				'''
				block0= [subCollection[x] for x in block0Index]
				block1 = [subCollection[x] for x in block1Index]
				block2 = [subCollection[x] for x in block2Index]
				block3 = [subCollection[x] for x in block3Index]
				'''
				
				'''
				print("block0: {} ".format(block0))
				print("block1: {} ".format(block1))
				print("block2: {} ".format(block2))
				print("MaxBlock: {} ".format(maxBlock))
				'''
				
				size0 = len(block0)
				size1 = len(block1)
				size2 = len(block2)
				size3 = len(block3)
				sizeMax = len(maxBlock)
				
				prevClassifier0 = prevClassifier.get(block0[0])
				prob0 = [combineProbability(currentProbability.get(i)) for i in block0]
				averageProb0= np.mean(prob0)
				averageUncertainty0 = -averageProb0* np.log2(averageProb0) - (1- averageProb0)* np.log2(1- averageProb0)
				
				
				prevClassifier1 = prevClassifier.get(block1[0])
				prob1 = [combineProbability(currentProbability.get(i)) for i in block1]
				averageProb1= np.mean(prob1)
				averageUncertainty1 = -averageProb1* np.log2(averageProb1) - (1- averageProb1)* np.log2(1- averageProb1)
				
				prevClassifier2 = prevClassifier.get(block2[0])
				prob2 = [combineProbability(currentProbability.get(i)) for i in block2]
				averageProb2= np.mean(prob2)
				averageUncertainty2 = -averageProb2* np.log2(averageProb2) - (1- averageProb2)* np.log2(1- averageProb2)
				
				
				prevClassifier3 = prevClassifier.get(block3[0])							
				prob3 = [combineProbability(currentProbability.get(i)) for i in block3]
				averageProb3= np.mean(prob3)
				averageUncertainty3 = -averageProb3* np.log2(averageProb3) - (1- averageProb3)* np.log2(1- averageProb3)
				
				
				[nextBestClassifier0,deltaUncertainty0] = chooseNextBest(prevClassifier0[0],averageUncertainty0)
				[nextBestClassifier1,deltaUncertainty1] = chooseNextBest(prevClassifier1[0],averageUncertainty1)
				[nextBestClassifier2,deltaUncertainty2] = chooseNextBest(prevClassifier2[0],averageUncertainty2)
				[nextBestClassifier3,deltaUncertainty3] = chooseNextBest(prevClassifier3[0],averageUncertainty3)
				
				
				val0 = float(size0)*float(deltaUncertainty0)
				val1 = float(size1)*float(deltaUncertainty1)
				val2 = float(size2)*float(deltaUncertainty2)
				val3 = float(size3)*float(deltaUncertainty3)				
				
			else:
				
				sizeSubCollection = len(subCollection)
				sizeBlock = sizeSubCollection/4
				print sizeBlock
				if(sizeSubCollection > 200):
					#subset= random.choice(subCollection,sizeBlock, replace=False)
					block0 = subCollection[0:sizeBlock]
					#print block0
				else:				
					block0= subCollection
					
				size0 = float(len(block0))
				prevClassifier0 = prevClassifier.get(block0[0])
				uncertainty0= [currentUncertainty[i] for i in block0]
				averageUncertainty0 = np.mean(uncertainty0)
				[nextBestClassifier0,deltaUncertainty0] = chooseNextBest(prevClassifier0[0],averageUncertainty0)
				
				
				val0 = float(size0)*float(deltaUncertainty0)
				val1=0
				val2=0
				val3=0
				sizeMax = float(len(maxBlock))
				#if flag !=0 and sizeMax !=0:
					#valMax = float(sizeMax)*float(maxDeltaUncertainty)
				'''
				val0 = float(deltaUncertainty0)*cost(nextBestClassifier0)
				val1 = 0
				val2 = 0
				if maxNextBestClassifier !='':
					valMax = float(maxDeltaUncertainty)/cost(maxNextBestClassifier)
				'''
			#print 'deltaUncertainty0'
			#print 'minval for state:%s is :%f'%(state,min(val0,val1,val2))
			if(min(val0,val1,val2,val3) < valMax):
			#if(min(val0,val1,val2) < valMax):
				if val0 < val1 and val0 < val2 and val0<val3:
				#if val0 < val1 and val0 < val2:
					maxNextBestClassifier = nextBestClassifier0
					maxDeltaUncertainty = deltaUncertainty0
					sizeMax = size0
					maxBlock[:]=[]
					maxBlock = block0[:]
					valMax = val0
					print 'block0 selected for state %s'%(str)
				if val1 < val0 and val1 < val2 and val0<val3:
				#if val1 < val0 and val1 < val2:
					maxNextBestClassifier = nextBestClassifier1
					maxDeltaUncertainty = deltaUncertainty1
					sizeMax = size1
					maxBlock[:]=[]
					maxBlock = block1[:]
					valMax = val1
					print 'block1 selected for state %s'%(str)
				if val2 < val1 and val2 < val0 and val0<val3:
				#if val2 < val1 and val2 < val0:
					maxNextBestClassifier = nextBestClassifier2
					maxDeltaUncertainty = deltaUncertainty2
					sizeMax = size2
					maxBlock[:]=[]
					maxBlock = block2[:]
					valMax = val2
					print 'block2 selected for state %s'%(str)
				
				if val3 < val0 and val3 < val1 and val3<val2:
					maxNextBestClassifier = nextBestClassifier3
					maxDeltaUncertainty = deltaUncertainty3
					sizeMax = size3
					maxBlock[:]=[]
					maxBlock = block3[:]
					valMax = val3
					#print 'block3 selected for state %s'%(str)
				
				block0[:]=[]
				block1[:]=[]
				block2[:]=[]
				block3[:]=[]
			print 'valMax:%f selected for state %s'%(valMax,str)
		subCollection[:] = []
		featureVector[:]=[]
		
		
	return [maxNextBestClassifier,maxBlock]
	
def calculateBlockSize(budget, thinkTime,thinkTimePercent):
	costClassifier = float(cost('GNB')+cost('ET')+cost('RF')+cost('SVM'))/4
	print 'costClassifier:%f'%(costClassifier)
	print 'budget:%f'%(budget)
	print 'thinkTime:%f'%(thinkTime)
	thinkBudget = thinkTimePercent * budget
	numIteration = math.floor(float(thinkBudget)/thinkTime)
	blockSize = (1-thinkTimePercent)*thinkTime/(thinkTimePercent*costClassifier)
	return int(blockSize)
	
def cost(classifier):
	cost=0
	'''
	Cost in Muct Dataset
	gnb,et,rf,svm
	[0.029360,0.018030,0.020180,0.790850]

	'''
	#costSet = [0.029360,0.018030,0.020180,0.790850]
	#print 'classifier'
	#print classifier
	
	if classifier =='ET':
		cost = 0.018030
	if classifier =='SVM':
		cost = 0.790850		
	if classifier =='LDA':
		cost = 0.018175
	if classifier =='DT':
		cost = 0.029360
	if classifier =='GNB':
		cost = 0.018030
	if classifier =='RF':
		cost = 0.020180
	if classifier =='KNN':
		cost = 0.790850
		
		
	return cost
	
def combineProbability (probList):
	sumProb = 0
	countProb = 0
	flag = 0
	#print probList
	weights = determineWeights()
	
	for i in range(len(probList[0])):
		if probList[0][i]!=-1:
			sumProb = sumProb+weights[i]*probList[0][i]
			countProb = countProb+weights[i]
			flag = 1
	
	if flag ==1:
		return float(sumProb)/countProb
	else:
		return 0.5
		
	 

def convertToRocProb(prob,operator):
	#print 'In convertToRocProb method, %f'%prob
	#print operator
	clf_thresholds =[]
	clf_fpr =[]
	if operator.__name__== 'genderPredicate1' :
		clf_thresholds = lr_thresholds
		clf_fpr = lr_fprs
	if operator.__name__== 'genderPredicate2' :
		clf_thresholds = et_thresholds
		clf_fpr = et_fprs
	if operator.__name__== 'genderPredicate3' :
		clf_thresholds = rf_thresholds
		clf_fpr = rf_fprs
	if operator.__name__== 'genderPredicate4' :
		clf_thresholds = ab_thresholds
		clf_fpr = ab_fprs
	if operator.__name__== 'genderPredicate5' :
		clf_thresholds = svm_thresholds
		clf_fpr = svm_fprs
	
	thresholdIndex = (np.abs(clf_thresholds - prob)).argmin()
	rocProb = 1- clf_fpr[thresholdIndex]
	return rocProb
	
	
def findUncertainty(prob):
	if prob ==0 or prob == 1:
		return 0
	else :
		return (-prob* np.log2(prob) - (1- prob)* np.log2(1- prob))
	


def findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3):
	probabilitySet = []
	probDictionary = {}
	for i in range(len(dl)):
		''' For Noisy OR Model
		combinedProbability = 0
		productProbability =1
		'''
		combinedProbability1 = combineProbability(currentProbFeature1[i])	
		combinedProbability2 = combineProbability(currentProbFeature2[i])
		combinedProbability3 = combineProbability(currentProbFeature3[i])
		jointProb = combinedProbability1 * combinedProbability2 * combinedProbability3
		
		probabilitySet.append(jointProb)
		
		key = i
		value = jointProb
		probDictionary[key] = [value]
	#probabilitySet.sort(reverse=True)
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	#x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
	sorted_x = sorted(probDictionary.items(), key=operator.itemgetter(1), reverse = True)
	
	#print 'sorted_x'
	#print sorted_x
	#print 'probabilitySet'
	#print probabilitySet
	#print("probDictionary: {} ".format(sorted_x))
	#print("sorted probabilitySet: {} ".format(sortedProbSet))
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability =0
		for j in range(i):
			#probThreshold = sorted_x.get(j)
			sumOfProbability = sumOfProbability + sortedProbSet[j]   #without dictionary
			#sumOfProbability = sumOfProbability + sorted_x.get(j)
		if i>0:
			precision = float(sumOfProbability)/(i)
			recall = float(sumOfProbability)/totalSum
			f1Value = 2*precision*recall/(precision+recall)
			#f1Value = 2*float(sumOfProbability)/(totalSum +i)
		#print 'precision Value: %f'%(precision)
		#print 'recall Value: %f'%(recall)
		#print 'f1Value: %f'%(f1Value)
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print 'indexSorted value : %d'%(indexSorted)
	
	returnedImages = []
	'''
	for j in range(indexSorted):
		#probValue = sortedProbSet[j]
		#indexProbabilitySet = [i for i, x in enumerate(probabilitySet) if x == probValue]
		indexProbabilitySet = [k for k in range(len(probabilitySet)) if probabilitySet[k] == probValue]
		#returnedImages.append(indexProbabilitySet)
		#sorted_x.get(j)
	'''
	
	for key in sorted_x[:indexSorted]:
		returnedImages.append(key[0])
	
	# this part is to eliminate objects which have not gone through any of the classifiers.
	eliminatedImage = []
	for k in range(len(returnedImages)):
		flag1 = 0
		for p in currentProbFeature1[returnedImages[k]][0]:
				if p!=-1 :
					flag1 = 1
		for p in currentProbFeature2[returnedImages[k]][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			eliminatedImage.append(returnedImages[k])

	selectedImages = [x for x in returnedImages if x not in eliminatedImage]
			
	#return [prevF1,precision, recall]
	#return [prevF1,precision, recall, returnedImages]
	return [prevF1,precision, recall, selectedImages]


def determineWeights():
	#set = [0.85,0.92,0.92,0.89]
	#set = [1,2,2,1]
	set = [1,2,2,1]
	sumValue = sum(set)
	weightValues = [float(x)/sumValue for x in set]
	return weightValues
	
	
def findRealF1(imageList):
	sizeAnswer = len(imageList)	
	sizeDataset = len(nl)
	
	#num_ones = (nl==1).sum()
	num_ones=0
	for j in range(len(nl)):
		if (nl[j]==1 and nl2[j] ==1 and nl3[j]==1):
			num_ones+=1
	count = 0
	print 'number of ones=%f'%(num_ones)
	for i in imageList:
		if (nl[i]==1 and nl2[i] ==1 and nl3[i]==1):
			#print 'image number=%d'%(i)
			count+=1
	print 'correct number of ones=%f'%(count)
	
	precision = float(count)/sizeAnswer
	if num_ones >0:
		recall = float(count)/num_ones
	else:
		recall =0
	if precision !=0 and recall !=0:
		f1Measure = (2*precision*recall)/(precision+recall)
	else:
		f1Measure = 0
	print 'precision:%f, recall : %f, f1 measure: %f'%(precision,recall,f1Measure)
	return f1Measure
	
def findStates(outsideObjects,prevClassifier):
	stateCollection = []
	for i in range(len(outsideObjects)):
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = 'init'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '0'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '1'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '2'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '3'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '01'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '02'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '03'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '12'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '13'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '23'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '012'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '013'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '023'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '123'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = 'NA'
		
		stateCollection.append(state)
	
	
	return stateCollection
	
			
#Optimization based on a batch
def adaptiveOrder3():
	#1:logistic regression
	#2:extra tree
	#3:random forest
	#4:Adaboost
	# each image based ordering
	
	# This is the dynamic workflow. First classifier is chosen based on (delta f1/ delta cost) value. For the remaining classifiers, we check the probability output and try to match 
	# an image in the validation dataset which has a probability value very close to this value. From that imge we retrieve the probability values of the other classifiers. We check
	# the FPR value corresponding to that classifier. The classifier with minimum FPR*Cost value is chosen.
	
	f1 = open('queryTestResImageAdaptive3.txt','w+')

	#lr,et,rf,ab
	
	
	
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	
	# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
	# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]			
	#print currentProbability
	
	
	# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
	# The bit vector corresponding to classifier 2 and classifier 3 are set.
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
			
	#print prevClassifier
	
	#currentUncertainty list stores the information of current uncertainty of all the images.
	
	currentUncertainty = [1]*len(dl)
	#print currentUncertainty
	operator = set[0]
	count = 0
	while True:
		t1 = time.time()
		for i in range(len(dl)):
			#print 'image number:%d'%(i)
			if count !=0 :
				if nextBestClassifier[i] == 'LR':
					operator = set[0]
				if nextBestClassifier[i] == 'ET':
					operator = set[1]
				if nextBestClassifier[i] == 'RF':
					operator = set[2]
				if nextBestClassifier[i] == 'SVM':
					operator = set[3]
			prob = operator(dl[i])
			#rocProb = prob[0]
			rocProb = convertToRocProb(prob[0],operator)
			#print 'rocProb:%f'%rocProb
			
			#finding index of classifier
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			tempProb[indexClf] = rocProb
			#print currentProbability[i]
			
			# setting the bit for the corresponding classifier
			tempClf = prevClassifier[i][0]
			tempClf[indexClf] = 1
			#print prevClassifier[i]
			
			#combining the probability using Noisy-OR Model.
			combinedProbability = 0
			productProbability =1
			sumProbability = 0
			count2 =0
			flag = 0
			for p in currentProbability[i][0]:
				if p !=-1 :
					sumProbability +=p
					count2+=1
					flag = 1
					#productProbability = productProbability*(1-p)
			if flag ==1:
				combinedProbability = float(sumProbability)/count2
				#combinedProbability = 1 - productProbability
			else:
				combinedProbability =0.5
			
			# using the combined probability value to calculate uncertainty
			uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
			currentUncertainty[i] = uncertainty
			#print 'current uncertainty for image:%d is %f'%(i,uncertainty)
			
			
		#currentUncertainty = [0.5]*len(dl)
		#print currentUncertainty
		nextBestClassifier = [0]*len(dl)
		deltaUncertainty = [0] *len(dl)
		for i in range(len(dl)):
			#print prevClassifier.get(i)[0]
			[nextBestClassifier[i],deltaUncertainty[i]] = chooseNextBest(prevClassifier.get(i)[0],currentUncertainty[i])
		#print nextBestClassifier
		print deltaUncertainty 
		#array = numpy.array([4,2,7,1])
		seq = sorted(deltaUncertainty)
		index = [seq.index(v) for v in deltaUncertainty]
		print index
		
		t2 = time.time()
		timeElapsed = t2-t1
		print 'round %d completed'%(count)
		print 'time taken %f'%(timeElapsed)
		
		qualityOfAnswer = findQuality(currentProbability)
		print 'f1 measure of the answer set: %f, precision:%f, recall:%f'%(qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
		
		if count >= 1:
			break
		count=count+1
	
	#qualityOfAnswer = findQuality(currentProbability)
	#print 'quality of the answer set: %f'%(qualityOfAnswer)
	
#Optimization based on individual images. Objects are ordered based on  (delta uncertainty)/(delta cost).
def adaptiveOrder4(timeBudget):
	#1:logistic regression
	#2:extra tree
	#3:random forest
	#4:Adaboost
	
	f1 = open('queryTestResImageAdaptiveOptimal.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	
	# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
	# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]			
	#print currentProbability
	
	
	# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
	# The bit vector corresponding to classifier 2 and classifier 3 are set.
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
			
	#print prevClassifier
	
	#currentUncertainty list stores the information of current uncertainty of all the images.
	
	currentUncertainty = [1]*len(dl)
	#print currentUncertainty
	operator = set[0]
	count = 0
	t1 = time.time()
	while True:
		if count !=0 :
			if nextBestClassifier[i] == 'GNB':
				operator = set[0]
			if nextBestClassifier[i] == 'ET':
				operator = set[1]
			if nextBestClassifier[i] == 'RF':
				operator = set[2]
			if nextBestClassifier[i] == 'SVM':
				operator = set[3]
		if(count==0):
			i=0

		prob = operator(dl[i])
		rocProb = prob[0]
		
		
		#finding index of classifier
		indexClf = set.index(operator)
		tempProb = currentProbability[i][0]
		tempProb[indexClf] = rocProb
		#print currentProbability[i]
		
		# setting the bit for the corresponding classifier
		tempClf = prevClassifier[i][0]
		tempClf[indexClf] = 1
		#print prevClassifier[i]
		
		#combining the probability using Noisy-OR Model.
		combinedProbability = 0
		productProbability =1
		sumProbability = 0
		count =0
		flag = 0
		for p in currentProbability[i][0]:
			if p !=-1:
				#productProbability = productProbability*(1-p) # For Noisy OR model
				sumProbability +=p
				count+=1
				flag =1

		if flag ==1:
			combinedProbability = float(sumProbability)/count
			#combinedProbability = 1 - productProbability
		else:
			combinedProbability =0.5
		
		# using the combined probability value to calculate uncertainty
		uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
		currentUncertainty[i] = uncertainty
		nextBestClassifier = [0]*len(dl)
		deltaUncertainty = [0] *len(dl)
		
		for j in range(len(dl)):
			[nextBestClassifier[j],deltaUncertainty[j]] = chooseNextBest(prevClassifier.get(j)[0],currentUncertainty[j])
		seq = sorted(deltaUncertainty, reverse=True)
		#print seq
		#Ordering the objects based on deltaUncertainty Value
		order = [seq.index(v) for v in deltaUncertainty]
		topIndex= order.index(min(order))
		print 'top index:%d'%(topIndex)
		i=topIndex #next image to be run
		t2 = time.time()
		timeElapsed = t2-t1
		print 'round %d completed'%(count)
		print 'time taken %f'%(timeElapsed)
		
		qualityOfAnswer = findQuality(currentProbability)
		#print 'quality of the answer set: %f'%(qualityOfAnswer)
		print sortedTruth
		print 'f1 measure of the answer set: %f, precision:%f, recall:%f'%(qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
		if(timeElapsed>timeBudget):
			break
		#if count >= 5000:
		#	break
		count=count+1
	
	#qualityOfAnswer = findQuality(currentProbability)
	#print 'quality of the answer set: %f'%(qualityOfAnswer)
	
#Optimization based on each clusters
# In this algorithm, we create a cluster of objects for each state. clustering is based on probability values of the objetcs.
# Decision is based on probability values of the objects.
# 
#Optimization based on a batch of images. Batches are ordered based on  (delta uncertainty)/(delta cost).
def adaptiveOrder5(timeBudget):
	#1:Gaussian Naive Bayes
	#2:extra tree
	#3:random forest
	#4:SVM
	# each image based ordering
	f1 = open('queryExecutionResult2.txt','w+')

	#gnb,et,rf,svm
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	
	# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
	# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]			
	#print currentProbability
	num_ones = (nl==1).sum()
	
	# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
	# The bit vector corresponding to classifier 2 and classifier 3 are set.
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]

	currentUncertainty = [1]*len(dl)
	operator = set[0]
	count = 0
	t1 = time.time()
	while True:
		nextBestClassifier = [0]*len(dl)
		deltaUncertainty = [0] *len(dl)
		if count>=1:
			[nextBestClassifier,blockOfImages] = chooseNextBestBasedOnBlocking(prevClassifier,currentUncertainty,currentProbability)
			if(len(blockOfImages)==0):
				break
		else:
			blockOfImages = [0]
			nextBestClassifier = 'GNB'
		print 'nextBestClassifier'
		print nextBestClassifier # nextBestClassifier variable stores the best classifier among remaining classifiers for a particular block of images.
		print 'blockOfImages'
		print blockOfImages  # blockOfImages variable stores indices of all the images that will be considered next.
		#if len(blockOfImages)==0:
		#	break
		if count !=0 :
			if nextBestClassifier == 'GNB':
				operator = set[0]
			if nextBestClassifier == 'ET':
				operator = set[1]
			if nextBestClassifier == 'RF':
				operator = set[2]
			if nextBestClassifier == 'SVM':
				operator = set[3]
		images = [dl[k] for k in blockOfImages]  # storing actual objetcts in a temporary list.
		probValues = operator(images)					 # executing the classifier on the set of objetcts.
		for j in range(len(blockOfImages)):
			#print blockOfImages[j]
			#prob = operator(dl[blockOfImages[j]])
			imageProb = probValues[j]
			#print imageProb
			#print imageProb[0]
			rocProb = imageProb
			#rocProb = convertToRocProb(prob[0],operator)
			#finding index of classifier
			indexClf = set.index(operator)
			tempProb = currentProbability[blockOfImages[j]][0]
			tempProb[indexClf] = rocProb
			#print currentProbability[i]
			
			# setting the bit for the corresponding classifier
			tempClf = prevClassifier[blockOfImages[j]][0]
			tempClf[indexClf] = 1
			#print 'prevClassifier'
			#print prevClassifier[j]
			
			#combining the probability using Noisy-OR Model.
			combinedProbability = 0
			productProbability =1
			sumProbability = 0
			count2 =0
			flag = 0
			for p in currentProbability[blockOfImages[j]][0]:
				if p !=-1:
					#productProbability = productProbability*(1-p)
					sumProbability +=p
					count2+=1
					flag =1

			if flag ==1:
				combinedProbability = float(sumProbability)/count2
			else:
				combinedProbability =0	
			# using the combined probability value to calculate uncertainty
			uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
			currentUncertainty[j] = uncertainty
		t2 = time.time()
		timeElapsed = t2-t1
		print 'round %d completed'%(count)
		print 'time taken %f'%(timeElapsed)
		qualityOfAnswer = findQuality(currentProbability)
		count=count+1
		#print 'quality of the answer set: %f'%(qualityOfAnswer)
		print 'round no: %d, timestamp: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(count,timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
		
		#print>>f1,"Answer set: {} ".format(qualityOfAnswer[3])
		countTrue = 0
		for k in qualityOfAnswer[3] :
			if(nl[k] ==1):
				countTrue = countTrue+1
		precisionActual = float(countTrue)/len(qualityOfAnswer[3])
		
		recallActual = float(countTrue)/(num_ones)
		f1Actual = (2*precisionActual*recallActual)/(precisionActual+recallActual)
		#print>>f1,'f1 actual: %f'%(f1Actual)
		
		print>>f1,'round no: %d, timestamp: %f, f1 measure of the answer set: %f, precision:%f, recall:%f, Actual f1:%f'%(count,timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2],f1Actual)
		
		#if count>2:
			#print("Current Answer set : {} ".format(qualityOfAnswer[3]))
		#print("CurrentUncertainty: {} ".format(currentUncertainty))
		#if(timeElapsed>timeBudget):
		#	break
		if count >= 5000:
			break
		
	
	#qualityOfAnswer = findQuality(currentProbability)
	#print 'quality of the answer set: %f'%(qualityOfAnswer)
	

def runAllClassifiers():
	
	f1 = open('QueryExecutionResultAll.txt','w+')
	
	
	#Initialization step. 
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]	
			
	t1 = time.time()
	#lr,et,rf,ab
	#set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	num_ones = (nl==1).sum()
	
	for i in range(len(set)):
		operator = set[i]
		probValues = operator(dl)
		#rocProb = prob[0]
		for j in range(len(dl)):
			imageProb = probValues[j]
			rocProb = imageProb
			averageProbability = 0;
			print 'image:%d'%(j)
			print("Roc Prob : {} ".format(rocProb))
				
			#index of classifier
			indexClf = i
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(i)
			
			
	t2 = time.time()
	timeElapsed = t2-t1
	qualityOfAnswer = findQuality(currentProbability)
	countTrue = 0
	for k in qualityOfAnswer[3] :
		if(nl[k] ==1):
			countTrue = countTrue+1
	precisionActual = float(countTrue)/len(qualityOfAnswer[3])
	
	recallActual = float(countTrue)/(num_ones)
	f1Actual = (2*precisionActual*recallActual)/(precisionActual+recallActual)
		
	
	 
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f, Actual f1:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2],f1Actual)
	print>>f1,"Current Answer set : {} ".format(qualityOfAnswer[3])
	probSet = [probValues[x] for x in qualityOfAnswer[3]]	# storing probability values of the objects in answer set:
	print("Probability values of objects in Answer set : {} ".format(probSet))
	print>>f1,'Length of answer set:%d'%(len(qualityOfAnswer[3]))
	print>>f1,"Actual F1 measure : %f "%(findRealF1(qualityOfAnswer[3]))
	#print("Current Answer set : {} ".format(qualityOfAnswer[3]))
	
	
	

def runOneClassifier():
	
	f1 = open('ResultOneClfCompareMuct.txt','w+')
	
	trainX,trainY = pickle.load(open('MuctTrainGender2_XY.p','rb'))
	
	#gnb,et,rf,svm
	#set = [genderPredicate8,genderPredicate9]
	set = [genderPredicate6]
	
	pca = PCA()
	
	# transform / fit
	X_transformed = pca.fit_transform(trainX)
	dl_transformed = pca.transform(dl)
	
	
	for i in range(len(set)):
		#Initialization step. 
		currentProbability = {}
		for k in range(len(dl)):
			key = k
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
		t1 = time.time()
		operator = set[i]
		probValues = operator(dl_transformed)
		#rocProb = prob[0]
		for j in range(len(dl)):
			#probValues= operator(dl)
			imageProb = probValues[j]
			rocProb = imageProb
			#rocProb = operator(dl[j])
			#print 'image:%d'%(j)
			#print("Roc Prob : {} ".format(rocProb))
				
			#index of classifier
			indexClf = i
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(i)
			
			
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbability)
		print>>f1,operator
		print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
		print>>f1,"Current Answer set : {} ".format(qualityOfAnswer[3])
		probSet = [probValues[x] for x in qualityOfAnswer[3]]	# storing probability values of the objects in answer set:
		print>>f1,"Probability values of objects in Answer set : {} ".format(probSet)
		print>>f1,'Length of answer set:%d'%(len(qualityOfAnswer[3]))
		print>>f1,"Actual F1 measure : %f "%(findRealF1(qualityOfAnswer[3]))
		
		currentProbability.clear()



	
# This implementation us based on expected f1 measure. For each of the objects, we calculate the new probailities. Then we calculate the threshold and determine the F1 measure for each 
# of these new probabilities. After that we calculate combined f1 measure.
def adaptiveOrder6(timeBudget):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestResImageAdaptiveExpensive.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	
	# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
	# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]			
	#print currentProbability
	
	
	# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
	# The bit vector corresponding to classifier 2 and classifier 3 are set.
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
			
	#print prevClassifier
	
	#currentUncertainty list stores the information of current uncertainty of all the images.
	
	currentUncertainty = [1]*len(dl)
	currentF1measure = 0
	#print currentUncertainty
	operator = set[0]
	count = 0
	totalExecutionTime = 0
	totalThinkTime = 0
	
	t1 = time.time()
	while True:
		if count !=0 :
			if nextBestClassifier[i] == 'GNB':
				operator = set[0]
			if nextBestClassifier[i] == 'ET':
				operator = set[1]
			if nextBestClassifier[i] == 'RF':
				operator = set[2]
			if nextBestClassifier[i] == 'SVM':
				operator = set[3]
		if(count==0):
			i=0
		#calculating execution time
		t11 = time.time()
		prob = operator(dl[i])
		rocProb = prob[0]
		t12 = time.time()
		totalExecutionTime = totalExecutionTime + (t12-t11)
		
		
		#finding index of classifier
		indexClf = set.index(operator)
		tempProb = currentProbability[i][0]
		tempProb[indexClf] = rocProb
		print currentProbability[i]
		print currentProbability[i][0]
		
		# setting the bit for the corresponding classifier
		tempClf = prevClassifier[i][0]
		tempClf[indexClf] = 1
		#print prevClassifier[i]
		
	
		# calculating the current cobined probability
		combinedProbability = combineProbability(currentProbability[i])
		
		
		# using the combined probability value to calculate uncertainty
		uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
		currentUncertainty[i] = uncertainty
		
		
		nextBestClassifier = [0]*len(dl)
		deltaUncertainty = [0] *len(dl)
		benefitArray = [0] * len(dl)
		currentTempProbability = copy.deepcopy(currentProbability)
		newUncertaintyValue = 0 #initializing
		
		# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
		t21 = time.time()
		for j in range(len(dl)):
			#print 'deciding for object %d'%(j)
			[nextBestClassifier[j],deltaUncertainty[j]] = chooseNextBest(prevClassifier.get(j)[0],currentUncertainty[j])
			#benefitArray[j] = float(deltaUncertainty[j])/cost(nextBestClassifier[j])
			
			currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = currentUncertainty[j]  + float(deltaUncertainty[j])
			newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
			#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
			
			#finding index of classifier
			#indexTempProbClf = set.index(nextBestClassifier[j])
			if nextBestClassifier[j] == 'GNB':
				indexTempProbClf = 0
			if nextBestClassifier[j] == 'ET':
				indexTempProbClf = 1
			if nextBestClassifier[j] == 'RF':
				indexTempProbClf = 2
			if nextBestClassifier[j] == 'SVM':
				indexTempProbClf = 3
			currentTempProbValue = currentTempProbability[j][0]
			currentTempProbValue[indexTempProbClf] = newProbabilityValue1
			
			#calculate f1 measure using probability value 1.
			quality1 = findQuality(currentTempProbability)
			f1measure1 = quality1[0]			
			
			currentTempProbability.clear()
			
			currentTempProbability = copy.deepcopy(currentProbability)			
			newProbabilityValue2 = 1 - newProbabilityValue1			
			#calculate f1 measure using the second probability value
		
			#finding index of classifier
			#indexTempProbClf = set.index(nextBestClassifier[j])
			currentTempProbValue = currentTempProbability[j][0]
			currentTempProbValue[indexTempProbClf] = newProbabilityValue2
			
			#calculate f1 measure using probability value 1.
			quality2 = findQuality(currentTempProbability)
			f1measure2 = quality2[0]

			'''
			if combinedProbability < 0.5:
				combinedF1measure = combinedProbability * f1measure1 + (1- combinedProbability) * f1measure2
			else: 
				combinedF1measure = combinedProbability * f1measure2 + (1- combinedProbability) * f1measure1
			'''
			# combining the f1 measures. 
			combinedF1measure = combinedProbability* f1measure2 + (1- combinedProbability) * f1measure1 
			
			deltaF1measure = combinedF1measure - currentF1measure
			benefit = float(deltaF1measure/cost(nextBestClassifier[j]))
			benefitArray[j] = benefit
			
		#seq = sorted(benefitArray)
		print benefitArray
		#Ordering the objects based on deltaUncertainty Value
		#order = [seq.index(v) for v in benefitArray]
		topIndex= benefitArray.index(max(benefitArray))
		print 'top index:%d'%(topIndex)
		t22 = time.time()
		totalThinkTime = totalThinkTime + (t22-t21)
		
		i=topIndex #next image to be run
		t2 = time.time()
		timeElapsed = t2-t1
		print 'round %d completed'%(count)
		print 'time taken %f'%(timeElapsed)
		
		qualityOfAnswer = findQuality(currentProbability)
		#print 'quality of the answer set: %f'%(qualityOfAnswer)
		#print sortedTruth
		currentF1measure = qualityOfAnswer[0]
		print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
		if(timeElapsed>timeBudget):
			break
		#if count >= 5000:
		#	break
		count=count+1
	
	#qualityOfAnswer = findQuality(currentProbability)
	#print 'quality of the answer set: %f'%(qualityOfAnswer)	

def findUnprocessed(currentProbability):
	unprocessedImages = []
	for k in range(len(dl)):
		flag1 = 0
		for p in currentProbability[k][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			unprocessedImages.append(k)

	return unprocessedImages
	

def adaptiveOrder7(timeBudget):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestGenderMuct7.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	print timeBudget
	outsideObjects=[]
	
	#thinkPercentList = [0.001,0.002,0.005,0.007,0.01]
	#thinkPercentList = [0.005,0.006]
	thinkPercentList = [0.01,0.05,0.1,0.2]
	#thinkPercentList = [0.01]
	#thinkPercentList = [0.0005, 0.006]
	#blockList = [x * 10 for x in range(1, 500)]
	for percent in thinkPercentList:
	#for block in blockList:
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [1]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 10000	
		executionTime = 0
		stepSize = 20   #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = 20
		t11 = 0
		t12 = 0
		
		t1 = time.time()
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
			if count !=0:
				tempClf = ['GNB','ET','RF','SVM']
				for w in range(len(tempClf)):
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClf[w]]
					operator = set[w]
					images = [dl[k] for k in imageIndex]
					if len(imageIndex)!=0:
						t11 = time.time()
						probValues = operator(images)
						t12 = time.time()
						totalExecutionTime = totalExecutionTime + (t12-t11)
						if(totalExecutionTime +totalThinkTime)>timeBudget:
							break
						for i in range(len(imageIndex)):		
							#probValues = operator(dl[i])
							#rocProb = probValues
							rocProb = probValues[i]
							
							#finding index of classifier
							indexClf = set.index(operator)
							tempProb = currentProbability[imageIndex[i]][0]
							tempProb[indexClf] = rocProb
							#print currentProbability[imageIndex[i]]
							#if count !=0:
								#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClf = prevClassifier[imageIndex[i]][0]
							tempClf[indexClf] = 1
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i]])
							
							# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i]] = uncertainty
							
					if(totalExecutionTime +totalThinkTime)>timeBudget:
						break
					imageIndex[:]=[]
					images[:] =[]
					#probValues[:]=[]
				
			#t12 = time.time()
			#totalExecutionTime = totalExecutionTime + (t12-t11)	
			
			nextBestClassifier = [0]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray = [0] * len(dl)
			topKIndexes = [0] * 10000 # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			
			for j in range(len(dl)):
				#print 'deciding for object %d'%(j)
				[nextBestClassifier[j],deltaUncertainty[j]] = chooseNextBest(prevClassifier.get(j)[0],currentUncertainty[j])	
				newUncertaintyValue = currentUncertainty[j]  + float(deltaUncertainty[j])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[j] == 'GNB':
					indexTempProbClf = 0
				if nextBestClassifier[j] == 'ET':
					indexTempProbClf = 1
				if nextBestClassifier[j] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[j] == 'SVM':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[j])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[j]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[j])))
					benefitArray[j] = benefit
				else:
					benefitArray[j] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			#if len(outsideObjects) < blockSize :
			#	topKIndexes = heapq.nlargest(len(outsideObjects), range(len(benefitArray)), benefitArray.__getitem__)
			#else:
				#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			
			#i=topIndex #next image to be run
			t2 = time.time()
			#timeElapsed = timeElapsed+(t2-t11)
			#timeElapsed = timeElapsed + totalExecutionTime+ totalThinkTime 
			timeElapsed = totalExecutionTime + totalThinkTime
			#timeList.append(timeElapsed)
			print 'next images to be run'
			print topKIndexes
			
			print 'round %d completed'%(count)
			print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				thinkTime = t22-t21
				thinkTimePercent = percent
				blockSize = calculateBlockSize(timeBudget, thinkTime,thinkTimePercent)
				#blockSize = block
				topKIndexes[:]= []
				print 'blockSize: %d'%(blockSize)
			
			qualityOfAnswer = findQuality(currentProbability)
			print 'returned images'
			print qualityOfAnswer[3]
			if len(qualityOfAnswer[3]) > 0 :
				realF1 = findRealF1(qualityOfAnswer[3])
			else:
				realF1 = 0
			print 'real F1 : %f'%(realF1)
			#f1measure = qualityOfAnswer[0]
			f1measure = realF1
			timeList.append(timeElapsed)
			f1List.append(f1measure)
			
			'''
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				print 'time bound completed:%d'%(currentTimeBound)	
				print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				currentTimeBound = currentTimeBound + stepSize
			'''	
			
			if(timeElapsed>timeBudget):
				break
			#if count >= 5000:
			#	break
			count=count+1
			
		plt.title('Quality vs Time Value')
		print timeList
		print f1List
		#print>>f1,'percent : %f'%(percent)
		print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
		print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
		unprocessedObjects = findUnprocessed(currentProbability)
		print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
		print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))
		print>>f1,"budget values : {} ".format(timeList)
		print>>f1,"f1 measures : {} ".format(f1List)
		print>>f1,'total think time :%f'%(totalThinkTime)
		print>>f1,'total execution time :%f'%(totalExecutionTime)
	
		xValue = timeList
		yValue = f1List
		labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
		#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
		plt.plot(xValue, yValue,label=labelValue)

	plt.ylabel('Quality')
	plt.xlabel('Time')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive.png')
	#plt.show()
	plt.close()
	return [timeList,f1List]
	
# Here I plot (quality/execution) vs block size. Number of actions allowed is 5000. I calculate quality of answer set after those 5000 actions.
# I perform this experiment with varying block size. Block size will start from 10 and it will be varied upto 5000. 
def adaptiveOrder8(timeBudget):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestThreeFeatureMuct8.txt','w+')

	#lr,et,rf,ab
	
	
		
	set1 = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	set2 = [expressionPredicate6,expressionPredicate1,expressionPredicate3,expressionPredicate7]
	set3 = [agePredicate6,agePredicate1,agePredicate3,agePredicate7]
	
	print timeBudget
	outsideObjects=[]
	
	
	
	
	
	
	
	#blockList = [300]
	#blockList = [400] # for 600
	#blockList = [800]
	#blockList = [600]
	#blockList = [60]
	#blockList = [100]
	#blockList = [200]
	#blockList = [300]
	blockList = [600]
	executionPerformed = 0
	thinkTimeList = []
	executionTimeList = []
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		totalAllowedExecution = 1000
		executionPerformed = 0
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		
		
		
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbFeature1 = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbFeature1:
				currentProbFeature1[key].append(value)
			else:
				currentProbFeature1[key] = [value]			
	
	
		currentProbFeature2 = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbFeature2:
				currentProbFeature2[key].append(value)
			else:
				currentProbFeature2[key] = [value]
	
		currentProbFeature3 = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbFeature3:
				currentProbFeature3[key].append(value)
			else:
				currentProbFeature3[key] = [value]
	
	
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClfFeature1 = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClfFeature1:
				prevClfFeature1[key].append(value)
			else:
				prevClfFeature1[key] = [value]
	
		prevClfFeature2 = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClfFeature2:
				prevClfFeature2[key].append(value)
			else:
				prevClfFeature2[key] = [value]
				
		prevClfFeature3 = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClfFeature3:
				prevClfFeature3[key].append(value)
			else:
				prevClfFeature3[key] = [value]

		nextBestClassifierFeature1 = [0]*len(dl)
		nextBestClassifierFeature2 = [0]*len(dl)
		nextBestClassifierFeature3 = [0]*len(dl)
		currentUncertaintyFeature1 = [1]*len(dl)
		currentUncertaintyFeature2 = [1]*len(dl)
		currentUncertaintyFeature3 = [1]*len(dl)
	
		deltaUncertaintyFeature1 = [0] *len(dl)
		deltaUncertaintyFeature2 = [0] *len(dl)
		deltaUncertaintyFeature3 = [0] *len(dl)
	
		
		currentF1measure = 0
		#print currentUncertainty
		operator = set1[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = 4
		t11 = 0
		t12 = 0
		
		t1 = time.time()
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
			t1 = time.time()
			if count ==0:
				operator = set1[0]
				#val = dl[:10]
				
				
				for i in range(len(dl)):
					probValues = operator([dl[i]])
					
					indexClf = set1.index(operator)
					tempProb = currentProbFeature1[i][0]
					tempProb[indexClf] = probValues
					
					# setting the bit for the corresponding classifier
					tempClf = prevClfFeature1[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbFeature1[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertaintyFeature1[i] = uncertainty
					
					
				operator = set2[0]
				
				for i in range(len(dl)):
					
					probValues = operator([dl[i]])
					
					indexClf = set2.index(operator)
					tempProb = currentProbFeature2[i][0]
					tempProb[indexClf] = probValues
					
					# setting the bit for the corresponding classifier
					tempClf = prevClfFeature2[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbFeature2[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertaintyFeature2[i] = uncertainty
					
				operator = set3[0]
				
				for i in range(len(dl)):
					
					probValues = operator([dl[i]])
					
					indexClf = set3.index(operator)
					tempProb = currentProbFeature3[i][0]
					tempProb[indexClf] = probValues
					
					# setting the bit for the corresponding classifier
					tempClf = prevClfFeature3[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbFeature3[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertaintyFeature3[i] = uncertainty
				
				
				
				### Calculating the time and quality after running one classifiers each.########
				t2 = time.time()
				executionTime = executionTime + (t2- t1)
				#set.remove(genderPredicate8)

				qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
				print 'returned images'
				print qualityOfAnswer[3]
				print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(0)
				f1List.append(f1measure)
				#currentTimeBound = currentTimeBound + stepSize
				#print>>f1,'time bound completed:%d'%(currentTimeBound)
				###################################################################################	
				
					
			if count >0:
			
				###### Feature 1 ##########
				tempClf = ['DT','GNB','RF','KNN']
				flag = 0
				for w in range(4):
					tempClf = ['DT','GNB','RF','KNN']
					## Checking next best classifier value.
					imageIndex = [item for item in topKIndexes if (nextBestClassifierFeature1[item] == tempClf[w] and featureArray[item] ==1)]
					# Based on feature value, choosing the operator.
					
					operator = set1[w]
					images = [dl[k] for k in imageIndex]
					if len(imageIndex)!=0:
						#probValues = operator(images)						
						#if(totalExecutionTime +totalThinkTime)>timeBudget:
						#	break
						for i1 in range(len(imageIndex)):		
							t11 = time.time()
						
							probValues = operator([images[i1]])
							rocProb = probValues
						
							
							#finding index of classifier
							indexClf = set1.index(operator)
							tempProb = currentProbFeature1[imageIndex[i1]][0]
							tempProb[indexClf] = rocProb
							#print currentProbability[imageIndex[i]]
							#if count !=0:
								#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClf = prevClfFeature1[imageIndex[i1]][0]
							tempClf[indexClf] = 1
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbFeature1[imageIndex[i1]])
							
							# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertaintyFeature1[imageIndex[i1]] = uncertainty
							
							t12 = time.time()
							totalExecutionTime = totalExecutionTime + (t12-t11)	
						
							timeElapsed = timeElapsed +(t12-t11)	
						
							if timeElapsed > currentTimeBound:
								qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
		
								if len(qualityOfAnswer[3]) > 0:
									realF1 = findRealF1(qualityOfAnswer[3])
								else: 
									realF1 = 0
								#print>>f1,'real F1 : %f'%(realF1)
								#f1measure = qualityOfAnswer[0]
								f1measure = realF1
								timeList.append(timeElapsed)
								f1List.append(f1measure)
								#print>>f1,'F1 list : {}'.format(f1List)
		
								#print 'time bound completed:%d'%(currentTimeBound)	
								#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
								#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
		
								currentTimeBound = currentTimeBound + stepSize
								flag = 1
								#break 
						
							if timeElapsed > timeBudget:
								break		
					#imageIndex[:]=[]
					#images[:] =[]
					#probValues[:]=[]
				
				
				###### Feature 2 ##########
				
				#t11 = time.time()
				if flag !=1:
					for w in range(4):
						tempClf = ['DT','GNB','RF','KNN']
						## Checking next best classifier value.
						imageIndex = [item for item in topKIndexes if (nextBestClassifierFeature2[item] == tempClf[w] and featureArray[item] ==2)]
						# Based on feature value, choosing the operator.
					
						operator = set2[w]
						images = [dl[k] for k in imageIndex]
						if len(imageIndex)!=0:
							probValues = operator(images)					
							#if(totalExecutionTime +totalThinkTime)>timeBudget:
							#	break
							for i in range(len(imageIndex)):		
								t11 = time.time()
								
								#rocProb = probValues[i]
								
								probValues = operator([images[i]])
								rocProb = probValues
							
								#finding index of classifier
								indexClf = set2.index(operator)
								tempProb = currentProbFeature2[imageIndex[i]][0]
								tempProb[indexClf] = rocProb
								#print currentProbability[imageIndex[i]]
								#if count !=0:
									#print nextBestClassifier[imageIndex[i]]
							
								# setting the bit for the corresponding classifier
								tempClf = prevClfFeature2[imageIndex[i]][0]
								tempClf[indexClf] = 1
							
								# calculating the current cobined probability
								combinedProbability = combineProbability(currentProbFeature2[imageIndex[i]])
							
								# using the combined probability value to calculate uncertainty
								uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
								currentUncertaintyFeature2[imageIndex[i]] = uncertainty
							
								t12 = time.time()
								totalExecutionTime = totalExecutionTime + (t12-t11)	
								timeElapsed = timeElapsed +(t12-t11)	
						
						
								if timeElapsed > currentTimeBound:
									qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
		
									if len(qualityOfAnswer[3]) > 0:
										realF1 = findRealF1(qualityOfAnswer[3])
									else: 
										realF1 = 0
									#print>>f1,'real F1 : %f'%(realF1)
									#f1measure = qualityOfAnswer[0]
									f1measure = realF1
									timeList.append(timeElapsed)
									f1List.append(f1measure)
									#print>>f1,'F1 list : {}'.format(f1List)
		
									#print 'time bound completed:%d'%(currentTimeBound)	
									#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
									#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
		
									currentTimeBound = currentTimeBound + stepSize
									#break 
						
								if timeElapsed > timeBudget:
									break	
				###### Feature 3 ##########
				
				#t11 = time.time()
				if flag !=1:
					for w in range(4):
						tempClf = ['DT','GNB','RF','KNN']
						## Checking next best classifier value.
						imageIndex = [item for item in topKIndexes if (nextBestClassifierFeature3[item] == tempClf[w] and featureArray[item] ==3)]
						# Based on feature value, choosing the operator.
					
						operator = set3[w]
						images = [dl[k] for k in imageIndex]
						if len(imageIndex)!=0:
							#probValues = operator(images)						
							#if(totalExecutionTime +totalThinkTime)>timeBudget:
							#	break
							for i in range(len(imageIndex)):		
								t11 = time.time()
								#rocProb = probValues[i]
								probValues = operator([images[i]])
								rocProb = probValues
							
							
								#finding index of classifier
								indexClf = set3.index(operator)
								tempProb = currentProbFeature3[imageIndex[i]][0]
								tempProb[indexClf] = rocProb
								#print currentProbability[imageIndex[i]]
								#if count !=0:
									#print nextBestClassifier[imageIndex[i]]
							
								# setting the bit for the corresponding classifier
								tempClf = prevClfFeature3[imageIndex[i]][0]
								tempClf[indexClf] = 1
							
								# calculating the current cobined probability
								combinedProbability = combineProbability(currentProbFeature3[imageIndex[i]])
							
								# using the combined probability value to calculate uncertainty
								uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
								currentUncertaintyFeature3[imageIndex[i]] = uncertainty
							
								t12 = time.time()
								totalExecutionTime = totalExecutionTime + (t12-t11)	
								timeElapsed = timeElapsed +(t12-t11)	
						
						
								if timeElapsed > currentTimeBound:
									qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
		
									if len(qualityOfAnswer[3]) > 0:
										realF1 = findRealF1(qualityOfAnswer[3])
									else: 
										realF1 = 0
									#print>>f1,'real F1 : %f'%(realF1)
									#f1measure = qualityOfAnswer[0]
									f1measure = realF1
									timeList.append(timeElapsed)
									f1List.append(f1measure)
									#print>>f1,'F1 list : {}'.format(f1List)
		
									#print 'time bound completed:%d'%(currentTimeBound)	
									#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
									#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
		
									currentTimeBound = currentTimeBound + stepSize
									#break 
						
								if timeElapsed > timeBudget:
									break	
				imageIndex[:]=[]
				images[:] =[]
					#probValues[:]=[]
				'''	
				t12 = time.time()
				totalExecutionTime = totalExecutionTime + (t12-t11)	
				executionTimeList.append(t12-t11)
				'''
			executionTimeList.append(totalExecutionTime)
			nextBestClassifier = [0]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray = [-10] * len(dl)
			featureArray = [0] * len(dl)
			
			topKIndexes = [0] * 10000 # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			
			outsideObjects = allObjects
			
			### Uncomment this part for choosing objects from outside of the answer set.
			#outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			
			
			print>>f1,"inside objects : {} ".format(currentAnswerSet)
			print>>f1,"length of inside objects : %f"%len(currentAnswerSet)
			stateListInside1 = findStates(currentAnswerSet,prevClfFeature1)
			print>>f1,"state of inside objects for feature 1: {}".format(stateListInside1)
			stateListInside2 = findStates(currentAnswerSet,prevClfFeature2)
			print>>f1,"state of inside objects for feature 2: {}".format(stateListInside2)
			stateListInside3 = findStates(currentAnswerSet,prevClfFeature3)
			print>>f1,"state of inside objects for feature 3: {}".format(stateListInside3)
			
			
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			print>>f1,"length of outsideObjects : %f"%len(outsideObjects)
			stateListOutside1 = findStates(outsideObjects,prevClfFeature1)
			print>>f1,"state of outside objects for feature 1 : {}".format(stateListOutside1)
			stateListOutside2 = findStates(outsideObjects,prevClfFeature2)
			print>>f1,"state of outside objects for feature 2 : {}".format(stateListOutside2)
			stateListOutside3 = findStates(outsideObjects,prevClfFeature3)
			print>>f1,"state of outside objects for feature 3 : {}".format(stateListOutside3)
			
			
			flag = 0
			
			if(len(outsideObjects)==0):
				break
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			
			for j in range(len(outsideObjects)):
				#print>>f1,'deciding for object %d'%(outsideObjects[j])
				#print>>f1,"currentUncertainty: {}".format(currentUncertainty)
				
				[nextBestClassifierFeature1[outsideObjects[j]],deltaUncertaintyFeature1[outsideObjects[j]]] = chooseNextBest(prevClfFeature1.get(outsideObjects[j])[0],currentUncertaintyFeature1[outsideObjects[j]],1)	
				newUncertaintyValue1 = currentUncertaintyFeature1[outsideObjects[j]]  + float(deltaUncertaintyFeature1[outsideObjects[j]])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue1)
			
				[nextBestClassifierFeature2[outsideObjects[j]],deltaUncertaintyFeature2[outsideObjects[j]]] = chooseNextBest(prevClfFeature2.get(outsideObjects[j])[0],currentUncertaintyFeature2[outsideObjects[j]],2)	
				newUncertaintyValue2 = currentUncertaintyFeature2[outsideObjects[j]]  + float(deltaUncertaintyFeature2[outsideObjects[j]])
				newProbabilityValue2 = convertEntropyToProb(newUncertaintyValue2)
				
				[nextBestClassifierFeature3[outsideObjects[j]],deltaUncertaintyFeature3[outsideObjects[j]]] = chooseNextBest(prevClfFeature3.get(outsideObjects[j])[0],currentUncertaintyFeature3[outsideObjects[j]],3)	
				newUncertaintyValue3 = currentUncertaintyFeature3[outsideObjects[j]]  + float(deltaUncertaintyFeature3[outsideObjects[j]])
				newProbabilityValue3 = convertEntropyToProb(newUncertaintyValue3)
			

				probability1_i = combineProbability(currentProbFeature1[outsideObjects[j]])
				probability2_i = combineProbability(currentProbFeature2[outsideObjects[j]])
				probability3_i = combineProbability(currentProbFeature3[outsideObjects[j]])
			
				######## Comparing the benefit values ##############
				jointProbInitial = probability1_i * probability2_i * probability3_i
				newJointProb1 = newProbabilityValue1 * probability2_i * probability3_i
				if 	cost(nextBestClassifierFeature1[outsideObjects[j]]) !=0:			
					benefit1 = float((newJointProb1*jointProbInitial)/float(cost(nextBestClassifierFeature1[outsideObjects[j]])))
				else:
					benefit1 = 0
				
				newJointProb2 = probability1_i * newProbabilityValue2 * probability3_i
				if 	cost(nextBestClassifierFeature2[outsideObjects[j]]) !=0:			
					benefit2 = float((newJointProb2*jointProbInitial)/float(cost(nextBestClassifierFeature2[outsideObjects[j]])))
				else:
					benefit2 = 0
					
				newJointProb3 = probability1_i * probability2_i * newProbabilityValue3
				if 	cost(nextBestClassifierFeature3[outsideObjects[j]]) !=0:			
					benefit3 = float((newJointProb3*jointProbInitial)/float(cost(nextBestClassifierFeature3[outsideObjects[j]])))
				else:
					benefit3 = 0
				
				if benefit1 == max(benefit1,benefit2,benefit3) : 
					benefit = benefit1
					feature = 1
				elif benefit2 == max(benefit1,benefit2,benefit3):
					benefit = benefit2
					feature = 2
				else:
					benefit = benefit3
					feature = 3
				
					
				if ( cost(nextBestClassifierFeature1[outsideObjects[j]]) or cost(nextBestClassifierFeature2[outsideObjects[j]]) or cost(nextBestClassifierFeature3[outsideObjects[j]])) != 0:
					benefitArray[outsideObjects[j]] = benefit
					featureArray[outsideObjects[j]] = feature
				else:
					benefitArray[outsideObjects[j]] = -1
					featureArray[outsideObjects[j]] = -1
				

				
			
			###################################
			#Choosing the  block size #########
			###################################
			
			if len(outsideObjects) < blockSize :
				topKIndexes = outsideObjects
				#topKIndexes = heapq.nlargest(len(outsideObjects), range(len(outsideObjects)), benefitArray.__getitem__)
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			thinkTimeList.append(t22-t21)
			
			#i=topIndex #next image to be run
			t2 = time.time()
			#timeElapsed = timeElapsed+(t2-t11)
			#timeElapsed = timeElapsed + totalExecutionTime+ totalThinkTime 
			timeElapsed = totalExecutionTime + totalThinkTime
			
			#timeList.append(timeElapsed)
			
			######Printing the next images to be run. Also the corresponding classifier. ##############
			#print 'next images to be run'
			#print topKIndexes
			
			#print>>f1,'benefit array: {}'.format(benefitArray)
			#print>>f1,'next images to be run: {}'.format(topKIndexes)
			
			classifierSet1 = [nextBestClassifierFeature1[item2] for item2 in topKIndexes]
			#print>>f1,'classifier set 1: {}'.format(classifierSet1)
			
			classifierSet2 = [nextBestClassifierFeature2[item2] for item2 in topKIndexes]
			#print>>f1,'classifier set 2: {}'.format(classifierSet2)
			
			classifierSet3 = [nextBestClassifierFeature3[item2] for item3 in topKIndexes]
			
			
			
			#print>>f1,'feature array: {}'.format(featureArray)
			if(all(element1=='NA' for element1 in classifierSet1) and all(element2=='NA' for element2 in classifierSet2)  and all(element3=='NA' for element3 in classifierSet3) and count > 10):
				break
			##############################################################################################
			
			print 'round %d completed'%(count)
			print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				#thinkTime = t22-t21
				#thinkTimePercent = percent
				#blockSize = calculateBlockSize(timeBudget, thinkTime,thinkTimePercent)
				blockSize = block
				topKIndexes[:]= []
				print 'blockSize: %d'%(blockSize)
			
			
			
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
				#f1measure = qualityOfAnswer[0]
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
				
				
				f1measure = realF1
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				#print 'time bound completed:%d'%(currentTimeBound)	
				#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(realF1,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				currentTimeBound = currentTimeBound + stepSize
				
			if timeElapsed > timeBudget:
				break
			'''
			executionPerformed = executionPerformed + 	blockSize	
			if(executionPerformed>totalAllowedExecution):
				qualityOfAnswer = findQuality(currentProbability)
				print 'returned images'
				print qualityOfAnswer[3]
				if len(qualityOfAnswer[3]) > 0 :
					realF1 = findRealF1(qualityOfAnswer[3])
				else:
					realF1 = 0
				print 'real F1 : %f'%(realF1)
				
	
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				f1measurePerAction = float(f1measure)/totalAllowedExecution
				timeList.append(timeElapsed)
				f1List.append(f1measurePerAction)
				realF1List.append(f1measure)
				
				print>>f1,'block size:%f'%(blockSize)
				print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
				print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
				unprocessedObjects = findUnprocessed(currentProbability)
				print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
				print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))
				print>>f1,'think time list: {}'.format(thinkTimeList)
				print>>f1,'execution time list: {}'.format(executionTimeList)
				
				break
				'''
			#if count >= 5000:
			#	break
			count=count+1
		
		'''	
		plt.title('Quality vs Time Value')
		#print>>f1,'percent : %f'%(percent)
		#print>>f1,'block size : %f'%(block)
		#print>>f1,"f1 measures : {} ".format(realF1List)
		#print>>f1,'total think time :%f'%(totalThinkTime)
		#print>>f1,'total execution time :%f'%(totalExecutionTime)
		plt.ylabel('Quality')
		plt.xlabel('time')
		xValue = timeList
		yValue = f1List
		print>>f1,"x value : {} ".format(xValue)
		print>>f1,"y value : {} ".format(yValue)
		print "x value : {} ".format(xValue)
		print "y value : {} ".format(yValue)
	
		plt.plot(xValue, yValue,'b')
		plt.ylim([0, 1])
		plt.legend(loc="upper left")
		plt.savefig('plotQualityAdaptive8MultiFeature'+str(block)+'.eps', format='eps')
		plt.title('Quality vs Time for block size = '+str(block))
		#plt.show()
		plt.close()
		#xValue = timeList
		#yValue = f1List
		'''
		
	'''
	plt.ylabel('Quality')
	plt.xlabel('Block Size')
	xValue = blockList
	yValue = realF1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	'''
	plt.ylabel('Quality')
	plt.xlabel('time')
	xValue = timeList
	yValue = f1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	'''
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive8MultiFeature.png')
	#plt.show()
	plt.close()
	'''
	return [timeList,f1List]


# Here I plot (quality/execution) vs block size. Number of actions allowed is 5000. I calculate quality of answer set after those 5000 actions.
# I perform this experiment with varying block size. Block size will start from 10 and it will be varied upto 500. 
# In this case I do not choose the objects randomly.
def adaptiveOrder9(timeBudget):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestGenderMuct9.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	
	
	print timeBudget
	outsideObjects=[]
	
	#thinkPercentList = [0.001,0.002,0.005,0.007,0.01]
	#thinkPercentList = [0.005,0.006]
	#thinkPercentList = [0.01,0.05,0.1,0.2]
	#thinkPercentList = [0.01]
	#thinkPercentList = [0.0005, 0.006]
	#blockList = [x*50 for x in range(1,10)]
	blockList = [10,20,50,100,200,300,400,500]
	executionPerformed = 0
	
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		totalAllowedExecution = 3000
		executionPerformed = 0
		thinkTimeList = []
		#######This part is for choosing objects in the first iteration
		probArray = [0] * len(dl)
		operator1 = genderPredicate3
		probArray = operator1(dl)
		print>>f1,probArray
		print>>f1,'block size:%f'%(block)
		print 'block size:%f'%(block)
		lowestTopKProbs = heapq.nsmallest(block, range(len(probArray)), probArray.__getitem__)
		print>>f1,lowestTopKProbs
		print>>f1,probArray[lowestTopKProbs[0]]
		#print>>f1,probArray[lowestTopKProbs[block-1]]
		#print>>f1,probArray[lowestTopKProbs[block-50]]
		###################################################################
		
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [1]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		stepSize = 20   #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = 20
		t11 = 0
		t12 = 0
		
		t1 = time.time()
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
			if count !=0:
				tempClfString = ['GNB','ET','RF','SVM']
				for w in range(len(tempClfString)):
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfString[w]]
					print>>f1,'Number of objects:%d, for classifier:%s'%(len(imageIndex),tempClfString[w])
					if w!=4:
						operator = set[w]
					'''
					else:
						if(len(imageIndex)==len(topKIndexes)):    #This implies no more images to be run
							break
					'''
					images = [dl[k] for k in imageIndex]
					if len(imageIndex)!=0:
						t11 = time.time()
						probValues = operator(images)
						t12 = time.time()
						totalExecutionTime = totalExecutionTime + (t12-t11)
						#if(totalExecutionTime +totalThinkTime)>timeBudget:
						#	break
						for i in range(len(imageIndex)):		
							#probValues = operator(dl[i])
							#rocProb = probValues
							rocProb = probValues[i]
							
							#finding index of classifier
							indexClf = set.index(operator)
							tempProb = currentProbability[imageIndex[i]][0]
							tempProb[indexClf] = rocProb
							#print currentProbability[imageIndex[i]]
							#if count !=0:
								#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClf = prevClassifier[imageIndex[i]][0]
							tempClf[indexClf] = 1
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i]])
							
							# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i]] = uncertainty						
					
					#imageIndex[:]=[]
					#images[:] =[]
					#topKIndexes[:] =[]
					#probValues[:]=[]
				'''
				if(len(imageIndex)==len(topKIndexes) and w ==4):    #This implies no more images to be run
							break
				'''
			nextBestClassifier = [0]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray = [0] * len(dl)
			#topKIndexes = [0] * 10000 # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			
			for j in range(len(dl)):
				#print 'deciding for object %d'%(j)
				[nextBestClassifier[j],deltaUncertainty[j]] = chooseNextBest(prevClassifier.get(j)[0],currentUncertainty[j])	
				newUncertaintyValue = currentUncertainty[j]  + float(deltaUncertainty[j])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[j] == 'GNB':
					indexTempProbClf = 0
				if nextBestClassifier[j] == 'ET':
					indexTempProbClf = 1
				if nextBestClassifier[j] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[j] == 'SVM':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[j])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[j]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[j])))
					benefitArray[j] = benefit
				else:
					benefitArray[j] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			#if len(outsideObjects) < blockSize :
			#	topKIndexes = heapq.nlargest(len(outsideObjects), range(len(benefitArray)), benefitArray.__getitem__)
			#else:
				#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
	
			thinkTimeList.append(t22-t21)
			#i=topIndex #next image to be run
			t2 = time.time()
			#timeElapsed = timeElapsed+(t2-t11)
			#timeElapsed = timeElapsed + totalExecutionTime+ totalThinkTime 
			timeElapsed = totalExecutionTime + totalThinkTime
			
			#timeList.append(timeElapsed)
			print>>f1,'benefit array: {}'.format(benefitArray)
			print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			print>>f1,'classifier set: {}'.format(classifierSet)
			#print>>f1,'next best classifier set: {}'.format(nextBestClassifier)
			
			
			print 'round %d completed'%(count)
			print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				#thinkTime = t22-t21
				#thinkTimePercent = percent
				#blockSize = calculateBlockSize(timeBudget, thinkTime,thinkTimePercent)
				blockSize = block
				#topKIndexes[:]= []
				print 'blockSize: %d'%(blockSize)
			
			
			
			'''
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				print 'time bound completed:%d'%(currentTimeBound)	
				print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				currentTimeBound = currentTimeBound + stepSize
			'''	
			executionPerformed = executionPerformed + 	blockSize	
			if(executionPerformed>totalAllowedExecution):
				qualityOfAnswer = findQuality(currentProbability)
				print 'returned images'
				print qualityOfAnswer[3]
				if len(qualityOfAnswer[3]) > 0 :
					realF1 = findRealF1(qualityOfAnswer[3])
				else:
					realF1 = 0
				print 'real F1 : %f'%(realF1)
				
	
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				f1measurePerAction = float(f1measure)/totalAllowedExecution
				timeList.append(timeElapsed)
				f1List.append(f1measurePerAction)
				realF1List.append(f1measure)
				
				print>>f1,'block size:%f'%(blockSize)
				print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
				print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
				unprocessedObjects = findUnprocessed(currentProbability)
				
				print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
				print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))				
				print>>f1,'think time list: {}'.format(thinkTimeList)
				
				break
			#if count >= 5000:
			#	break
			count=count+1
			
		plt.title('Quality vs Time Value')
		#print>>f1,'percent : %f'%(percent)
		#print>>f1,'block size : %f'%(block)
		#print>>f1,"f1 measures : {} ".format(realF1List)
		#print>>f1,'total think time :%f'%(totalThinkTime)
		#print>>f1,'total execution time :%f'%(totalExecutionTime)
	
		#xValue = timeList
		#yValue = f1List
		

	plt.ylabel('Quality')
	plt.xlabel('Block Size')
	xValue = blockList
	yValue = realF1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive9.png')
	#plt.show()
	plt.close()
	return [timeList,f1List]	
	
	
def baseline1(N):  
	'''
	For this algorithm, one classifier is chosen randomly from a given set of classifiers.
	'''
	f1 = open('QueryExecutionResultMuctBaseline1GenderAverage.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	
	
	for k in range(0,N):  # number of times this algorithm will be executed
		#Initialization step. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
		t1 = time.time()
		#gnb,et,rf,svm
		set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
		workflow =[]
		round = 1 
		
		while len(set) >0:
			operator = random.choice(set)
			probValues = operator(dl)
			workflow.append(operator)
			#rocProb = prob[0]
			for j in range(len(dl)):
				imageProb = probValues[j]
				rocProb = imageProb
				averageProbability = 0;
				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				tempProb = currentProbability[j][0]
				tempProb[indexClf] = rocProb

			print 'round %d completed'%(round)
			set.remove(operator)
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbability)
		f1measure = qualityOfAnswer[0]
		
		totalTime += timeElapsed
		totalQuality += f1measure
		print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
		print>>f1,"Workflow : {} ".format(workflow)
	
	averageTime = float(totalTime)/N
	averageQuality = float(totalQuality)/N
	print>>f1,'Average time taken: %f, Average f1 measure of the answer set: %f'%(averageTime, averageQuality)	
	
	
def baseline2():  
	'''
	For this algorithm, classifiers are ordered based on (AUC)/Cost value.
	'''
	f1 = open('QueryExecutionResultMuctBaseline2Gender.txt','w+')
	
	
	#Initialization step. 
	currentProbability = {}
	
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]	
			
	t1 = time.time()
	
	#gnb,et,rf,svm
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	aucSet = [0.85,0.92,0.92,0.89]
	#costSet = [0.063052,0.014482,0.015253,1.567327]
	costSet = [0.029360,0.018030,0.020180,0.790850]
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print workflow
	round = 1 
	
	
	for i in range(len(workflow)):
		operator = workflow[i]
		probValues = operator(dl)
		
		for j in range(len(dl)):
			imageProb = probValues[j]
			rocProb = imageProb
			averageProbability = 0;
			#print 'image:%d'%(j)
			#print("Roc Prob : {} ".format(rocProb))
				
			#index of classifier
			indexClf = set.index(operator)
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(round)
		set.remove(operator)
		round = round + 1
		
			
	t2 = time.time()
	timeElapsed = t2-t1
	qualityOfAnswer = findQuality(currentProbability)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	print("Workflow : {} ".format(workflow))


def baseline3(budget):  
	'''
	For this algorithm, classifiers are chosen based on auc/cost value. But for one classifier, we try to run it on all the images.
	'''
	f1 = open('QueryResultBaseline3GenderMuct.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	timeList =[]
	f1List = []
	
	executionTime = 0
	stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound = 4
	
	#dt,gnb,rf,knn
	set1 = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet1 = [0.69,0.85,0.93,0.80]
	costSet1 = [0.014718,0.095875,0.023094, 0.866073 ]
	
	set2 = [expressionPredicate6,expressionPredicate1,expressionPredicate3,expressionPredicate7]
	aucSet2 = [0.51,0.73,0.70,0.68]
	costSet2 = [0.001,0.018030,0.020180,0.790850]
	
	set3 = [agePredicate6,agePredicate1,agePredicate3,agePredicate7]
	aucSet3 = [0.51,0.73,0.70,0.68]
	costSet3 = [0.001,0.018030,0.020180,0.790850]
	
	
	count =0
	
	currentProbFeature1 = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbFeature1:
			currentProbFeature1[key].append(value)
		else:
			currentProbFeature1[key] = [value]	
	
	currentProbFeature2 = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbFeature2:
			currentProbFeature2[key].append(value)
		else:
			currentProbFeature2[key] = [value]	
			
	currentProbFeature3 = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbFeature3:
			currentProbFeature3[key].append(value)
		else:
			currentProbFeature3[key] = [value]	
	
	
	t1 = time.time()
	if count ==0:
		
		###### Running the first classifier for feature 1 #############
		operator = set1[0]			
			
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#print>>f1,probValues
			indexClf = set1.index(operator)
			tempProb = currentProbFeature1[i][0]
			tempProb[indexClf] = probValues
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbFeature1[i])
					
			
		###### Running the first classifier for feature 2 #############
		operator = set2[0]			
				
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#print>>f1,probValues
			indexClf = set2.index(operator)
			tempProb = currentProbFeature2[i][0]
			tempProb[indexClf] = probValues
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbFeature2[i])
					
		###### Running the first classifier for feature 2 #############
		operator = set3[0]			
				
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#print>>f1,probValues
			indexClf = set3.index(operator)
			tempProb = currentProbFeature3[i][0]
			tempProb[indexClf] = probValues
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbFeature3[i])
			
	
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set1.remove(genderPredicate6)
	set2.remove(expressionPredicate6)
	set3.remove(agePredicate6)
	
	######## Finding initial quality value ######################	
	qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
	#print 'returned images'
	#print qualityOfAnswer[3]
	#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#######################################################
	
	set1 = [genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet1 = [0.85,0.93,0.80]
	costSet1 = [0.095875,0.023094, 0.866073 ]
	
	benefitSet1 = [ float(aucSet1[i])/costSet1[i] for i in range(len(aucSet1))]
	#print benefitSet1
	workflow1 =[x for y, x in sorted(zip(benefitSet1, set1),reverse=True)]
	#print workflow1
	 
	
	
	set2 = [expressionPredicate1,expressionPredicate3,expressionPredicate7]
	aucSet2 = [0.73,0.70,0.68]
	costSet2 = [0.018030,0.020180,0.790850]
	
	
	
	benefitSet2 = [ float(aucSet2[i])/costSet2[i] for i in range(len(aucSet2))]
	#print benefitSet2
	workflow2 =[x for y, x in sorted(zip(benefitSet2, set2),reverse=True)]
	#print workflow2
	
	
	set3 = [agePredicate1,agePredicate3,agePredicate7]
	aucSet3 = [0.73,0.70,0.68]
	costSet3 = [0.018030,0.020180,0.790850]
	
	benefitSet3 = [ float(aucSet3[i])/costSet3[i] for i in range(len(aucSet3))]
	#print benefitSet2
	workflow3 =[x for y, x in sorted(zip(benefitSet3, set3),reverse=True)]
	#print workflow2
	
	round = 1
	
	for k in range(0,1):  # number of times this algorithm will be executed
		#Initialization step. 
		
				
		t1 = time.time()
	
		#print("Workflow : {} ".format(workflow1))
		
		for i in range(len(workflow1)):
			operator1 = workflow1[i]
			operator2 = workflow2[i]
			operator3 = workflow3[i]
			
						
			#rocProb = prob[0]
			for j in range(len(dl)):
				
				t11 = time.time()
				imageProb = operator1([dl[j]])
				rocProb = imageProb

				#index of classifier
				indexClf = set1.index(operator1)
				tempProb = currentProbFeature1[j][0]
				tempProb[indexClf] = rocProb
				
				t12 = time.time()
				executionTime = executionTime + (t12- t11)
				
				
				t11 = time.time()
				imageProb = operator2([dl[j]])
				rocProb = imageProb

				#index of classifier
				indexClf = set2.index(operator2)
				tempProb = currentProbFeature2[j][0]
				tempProb[indexClf] = rocProb
				
				t12 = time.time()
				executionTime = executionTime + (t12- t11)
				
				
				#index of classifier
				t11 = time.time()
				imageProb = operator3([dl[j]])
				rocProb = imageProb

				#index of classifier
				indexClf = set3.index(operator3)
				tempProb = currentProbFeature3[j][0]
				tempProb[indexClf] = rocProb
				
				t12 = time.time()
				executionTime = executionTime + (t12- t11)
				
			
			
			
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
					#print 'returned images'
					#print qualityOfAnswer[3]
					#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else: 
						realF1 = 0
					#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print>>f1,'time bound completed:%d'%(currentTimeBound)
				
				if executionTime > budget:
					break
				
			#print 'round %d completed'%(round)
			
			if executionTime > budget:
					break
			
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		'''
		plt.title('Quality vs Time Value for BaseLine 3')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylim([0, 1])
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.savefig('QualityBaseLine3GenderMuct.eps', format='eps')
		#plt.show()
		plt.close()
		'''
		
	'''	
	print>>f1,"Workflow : {} ".format(workflow1)
	print>>f1,"Workflow : {} ".format(workflow2)
	print>>f1,'Time taken: %f'%(timeElapsed)
	'''
	#return f1measure
	return [timeList,f1List]


def baseline4(budget):  
	'''
	For this algorithm, classifiers are ordered based on auc/cost value. But for each images, we try to run all the classifiers before going to another image.
	'''
	
	#f1 = open('QueryExecutionResultBaseline4GenderAverage.txt','w+')
	f1 = open('QueryResultBaseline4GenderMuct.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	print 'Query budget:%f'%(budget)
	
	timeList =[]
	f1List = []
	executionTime = 0
	stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound = 4
	count =0
	
	#gnb,et,rf,svm
	#set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	#aucSet = [0.85,0.92,0.92,0.89]
	#costSet = [0.063052,0.014482,0.015253,1.567327]
	
	
	set1 = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet1 = [0.69,0.85,0.93,0.80]
	costSet1 = [0.014718,0.095875,0.023094, 0.866073 ]
	
	set2 = [expressionPredicate6,expressionPredicate1,expressionPredicate3,expressionPredicate7]
	aucSet2 = [0.51,0.73,0.70,0.68]
	costSet2 = [0.001,0.018030,0.020180,0.790850]
	
	set3 = [agePredicate6,agePredicate1,agePredicate3,agePredicate7]
	aucSet3 = [0.51,0.73,0.70,0.68]
	costSet3 = [0.001,0.018030,0.020180,0.790850]
	
	
	
	print 'size of the dataset:%d'%(len(dl))
	print 'budget:%d'%(budget)
	
	
	
	currentProbFeature1 = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbFeature1:
			currentProbFeature1[key].append(value)
		else:
			currentProbFeature1[key] = [value]	
	
	currentProbFeature2 = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbFeature2:
			currentProbFeature2[key].append(value)
		else:
			currentProbFeature2[key] = [value]	
			
	currentProbFeature3 = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbFeature3:
			currentProbFeature3[key].append(value)
		else:
			currentProbFeature3[key] = [value]	
	
	
	
	t1 = time.time()
	if count ==0:
		
		###### Running the first classifier for feature 1 #############
		operator = set1[0]			
			
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			print>>f1,probValues
			indexClf = set1.index(operator)
			tempProb = currentProbFeature1[i][0]
			tempProb[indexClf] = probValues

			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbFeature1[i])
			
		
		###### Running the first classifier for feature 2 #############
		operator = set2[0]			
				
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#print>>f1,probValues
			indexClf = set2.index(operator)
			tempProb = currentProbFeature2[i][0]
			tempProb[indexClf] = probValues										
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbFeature2[i])
			
		###### Running the first classifier for feature 3 #############
		operator = set3[0]			
				
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#print>>f1,probValues
			indexClf = set3.index(operator)
			tempProb = currentProbFeature3[i][0]
			tempProb[indexClf] = probValues										
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbFeature3[i])
		
	
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set1.remove(genderPredicate6)
	set2.remove(expressionPredicate6)
	set3.remove(agePredicate6)
	
	######## Finding initial quality value ######################	
	qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
	#print 'returned images'
	#print qualityOfAnswer[3]
	#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#######################################################
	
	set1 = [genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet1 = [0.85,0.93,0.80]
	costSet1 = [0.095875,0.023094, 0.866073 ]
	
	benefitSet1 = [ float(aucSet1[i])/costSet1[i] for i in range(len(aucSet1))]
	print benefitSet1
	workflow1 =[x for y, x in sorted(zip(benefitSet1, set1),reverse=True)]
	print workflow1
	 
	########################################
	
	set2 = [expressionPredicate1,expressionPredicate3,expressionPredicate7]
	aucSet2 = [0.73,0.70,0.68]
	costSet2 = [0.018030,0.020180,0.790850]
	
	benefitSet2 = [ float(aucSet2[i])/costSet2[i] for i in range(len(aucSet2))]
	print benefitSet2
	workflow2 =[x for y, x in sorted(zip(benefitSet2, set2),reverse=True)]
	print workflow2
	
	##########################################
	
	set3 = [agePredicate1,agePredicate3,agePredicate7]
	aucSet3 = [0.73,0.70,0.68]
	costSet3 = [0.018030,0.020180,0.790850]
	
	benefitSet3 = [ float(aucSet3[i])/costSet3[i] for i in range(len(aucSet3))]
	#print benefitSet2
	workflow3 =[x for y, x in sorted(zip(benefitSet3, set3),reverse=True)]
	#print workflow2
	
	
	round = 1
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		
		t1 = time.time()
	
		#print("Workflow : {} ".format(workflow))
		

		for j in range(len(dl)):
				#imageProb = probValues[j]
			for i in range(3):
				### Running function of feature 1 ###
				operator1 = workflow1[i]
				t11 = time.time()
				imageProb = operator1([dl[j]])
				t12 = time.time()
				rocProb = imageProb
				averageProbability = 0;

				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set1.index(operator1)
				tempProb = currentProbFeature1[j][0]
				tempProb[indexClf] = rocProb
				
				
				executionTime = executionTime + (t12- t11)
				
				### Running function of feature 2 ###
				#if rocProb >0.6:
				operator2 = workflow2[i]
				t11 = time.time()
				imageProb = operator2([dl[j]])
				t12 = time.time()
				rocProb = imageProb
				

				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
				
				#index of classifier
				indexClf = set2.index(operator2)
				tempProb = currentProbFeature2[j][0]
				tempProb[indexClf] = rocProb
			
				
				executionTime = executionTime + (t12- t11)
				
				### Running function of feature 3 ###
				#if rocProb >0.6:
				operator3 = workflow3[i]
				t11 = time.time()
				imageProb = operator3([dl[j]])
				t12 = time.time()
				rocProb = imageProb
				

				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
				
				#index of classifier
				indexClf = set3.index(operator3)
				tempProb = currentProbFeature3[j][0]
				tempProb[indexClf] = rocProb
			
				
				executionTime = executionTime + (t12- t11)
				
				
				
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
					#print 'returned images'
					#print qualityOfAnswer[3]
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else:
						realF1 = 0
					print 'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					#f1measure = qualityOfAnswer[0]
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					print 'time bound completed:%d'%(currentTimeBound)	
					
					
				if executionTime > budget:
						break
			
			if executionTime > budget:
					break
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbFeature1,currentProbFeature2,currentProbFeature3)
		f1measure = qualityOfAnswer[0]
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		'''
		plt.title('Quality vs Time Value for BaseLine 4')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.ylim([0, 1])
		
		plt.savefig('QualityBaseLine4GenderMuct.eps', format='eps')
		#plt.show()
		plt.close()
		'''
	
	'''
	print>>f1,"Workflow1 : {} ".format(workflow1)
	print>>f1,"Workflow2 : {} ".format(workflow2)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,findRealF1(qualityOfAnswer[3]),qualityOfAnswer[1],qualityOfAnswer[2])
	'''
	#return f1measure
	return [timeList,f1List]
	
def compareCost():
	f1 = open('UncertaintyExperiments\Results\NewAlgorithm\CostCompare.txt','w+')
	operatorList = [genderPredicate1,genderPredicate2, genderPredicate3,genderPredicate4]
	objects = [dl[k] for k in range(100)]
	for i in range(len(operatorList)):
		operator = operatorList[i]
		t11 = time.time()
		print>>f1,operator
		for j in range(100):
			#imageProb = probValues[j]
			#print '%d th image done'%(j)
			imageProb = operator(dl[j])
		t12 = time.time()
		print 'individual time:%f'%(t12-t11)
		print>>f1,'individual time:%f'%(t12-t11)
		print>>f1,'individual average time:%f'%((t12-t11)/100)
		
		
		t21 = time.time()	
		prob2 = operator(objects)
		#prob2 = operator(dl)
		t22 = time.time()
		print 'aggregated time:%f'%(t22-t21)
		print>>f1,'aggregated time:%f'%(t22-t21)
		print>>f1,'aggregated average time:%f'%((t22-t21)/100)


def generateMultipleExecutionResult():
	#createSample()
	#createRandomSample()
	f1 = open('queryTestMultiPieMultiFeature.txt','w+')
	
	#q1_all,q2_all,q3_all = [],[],[]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	
	for i in range(40):
		global dl,nl,nl2
		#dl,nl =pickle.load(open('5Samples/MuctTrainGender'+str(i)+'_XY.p','rb'))
		
		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl4)), 750))]
		
		dl_test = [dl4[i1] for i1 in  imageIndex]
		nl_test = [nl4[i1] for i1 in imageIndex]
		nl_test2 = [nl3[i1] for i1 in imageIndex]
	
		
		
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		nl2 = np. array(nl_test2)
		[t1,q1]=baseline3(60)
		[t2,q2] =baseline4(60)
		[t3,q3] =adaptiveOrder8(60)
		t1_all.append(t1)
		t2_all.append(t2)
		t3_all.append(t3)
		q1_all.append(q1)
		q2_all.append(q2)
		q3_all.append(q3)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		print>>f1,"t3 = {} ".format(t3)
		print>>f1,"q3 = {} ".format(q3)
		'''
		plt.plot(t1, q1,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
		plt.plot(t2, q2,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
		plt.plot(t3, q3,lw=2,color='blue', label='Iterative Approach') ##2,000

		plt.ylim([0, 1])
		plt.xlim([0, 20])
		plt.title('Quality vs Cost')
		plt.legend(loc="lower left",fontsize='medium')
		plt.ylabel('F1-measure')
		plt.xlabel('Cost')	
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.png', format='png')
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.eps', format='eps')
		#plt.show()
		plt.close()
		'''
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	plt.plot(t1, q1,lw=2,color='green',marker='o',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2, q2,lw=2,color='orange',marker='^',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3, q3,lw=2,color='blue',marker='d', label='Iterative Approach') ##2,000

	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
           
	plt.ylabel('F1-measure')
	plt.xlabel('Cost')	
	plt.savefig('PlotQualityComparisonMultiPieBaseline_three_feature_750_obj_120sec.png', format='png')
	plt.savefig('PlotQualityComparisonMultiPieBaseline_three_feature_750_obj_120sec.eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	
	
	min_val = min(min(q1_new),min(q2_new),min(q3_new))
	max_val = max(max(q1_new),max(q2_new),max(q3_new))
	
	
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	q3_norm = (q3_new-min_val)/(max_val - min_val)

	plt.plot(t1_new, q1_norm,lw=2,color='green',marker='o',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',marker='^',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue',marker='d', label='Iterative Approach') ##2,000
	
	'''
	plt.plot(t1_new, q1_new,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_new,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_new,lw=2,color='blue', label='Iterative Approach') ##2,000
	'''

	
	plt.title('Quality vs Cost')
	#plt.legend(loc="upper left",fontsize='medium')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('Gain')
	plt.xlabel('Cost')
	#plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])	
	plt.savefig('PlotQualityComparisonMultiPieBaseline_three_feature_750_obj_gain.png', format='png')
	plt.savefig('PlotQualityComparisonMultiPieBaseline_three_feature_750_obj_gain.eps', format='eps')
		#plt.show()
	plt.close()

	
	
	'''
	plt.plot(t1, q1,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2, q2,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3, q3,lw=2,color='blue', label='Iterative Approach') ##2,000

	plt.ylim([0, 1])
	#plt.xlim([0, 300])
	plt.title('Quality vs Cost')
	plt.legend(loc="lower left",fontsize='medium')
	plt.ylabel('F1-measure')
	plt.xlabel('Cost')	
	plt.savefig('PlotQualityComparisonMuctBaseline_gender.png', format='png')
	plt.savefig('PlotQualityComparisonMuctBaseline_gender.eps', format='eps')
	#plt.show()
	plt.close()
	'''		
	
		
	
if __name__ == '__main__':
	t1 = time.time()
	#adaptiveOrder7(200)
	setupFeature1()
	setupFeature2()
	setupFeature3()
	#adaptiveOrder8(600)
	
	generateMultipleExecutionResult()
	'''
	[t1,q1]=baseline3(600)
	[t2,q2]=baseline4(600)
	print q1
	print q2
	'''
	#adaptiveOrder9(400)
	#runOneClassifier()
	#adaptiveOrder9(400)	
	#runAllClassifiers()
	#compareCost()
	