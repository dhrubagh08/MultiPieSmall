import sys,re
import os,pickle
import numpy as np
import csv
from skimage.feature import hog
from skimage import io
#from skimage import data,color,exposure



def getMultiPieValidationData2():	
	testX = []
	testY = []

	
	dfp = 'Multipie_Hog_Small_Validation_Memory.pkl'
	rd = pickle.load(open(dfp,'rb'))
	
		
	count1 = 0 
	#with open('MaleList.csv', 'rb') as f:
	with open('Age_greater_equal30_list.csv', 'rb') as f:
		f.next()	# to remove the header of csv file
		reader = csv.reader(f)
		male_list = [[int(x) for x in rec] for rec in reader]
		#male_list = list(reader)
				
	#with open('FemaleList.csv', 'rb') as f:
	with open('Age_less_30.csv', 'rb') as f:	
		f.next() 	
		reader = csv.reader(f)
		female_list = [[int(x) for x in rec] for rec in reader]
	
	print male_list
	print female_list
	
	count =0
	count1=0
	for fu in rd:
		
		namep = re.split('[_]',fu[0])
		templist = []
		templist.append(int(namep[0]))
		print namep
		if(templist in male_list):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
		'''	
		count1=count1+1
		if count1==100:
			break
		'''
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MultiPieValidationAge2_XY.p','wb'))
	#pickle.dump([testX,testY],open('MultiPieTestGender2_XY.p','wb'))
	#s_dump([testX,testY],ff)
		


if __name__ =='__main__':
	#getMultiPieTrainData()
	#getMultiPieValidationData()
	getMultiPieValidationData2()
	#getMuctTestData()
