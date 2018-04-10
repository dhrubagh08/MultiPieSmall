import sys,re
import os,pickle
import numpy as np
import csv
from skimage.feature import hog
from skimage import io
from skimage import data,color,exposure
#from sPickle import *
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



	
def getMultiPieTrainData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'MultiPie_Training.spkl'
	#rd = s_load(open(dfp,'rb'))
	
	
	with open('MaleList.csv', 'rb') as f:
		reader = csv.reader(f)
		male_list = [[x for x in rec] for rec in csv.reader(f, delimiter=',')]
		#male_list = list(reader)

	with open('FemaleList.csv', 'rb') as f:
		reader = csv.reader(f)
		female_list = [[x for x in rec] for rec in csv.reader(f, delimiter=',')]
	
	
	for fu in s_load(open(dfp)):
		namep = re.split('[_]',fu[0])
		print namep
		templist = []
		templist.append(int(namep[0]))
		if(templist in male_list):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
	
	# return dl
	print(testX)
	print(testY)
	sPickle.dump([testX,testY],open('MultiPieTrainGender_XY.p','wb'))
	
def getMultiPieValidationData():
	testX = []
	testY = []
	
	dfp = 'MultiPie_Testing6.spkl'
	#rd = pickle.load(open(dfp,'rb'))
	
	ff = open('MultiPie_Test_XY.spkl','w')

	count1 = 0 
	with open('MaleList.csv', 'rb') as f:
		f.next()	# to remove the header of csv file
		reader = csv.reader(f)
		male_list = [[int(x) for x in rec] for rec in reader]
		#male_list = list(reader)
				
	with open('FemaleList.csv', 'rb') as f:
		f.next() 	
		reader = csv.reader(f)
		female_list = [[int(x) for x in rec] for rec in reader]
	
	print male_list
	print female_list
	
	count =0
	for fu in s_load(open(dfp,'rb')):
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
		count = count + 1
		if (count%1000 ==0):
			print testX
			print testY
			print ([testX,testY])
			s_dump_elt([testX,testY],ff)
			testX = []
			testY = []
			#break		
		
	# return dl
	print(testX)
	print(testY)
	#pickle.dump([testX,testY],open('MultiPieTestingGender_XY.p','wb'))
	s_dump([testX,testY],ff)
	
	
	
def getMultiPieValidationData2():	
	testX = []
	testY = []

	
	dfp = 'Multipie_Hog_Small_Test_Memory.pkl'
	rd = pickle.load(open(dfp,'rb'))
	
		
	count1 = 0 
	with open('EthnicityAmerican.csv', 'rb') as f:
		f.next()	# to remove the header of csv file
		reader = csv.reader(f)
		male_list = [[int(x) for x in rec] for rec in reader]
		#male_list = list(reader)
				
	with open('EthnicityNonAmerican.csv', 'rb') as f:
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
		count1=count1+1
		if count1==100:
			break
		
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MultiPieTestGender2_XY.p','wb'))
	#s_dump([testX,testY],ff)
		
	
def getMultiPieTestData():
	testX = []
	testY = []
	#yaleDirectory = 'E:\Codes\Codes\ExtendedYaleB\ExtendedYaleB\TestImageSmall'
	dfp = 'MultiPie_Testing6.pkl'
	rd = pickle.load(open(dfp,'rb'))
	
	with open('MaleList.csv', 'rb') as f:
		reader = csv.reader(f)
		male_list = [[x for x in rec] for rec in csv.reader(f, delimiter=',')]
		#male_list = list(reader)

	with open('FemaleList.csv', 'rb') as f:
		reader = csv.reader(f)
		female_list = [[x for x in rec] for rec in csv.reader(f, delimiter=',')]
	
	for fu in rd:
		namep = re.split('[_]',fu[0])
		templist = []
		templist.append(int(namep[0]))
		#print templist
		if(templist in male_list):
			testX.append(fu[2])
			testY.append(1)
		else:
			testX.append(fu[2])
			testY.append(0)
	# return dl
	print(testX)
	print(testY)
	pickle.dump([testX,testY],open('MultiPieTestGender_XY.p','wb'))
    


if __name__ =='__main__':
	#getMultiPieTrainData()
	#getMultiPieValidationData()
	getMultiPieValidationData2()
	#getMuctTestData()
