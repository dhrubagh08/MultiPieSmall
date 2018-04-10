import os.path
import csv
from os.path import isfile, join
import numpy as np
from operator import itemgetter


def generateUncertaintyValuesInitial():
	data = list(csv.reader(open('AllProbabilities.csv')))
	print data
	data = [[float(y.decode('utf8').encode('ascii', errors='ignore')) for y in x] for x in data]
	uncertainty1 = []
	uncertainty2 = []
	
	
	for i in range(len(data)):
		uncertaintyValue1 =1  # all the images have initial uncertainty of 1	
		minUncertaintyValue = 1
		minUncertaintyIndex = 10
		for k in range(1,len(data[0])):	
			combinedProbability = float(0.5+data[i][k])/2
			if(combinedProbability !=0 and combinedProbability !=1):
				uncertaintyValue2 = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
			else:
				uncertaintyValue2 =0
				
			if uncertaintyValue2 < minUncertaintyValue:
				minUncertaintyValue = uncertaintyValue2
				minUncertaintyIndex = k
		if minUncertaintyIndex == 1:
			minUncertaintyIndexValue = 'DT'
		if minUncertaintyIndex == 2:
			minUncertaintyIndexValue = 'GNB'
		if minUncertaintyIndex == 3:
			minUncertaintyIndexValue = 'RF'
		if minUncertaintyIndex == 4:
			minUncertaintyIndexValue = 'KNN'
		tempValue = [uncertaintyValue1,minUncertaintyValue-uncertaintyValue1,minUncertaintyIndexValue]
		uncertainty2.append(tempValue)
	uncertainty2 = sorted(uncertainty2, key=itemgetter(0))
	
	with open("listInitial.csv",'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerow((['From Initial']))
		wr.writerows(uncertainty2)
	

def generateUncertaintyValues1(currentDecidingIndex,outputFile):
	
	data = list(csv.reader(open('AllProbabilities.csv')))
	data = [[float(y.decode('utf8').encode('ascii', errors='ignore')) for y in x] for x in data]
	uncertainty1 = []
	uncertainty2 = []
	#currentDecidingIndex = 1
	
	for i in range(len(data)):
		if( data[i][currentDecidingIndex] !=1 and data[i][currentDecidingIndex] !=0):
			#print 'data[i][currentDecidingIndex] : %f'%(data[i][currentDecidingIndex])
			uncertaintyValue1 = -data[i][currentDecidingIndex]* np.log2(data[i][currentDecidingIndex]) - (1- data[i][currentDecidingIndex])* np.log2(1- data[i][currentDecidingIndex])
			#print 'uncertainty1 : %f'%(uncertaintyValue1)
		else:
			uncertaintyValue1 =0
		uncertainty1.append(uncertaintyValue1)
		
		minUncertaintyValue = 1
		minUncertaintyIndex = 10
		
		for k in range(1,len(data[0])):
			if k !=currentDecidingIndex :
				#combinedProbability = 1- (1-data[i][3])*(1-data[i][k]) # If we use Noisy-OR model, then this is combined probability
				combinedProbability = float(data[i][currentDecidingIndex]+data[i][k])/2   # If we use Average Probability model, then this is combined probability
				if(combinedProbability !=0 and combinedProbability !=1):
					uncertaintyValue2 = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
				else:
					uncertaintyValue2 =0
					
				if uncertaintyValue2 < minUncertaintyValue:
					minUncertaintyValue = uncertaintyValue2
					minUncertaintyIndex = k
		if minUncertaintyIndex == 1:
			minUncertaintyIndexValue = 'DT'
		if minUncertaintyIndex == 2:
			minUncertaintyIndexValue = 'GNB'
		if minUncertaintyIndex == 3:
			minUncertaintyIndexValue = 'RF'
		if minUncertaintyIndex == 4:
			minUncertaintyIndexValue = 'KNN'
		tempValue = [uncertaintyValue1,minUncertaintyValue-uncertaintyValue1,minUncertaintyIndexValue]
		uncertaintyValue1 = 0
		uncertaintyValue2 = 0
		uncertainty2.append(tempValue)
		
		
	#print uncertainty1
	#print uncertainty2
	if currentDecidingIndex ==1:
		firstRow = 'From DT'
	if currentDecidingIndex ==2:
		firstRow = 'From GNB'
	if currentDecidingIndex ==3:
		firstRow = 'From RF'
	if currentDecidingIndex ==4:
		firstRow = 'From KNN'
	
	uncertainty2 = sorted(uncertainty2, key=itemgetter(0))
	with open(outputFile,'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerow(([firstRow]))
		wr.writerows(uncertainty2)
	#print uncertainty2
	
def generateUncertaintyValues2(decidingIndex1,decidingIndex2,outputFile):
	
	data = list(csv.reader(open('AllProbabilities.csv')))
	data = [[float(y.decode('utf8').encode('ascii', errors='ignore')) for y in x] for x in data]
	uncertainty1 = []
	uncertainty2 = []
	#decidingIndex1 = 3
	#decidingIndex2 = 4
	
	for i in range(len(data)):
		#combinedProbability1 = 1- (1-data[i][decidingIndex1])*(1-data[i][decidingIndex2])
		combinedProbability1 = float(data[i][decidingIndex1]+data[i][decidingIndex2])/2
				
		if( combinedProbability1 !=1 and combinedProbability1 !=0):
			uncertaintyValue1 = -combinedProbability1* np.log2(combinedProbability1) - (1- combinedProbability1)* np.log2(1- combinedProbability1)
		else:
			uncertaintyValue1 =0
		uncertainty1.append(uncertaintyValue1)
		
		minUncertaintyValue = 1
		minUncertaintyIndex = 10
		
		for k in range(1,len(data[0])):
			if k !=decidingIndex1 and k!=decidingIndex2 :
				#combinedProbability = 1- (1-data[i][decidingIndex1])*(1-data[i][decidingIndex2])*(1-data[i][k])
				combinedProbability = float(data[i][decidingIndex1]+data[i][decidingIndex2] + data[i][k])/3
				if(combinedProbability !=0 and combinedProbability !=1):
					uncertaintyValue2 = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
				else:
					uncertaintyValue2 =0
					
				if uncertaintyValue2 < minUncertaintyValue:
					minUncertaintyValue = uncertaintyValue2
					minUncertaintyIndex = k
		if minUncertaintyIndex == 1:
			minUncertaintyIndexValue = 'DT'
		if minUncertaintyIndex == 2:
			minUncertaintyIndexValue = 'GNB'
		if minUncertaintyIndex == 3:
			minUncertaintyIndexValue = 'RF'
		if minUncertaintyIndex == 4:
			minUncertaintyIndexValue = 'KNN'
		tempValue = [uncertaintyValue1,minUncertaintyValue-uncertaintyValue1,minUncertaintyIndexValue]
		
		uncertainty2.append(tempValue)
	#print uncertainty1
	#print uncertainty2
	firstRow =''
	if decidingIndex1 ==1 and decidingIndex1 ==2:
		firstRow = 'From DT,GNB'
	if decidingIndex1 ==1 and decidingIndex2 ==3:
		firstRow = 'From DT,RF'
	if decidingIndex1 ==1 and decidingIndex2 ==4:
		firstRow = 'From DT,KNN'
	if decidingIndex1 ==2 and decidingIndex2 ==3:
		firstRow = 'From GNB,RF'
	if decidingIndex1 ==2 and decidingIndex2 ==4:
		firstRow = 'From GNB,KNN'
	if decidingIndex1 ==3 and decidingIndex2 ==4:
		firstRow = 'From RF,KNN'
	
		
	uncertainty2 = sorted(uncertainty2, key=itemgetter(0))
	with open(outputFile,'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerow(([firstRow]))
		wr.writerows(uncertainty2)
	#print uncertainty2

def generateUncertaintyValues3(decidingIndex1,decidingIndex2,decidingIndex3,outputFile):
	
	data = list(csv.reader(open('AllProbabilities.csv')))
	data = [[float(y.decode('utf8').encode('ascii', errors='ignore')) for y in x] for x in data]
	uncertainty1 = []
	uncertainty2 = []
	
	#decidingIndex1 = 2
	#decidingIndex2 = 3
	#decidingIndex3 = 4
	
	for i in range(len(data)):
		#combinedProbability1 = 1- (1-data[i][2])*(1-data[i][3])*(1-data[i][4])
		combinedProbability1 = float(data[i][decidingIndex1]+data[i][decidingIndex2]+data[i][decidingIndex3])/3
		
		if( combinedProbability1 !=1 and combinedProbability1 !=0):
			uncertaintyValue1 = -combinedProbability1* np.log2(combinedProbability1) - (1- combinedProbability1)* np.log2(1- combinedProbability1)
		else:
			uncertaintyValue1 =0
		uncertainty1.append(uncertaintyValue1)
		
		minUncertaintyValue = 1
		minUncertaintyIndex = 10
		
		for k in range(1,len(data[0])):
			if k !=decidingIndex1 and k!=decidingIndex2 and k!=decidingIndex3 :
				#combinedProbability = 1- (1-data[i][2])*(1-data[i][3])*(1-data[i][4])*(1-data[i][k])
				combinedProbability = float(data[i][decidingIndex1]+data[i][decidingIndex2]+data[i][decidingIndex3] + data[i][k])/4
				
				if(combinedProbability !=0 and combinedProbability !=1):
					uncertaintyValue2 = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
				else:
					uncertaintyValue2 =0
					
				if uncertaintyValue2 < minUncertaintyValue:
					minUncertaintyValue = uncertaintyValue2
					minUncertaintyIndex = k
		if minUncertaintyIndex == 1:
			minUncertaintyIndexValue = 'DT'
		if minUncertaintyIndex == 2:
			minUncertaintyIndexValue = 'GNB'
		if minUncertaintyIndex == 3:
			minUncertaintyIndexValue = 'RF'
		if minUncertaintyIndex == 4:
			minUncertaintyIndexValue = 'KNN'
		tempValue = [uncertaintyValue1,minUncertaintyValue-uncertaintyValue1,minUncertaintyIndexValue]
		
		uncertainty2.append(tempValue)
	#print uncertainty1
	#print uncertainty2
	firstRow =''
	if decidingIndex1 ==1 and decidingIndex1 ==2 and decidingIndex1 ==3:
		firstRow = 'From DT,GNB,RF'
	if decidingIndex1 ==1 and decidingIndex1 ==2 and decidingIndex1 ==4:
		firstRow = 'From DT,GNB,KNN'
	if decidingIndex1 ==1 and decidingIndex1 ==3 and decidingIndex1 ==4:
		firstRow = 'From DT,RF,KNN'
	if decidingIndex1 ==2 and decidingIndex1 ==3 and decidingIndex1 ==4:
		firstRow = 'From GNB,RF,KNN'
		
	uncertainty2 = sorted(uncertainty2, key=itemgetter(0))
	with open(outputFile,'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerow(([firstRow]))
		wr.writerows(uncertainty2)
	#print uncertainty2

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump	
	


def generateBins(inputFileName, outputFileName):	
	data = list(csv.reader(open(inputFileName)))
	data = [[y for y in x] for x in data]
	data.pop(0)	#removed the header
	#print data
	currentUncertainty = [float(x[:][0]) for x in data]
	deltaUncertainty = [float(x[:][1]) for x in data]
	nextBestClf = [x[:][2] for x in data]
	#print currentUncertainty
	#print deltaUncertainty
	#print nextBestClf
	
	#maxClf = max(nextBestClf,key=nextBestClf.count)
	#print maxClf
	
	subListU,subListDeltaU,sublistClf = [],[],[]
	finalU,finalClf,finalDelta,finalList= [],[],[],[]
	for i in frange(0.1,1,0.1):
		for x, y, z in zip(currentUncertainty, deltaUncertainty, nextBestClf):
			if i!=0.1 : 
				if (i-0.1)<x<=i:
					subListU.append(x)
					subListDeltaU.append(y)
					sublistClf.append(z)
			else:
				if x<=i:
					subListU.append(x)
					subListDeltaU.append(y)
					sublistClf.append(z)
		#print subListU
		#print subListDeltaU
		#print sublistClf
		if len(sublistClf)!=0:
			maxClf = max(sublistClf,key=sublistClf.count)
			print maxClf
			
			#Choosing values based on bestClassfiers
			sum = 0
			count=0
			for a,b in zip(subListDeltaU,sublistClf):
				if(b == maxClf):
					sum= sum+ a
					count= count+1
			print 'sum:%f, count:%d'%(sum,count)
			averageDelta= float(sum)/count
			finalU.append(i)
			finalClf.append(maxClf)
			finalDelta.append(averageDelta)
			finalValue= [i,maxClf,averageDelta]
			finalList.append(finalValue)
		
		subListU[:]=[] 
		subListDeltaU[:]=[]
		sublistClf[:]=[]
		
		
	
	#print finalU
	#print finalClf
	#print finalDelta
	#print finalList
	
	with open(outputFileName,'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		#wr.writerow((['From LR,RF,AB']))
		wr.writerows(finalList)
	
	#bin_width = int(round(duration / bins))
	#followers = [sum(followersList[i:i+bin_width]) for i in xrange(0, duration, bin_width)]
	
def generateBinsInitial():	
	data = list(csv.reader(open('listInitial.csv')))
	data = [[y for y in x] for x in data]
	data.pop(0)	#removed the header
	
	currentUncertainty = [float(x[:][0]) for x in data]
	deltaUncertainty = [float(x[:][1]) for x in data]
	nextBestClf = [x[:][2] for x in data]
	
	maxClf = max(nextBestClf,key=nextBestClf.count)
	
	sum = 0
	count=0
	finalU,finalClf,finalDelta,finalList= [],[],[],[]
	
	for a,b in zip(deltaUncertainty,nextBestClf):
		if(b == maxClf):
			sum= sum+ a
			count= count+1
	print 'sum:%f, count:%d'%(sum,count)
	averageDelta= float(sum)/count
	finalU.append(1)
	finalClf.append(maxClf)
	finalDelta.append(averageDelta)
	finalValue= [1,maxClf,averageDelta]
	finalList.append(finalValue)
		
	
	print finalList
	
	with open("listInitialDetails.csv",'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerows(finalList)
	
		
	
								 
								 
								 
if __name__ =='__main__':
	
	
	generateUncertaintyValuesInitial()
	generateUncertaintyValues1(1,'list0.csv')
	generateUncertaintyValues1(2,'list1.csv')
	generateUncertaintyValues1(3,'list2.csv')
	generateUncertaintyValues1(4,'list3.csv')
	
	generateUncertaintyValues2(1,2,'list01.csv')
	generateUncertaintyValues2(1,3,'list02.csv')
	generateUncertaintyValues2(1,4,'list03.csv')
	generateUncertaintyValues2(2,3,'list12.csv')
	generateUncertaintyValues2(2,4,'list13.csv')
	generateUncertaintyValues2(3,4,'list23.csv')
	
	generateUncertaintyValues3(1,2,3,'list012.csv')
	generateUncertaintyValues3(1,2,4,'list013.csv')
	generateUncertaintyValues3(1,3,4,'list023.csv')
	generateUncertaintyValues3(2,3,4,'list123.csv')
	
	
	#generateBins('listInitial.csv','listInitialDetails.csv')
	
	generateBins('list0.csv','list0Details.csv')
	generateBins('list1.csv','list1Details.csv')
	generateBins('list2.csv','list2Details.csv')
	generateBins('list3.csv','list3Details.csv')
	generateBins('list01.csv','list01Details.csv')
	generateBins('list02.csv','list02Details.csv')
	generateBins('list03.csv','list03Details.csv')
	generateBins('list12.csv','list12Details.csv')
	generateBins('list13.csv','list13Details.csv')
	generateBins('list23.csv','list23Details.csv')
	generateBins('list012.csv','list012Details.csv')
	generateBins('list013.csv','list013Details.csv')
	generateBins('list023.csv','list023Details.csv')
	generateBins('list123.csv','list123Details.csv')
	generateBinsInitial()
	
	
	#StoreFileMetadataLearning()