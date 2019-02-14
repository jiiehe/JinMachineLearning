
import numpy as np


def checkDuplicate(L):
    arr=[]
    for i in range(len(L)):
        if L[i] not in arr:
            arr.append(L[i])

    return arr, len(arr)

def ID3(labels,size,terms,attributionSize,checkActive,type,location,depth,branch):
    
   
    if type == 'Entropy' :
        Measures = calculateEntropy(labels,size)
    elif type == 'ME' :
        Measures = calculateME(labels,size)
    else :
        Measures = calculateGI(labels,size)
    
   
    ConditionMeasures = np.zeros((np.size(attributionSize),))
    

    examples = float(np.size(labels))
    
  
    infoVariables = float(-1);
 
    attributionSplite = -1
    

    for i in checkActive:
        ConditionMeasures = 0
  
        for j in range(attributionSize[i]):
          
            locs = np.where((terms[:,i]==j))[0]
            labls = labels[locs]
            if type == 'Entropy' :
                conditionalmeasures = calculateEntropy(labls,size)
            elif type == 'ME' :
                conditionalmeasures = calculateME(labls,size)
            else :
                conditionalmeasures = calculateGI(labls,size)
            ConditionMeasures = ConditionMeasures + conditionalmeasures*float(np.size(locs))/examples
        variable_temp = Measures-ConditionMeasures
     
        if variable_temp > infoVariables :
            infoVariables = variable_temp
            attributionSplite = i
    
    branchTree = np.zeros((1,np.max(attributionSize)))
    
    attributionTree =[ [attributionSplite  ] [-1]]
    
    checkActive = np.delete(checkActive,np.where(checkActive == attributionSplite)[0][0])
    

    for j in range(attributionSize[attributionSplite]) :
   
        locs = np.where((terms[:,attributionSplite]==j))[0]
        labls = labels[locs]
  
        if np.size(labls) == 0:
            (values,counts) = checkDuplicate(labels)
            branchTree[0,j] = -1*values[np.argmax(counts)]-1
        elif np.size(np.unique(labls)) == 1 or location == depth or np.size(checkActive) == 0 :
            (values,counts) = checkDuplicate(labls)
            branchTree[0,j] = -1*values[np.argmax(counts)]-1
     
        else :
            branch = branch + 1
            branchTree[0,j] = branch
            AT, BT, branch = ID3(labls,size,terms[locs,:],attributionSize,checkActive,type,location+1,depth,branch)
            attributionTree = np.vstack((attributionTree,AT))
            branchTree = np.vstack((branchTree,BT))
            
    
    return attributionTree, branchTree, branch




def calculateGI(A,B):
    C = np.shape(A)[0]
    getCalculation = float(1)
    if C > 0 :
        for i in range(B):
            D = float(np.size(np.where((A==i))[0]))/ float(C)
            getCalculation = getCalculation - D**2
    return getCalculation

def calculateME(A,B):
    C = np.shape(A)[0]
    getME = float(0)
    maximum = float(0)
    if C > 0 :
        for i in range(B):
            D = float(np.size(np.where((A==i))[0]))/ float(C)
            if D > maximum :
                maximum = D
    getME = 1-maximum
    return getME




def calculateEntropy(A,B) :
    C = np.shape(A)[0]
    D = float(0)
    if C > 0 :
        for i in range(B):
            E = float(np.size(np.where((A==i))[0]))/ float(C)
            if E > 0 :
                D = D - E*np.log2(E)
    return D






def question2():
	depth = 6
	types = 'GI'
        print("the answer for question 2 is below:")
	print('the current Measurement Type is:',types)
	filePath = 'train.csv'
	count = 0
	with open (filePath , 'r') as eachLine :
		for line in eachLine :
			terms = line.strip().split(',')
			if count == 0:
				numberAttributes = np.shape(terms)[0]-1
				attributesValues = np.vstack((terms[0:numberAttributes],numberAttributes*['']))
				labelValues = terms[-1]
				labelSize = 1
				labels = 0
				labelIndex = 1
				attributeSize =  np.ones((numberAttributes,),dtype = int)
				allterms = np.zeros((numberAttributes,),dtype = int)
				termIndex = np.zeros((numberAttributes,),dtype = int)
			else:
				if terms[-1] not in labelValues:
					labelValues = np.vstack((labelValues,terms[-1]))
					labelSize = labelSize + 1
				labelIndex = np.where(terms[-1]==labelValues)[0]
				labels = np.vstack((labels,labelIndex))
				for i in range(numberAttributes):
					if terms[i] not in attributesValues[0:attributeSize[i],i] :
						attributeSize[i] = attributeSize[i] + 1
						if attributeSize[i]  > np.shape(attributesValues)[0]:
							attributesValues = np.vstack((attributesValues,numberAttributes*['']))
						attributesValues[attributeSize[i]-1,i] = terms[i]
					termIndex[i] = np.where(terms[i]==attributesValues[0:attributeSize[i]+1,i])[0][0]
				allterms = np.vstack((allterms,termIndex))
			count = count+1
	continueBranch = 0
	activeAttribute = np.arange(numberAttributes)
	treeAttributions, treeBranches, continueBranch = ID3(labels,labelSize,allterms,attributeSize,activeAttribute,types,1,depth,continueBranch)
	examplesSet = np.shape(allterms)[0]
	outcomeTree = np.zeros((examplesSet,),dtype = int)
	countError = 0;
	for i in range(examplesSet) :
		outcomeTree[i] = treeBranches[0,allterms[i,treeAttributions[0]]]
		while outcomeTree[i] >= 0:
			outcomeTree[i] = treeBranches[outcomeTree[i],allterms[i,treeAttributions[outcomeTree[i]]]]
		outcomeTree[i] = -1*(outcomeTree[i]+1)
		if outcomeTree[i] != labels[i]:
			countError = countError + 1        
	erroroftrain = float(countError)/float(examplesSet)
	print('after calcuation, the training error should be:', erroroftrain)
	filePath = 'test.csv'
	allterms_test = np.zeros((0,numberAttributes),dtype = int)
	labels_test = np.zeros((0,1),dtype = int)
	with open (filePath , 'r') as eachLine :
		for line in eachLine :
			terms = line.strip().split(',')
			if terms[-1] not in labelValues:
				labelValues = np.vstack((labelValues,terms[-1]))
				labelSize = labelSize + 1
			labelIndex = np.where(terms[-1]==labelValues)[0]
			labels_test = np.vstack((labels_test,labelIndex))
			for i in range(numberAttributes):
				if terms[i] not in attributesValues[0:attributeSize[i],i] :
					attributeSize[i] = attributeSize[i] + 1
					if attributeSize[i]  > np.shape(attributesValues)[0]:
						attributesValues = np.vstack((attributesValues,numberAttributes*['']))
					attributesValues[attributeSize[i]-1,i] = terms[i]
				termIndex[i] = np.where(terms[i]==attributesValues[0:attributeSize[i]+1,i])[0][0]
			allterms_test = np.vstack((allterms_test,termIndex))
	examplesSet = np.shape(allterms_test)[0]
	outComeTest = np.zeros((examplesSet,),dtype = int)
	countError = 0;
	for i in range(examplesSet) :
	   outComeTest[i] = treeBranches[0,allterms_test[i,treeAttributions[0]]]
	   while outComeTest[i] >= 0:
			outComeTest[i] = treeBranches[outComeTest[i],allterms_test[i,treeAttributions[outComeTest[i]]]]
	   outComeTest[i] = -1*(outComeTest[i]+1)
	   if outComeTest[i] != labels_test[i]:
			countError = countError + 1
	test_error = float(countError)/float(examplesSet)
	print('after calculation, the Test Error should be:', test_error)
def question3a():
	depth = 9
	types = 'Entropy'
        print('the answer for question 3 a is below:')
	print('the current depth is:', depth)
	print('the current measurement type:',types)


	filePath = 'train1.csv'


	numAttributions = np.array([0,5,9,11,12,13,14],dtype = int)

	count = 0
	with open (filePath , 'r') as lines :
		for line in lines :
        
			terms = line.strip().split(',')
			if count == 0:
		
				numAttribution = np.shape(terms)[0]-1
			    
				attributionValues = np.vstack((terms[0:numAttribution],numAttribution*[''])).astype('U15')
			    
				labelValues = terms[-1]
				
				labelSize = 1
			
				labels = 0
				
				labelIndex = 1
			
				attributionSize =  np.ones((numAttribution,),dtype = int)
			    
				allterms = np.zeros((numAttribution,),dtype = int)
				
				tempAttributions = np.array(terms,dtype = 'U15')
			
				termsIndex = np.zeros((numAttribution,),dtype = int)
			else:
				
				if terms[-1] not in labelValues:
					labelValues = np.vstack((labelValues,terms[-1]))
					labelSize = labelSize + 1
				labelIndex = np.where(terms[-1]==labelValues)[0]
				labels = np.vstack((labels,labelIndex))
				
				for i in range(numAttribution):
					
					if (terms[i] not in attributionValues[0:attributionSize[i],i]) and (i not in numAttributions) :
						attributionSize[i] = attributionSize[i] + 1
						if attributionSize[i]  > np.shape(attributionValues)[0]:
							attributionValues = np.vstack((attributionValues,numAttribution*['']))
						attributionValues[attributionSize[i]-1,i] = terms[i]
					if i not in numAttributions :
						termsIndex[i] = np.where(terms[i]==attributionValues[0:attributionSize[i]+1,i])[0][0]
					else :
						termsIndex[i] = int(terms[i])
				allterms = np.vstack((allterms,termsIndex))
            
			count = count+1
	count=0
	med = np.zeros((np.size(numAttributions),),dtype = float)
	for i in numAttributions :
		med[count] = np.median(allterms[:,i])
		attributionSize[i]=2
		attributionValues[0:2,i] = [0,1]
		allterms[:,i] = allterms[:,i] > med[count]
		count=count+1

	
	branches = 0

	activeAttributions = np.arange(numAttribution)

	treeAttributions, treeBranches, branches = ID3(labels,labelSize,allterms,attributionSize,activeAttributions,types,1,depth,branches)

    

	exampleSet = np.shape(allterms)[0]
	outcomeTest = np.zeros((exampleSet,),dtype = int)
	countofError = 0;

	for i in range(exampleSet) :
		outcomeTest[i] = treeBranches[0,allterms[i,treeAttributions[0]]]
		while outcomeTest[i] >= 0:
			outcomeTest[i] = treeBranches[outcomeTest[i],allterms[i,treeAttributions[outcomeTest[i]]]]
		outcomeTest[i] = -1*(outcomeTest[i]+1)
		if outcomeTest[i] != labels[i]:
			countofError = countofError + 1
        
	errorofTrain = float(countofError)/float(exampleSet)
	print('after calcuation, the training error should be:', errorofTrain)

    
	filePath = 'test1.csv'

	
	termTest = np.zeros((0,numAttribution),dtype = int)
	labelTest = np.zeros((0,1),dtype = int)
	with open (filePath , 'r') as lines :
		for line in lines :
			terms = line.strip().split(',')
			if terms[-1] not in labelValues:
				labelValues = np.vstack((labelValues,terms[-1]))
				labelSize = labelSize + 1
			labelIndex = np.where(terms[-1]==labelValues)[0]
			labelTest = np.vstack((labelTest,labelIndex))
			for i in range(numAttribution):
				if (terms[i] not in attributionValues[0:attributionSize[i],i]) and (i not in numAttributions) :
					attributionSize[i] = attributionSize[i] + 1
					if attributionSize[i]  > np.shape(attributionValues)[0]:
						attributionValues = np.vstack((attributionValues,numAttribution*['']))
					attributionValues[attributionSize[i]-1,i] = terms[i]
				if i not in numAttributions :
					termsIndex[i] = np.where(terms[i]==attributionValues[0:attributionSize[i]+1,i])[0][0]
				else :
					termsIndex[i] = int(terms[i]) > med[np.where(i == numAttributions)[0][0]]
			termTest = np.vstack((termTest,termsIndex))

    
	exampleSet = np.shape(termTest)[0]

	outComeTest = np.zeros((exampleSet,),dtype = int)
	
	countofError = 0;

	for i in range(exampleSet) :
	   outComeTest[i] = treeBranches[0,termTest[i,treeAttributions[0]]]
	   while outComeTest[i] >= 0:
			outComeTest[i] = treeBranches[outComeTest[i],termTest[i,treeAttributions[outComeTest[i]]]]
	   outComeTest[i] = -1*(outComeTest[i]+1)
	   if outComeTest[i] != labelTest[i]:
			countofError = countofError + 1
        
	testError = float(countofError)/float(exampleSet)
	print('after calculation, the Test Error should be:', testError)




def question3b():
	depth = 1
	types = 'GI'
        print('the answer for question 3 b is below:')
	print('the current depth is:', depth)
	print('the current measurement type:',types)


	filePath = 'train1.csv'

	
	numAttributions = np.array([0,5,9,11,12,13,14],dtype = int)
	categoAttributions = np.array([1,2,3,4,6,7,8,10,15],dtype = int)

	count = 0
	with open (filePath , 'r') as lines :
		for line in lines :
        
			terms = line.strip().split(',')
			if count == 0:
			
				numAttribution = np.shape(terms)[0]-1
			
				attributionValues = np.vstack((terms[0:numAttribution],numAttribution*[''])).astype('U15')
				
				labelValues = terms[-1]
				
				labelSize = 1
				
				labels = 0
			
				labelIndex = 1
				
				attributionSize =  np.ones((numAttribution,),dtype = int)
				
				allterms = np.zeros((numAttribution,),dtype = int)
				
				termAttributions = np.array(terms,dtype = 'U15')
				termsIndex = np.zeros((numAttribution,),dtype = int)
				for i in range(numAttribution):
					if 'unknown' == terms[i]:
						attributionSize[i]=0
			else:
			
				if terms[-1] not in labelValues:
					labelValues = np.vstack((labelValues,terms[-1]))
					labelSize = labelSize + 1
				labelIndex = np.where(terms[-1]==labelValues)[0]
				labels = np.vstack((labels,labelIndex))
		
				for i in range(numAttribution):
				
					if (terms[i] not in attributionValues[0:attributionSize[i],i]) and (i not in numAttributions) and ('unknown' != terms[i]):
						attributionSize[i] = attributionSize[i] + 1
						if attributionSize[i]  > np.shape(attributionValues)[0]:
							attributionValues = np.vstack((attributionValues,numAttribution*['']))
						attributionValues[attributionSize[i]-1,i] = terms[i]
					if i not in numAttributions :
						if ('unknown' != terms[i]):
							termsIndex[i] = np.where(terms[i]==attributionValues[0:attributionSize[i]+1,i])[0][0]
						else :
							termsIndex[i] = -1
					else :
						termsIndex[i] = int(terms[i])
				allterms = np.vstack((allterms,termsIndex))
            
			count = count+1

	count = 0
	mode = np.zeros((np.size(categoAttributions),),dtype = int)
	for i in categoAttributions :
		L = allterms[:,i]
		if np.any(L== -1):
			L = np.delete(L,np.where(L == -1)[0])
		(values,counts) = checkDuplicate(L)
		mode[count] = values[np.argmax(counts)]
		allterms[np.where(-1 == allterms[:,i]),i] = mode[count]
		count = count + 1

	count=0
	med = np.zeros((np.size(numAttributions),),dtype = float)
	for i in numAttributions :
		med[count] = np.median(allterms[:,i])
		attributionSize[i]=2
		attributionValues[0:2,i] = [0,1]
		allterms[:,i] = allterms[:,i] > med[count]
		count=count+1

	
	branches = 0

	activeAttributions = np.arange(numAttribution)

	treeAttributions, treeBranches, branches = ID3(labels,labelSize,allterms,attributionSize,activeAttributions,types,1,depth,branches)

	

	exampleSet = np.shape(allterms)[0]
	outcomeTree = np.zeros((exampleSet,),dtype = int)
	countErrors = 0;

	for i in range(exampleSet) :
		outcomeTree[i] = treeBranches[0,allterms[i,treeAttributions[0]]]
		count=0
		while outcomeTree[i] >= 0 and count <2*depth+2:
			count=count+1
			if count == depth + 2:
				print(allterms[i,:])
			outcomeTree[i] = treeBranches[outcomeTree[i],allterms[i,treeAttributions[outcomeTree[i]]]]
		outcomeTree[i] = -1*(outcomeTree[i]+1)
		if outcomeTree[i] != labels[i]:
			countErrors = countErrors + 1
        
	trainErrors = float(countErrors)/float(exampleSet)
	print('after calcuation, the training error should be:', trainErrors)

	filePath = 'test1.csv'


	termTest = np.zeros((0,numAttribution),dtype = int)
	labelTest = np.zeros((0,1),dtype = int)
	with open (filePath , 'r') as lines :
		for line in lines :
			terms = line.strip().split(',')
			if terms[-1] not in labelValues:
				labelValues = np.vstack((labelValues,terms[-1]))
				labelSize = labelSize + 1
			labelIndex = np.where(terms[-1]==labelValues)[0]
			labelTest = np.vstack((labelTest,labelIndex))
			for i in range(numAttribution):
				if terms[i] == 'unknown':
					termsIndex[i] = mode[np.where(i== categoAttributions)[0][0]]
				else:
                
					if (terms[i] not in attributionValues[0:attributionSize[i],i]) and (i not in numAttributions) :
						attributionSize[i] = attributionSize[i] + 1
						if attributionSize[i]  > np.shape(attributionValues)[0]:
							attributionValues = np.vstack((attributionValues,numAttribution*['']))
						attributionValues[attributionSize[i]-1,i] = terms[i]
					if i not in numAttributions :
						termsIndex[i] = np.where(terms[i]==attributionValues[0:attributionSize[i]+1,i])[0][0]
					else :
						termsIndex[i] = int(terms[i]) > med[np.where(i == numAttributions)[0][0]]
			termTest = np.vstack((termTest,termsIndex))


	exampleSet = np.shape(termTest)[0]
	
	outcomeTest = np.zeros((exampleSet,),dtype = int)
	
	countErrors = 0;

	
	for i in range(exampleSet) :
	   outcomeTest[i] = treeBranches[0,termTest[i,treeAttributions[0]]]
	   while outcomeTest[i] >= 0:
			outcomeTest[i] = treeBranches[outcomeTest[i],termTest[i,treeAttributions[outcomeTest[i]]]]
	   outcomeTest[i] = -1*(outcomeTest[i]+1)
	   if outcomeTest[i] != labelTest[i]:
			countErrors = countErrors + 1
        
	testError = float(countErrors)/float(exampleSet)
	print('after calculation, the Test Error should be:', testError)

question2()
question3a()
question3b()
