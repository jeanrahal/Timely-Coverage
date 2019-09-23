import numpy as np
import matplotlib.pyplot as plt
import random
import time 
import copy
import itertools 
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.optimize import brentq


    
def return_zj(selectRates,d):
    
    return lambda x: np.prod(1. - np.array(selectRates)*(x-d))


def comp1(selectRates,d,removedRate):
    
    return lambda x: -(x-d)*np.prod(1. - np.array(selectRates)*(x-d))

def comp2(selectRates,d,max_rate):
    
    return  -1./max_rate**2 * np.prod(1. - np.array(selectRates)*(1./max_rate))


def objective_function_MinAge(N, d, ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea):
    objFn = 0.
    tempObjFn = []
            
    for p in range(len(selectedPartitionsArea)):
        selectRates = []
        if selectedPartitionsArea[p] != 0:
            for s in range(len(allPossibleSets[p])):
                idx = np.where(setofSelectedSensors == allPossibleSets[p][s])[0][0]
                selectRates.append(ratePerSensor[idx]) #select the corresponding rates
            r_max = max(selectRates)
            result = quad(return_zj(selectRates,d), d, d+1./r_max)
            tempObjFn.append(d+result[0])
    
        else:
            tempObjFn.append(0.)
            
    objFn = np.sum(tempObjFn*selectedPartitionsArea)/np.sum(selectedPartitionsArea)
            
    return objFn


def gradient_obj_fn_MinAge(N, ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea,d):
    grad_MinAge = np.zeros(int(numSelectedSensors))
    

    #for ii in range(numSelectedSensors):
    for p in range(len(selectedPartitionsArea)):
        selectedRates = []
        if selectedPartitionsArea[p] != 0:
            for s in range(len(allPossibleSets[p])):
                idx = np.where(setofSelectedSensors == allPossibleSets[p][s])[0][0]
                selectedRates.append(ratePerSensor[idx])
            
            max_rate = max(selectedRates)
            
            for s in range(len(allPossibleSets[p])): 
                idx = np.where(setofSelectedSensors == allPossibleSets[p][s])[0][0]
                if ratePerSensor[idx] == max_rate:
                    temp1 = comp1(np.delete(selectedRates,s),d,selectedRates[s]) 
                    result = quad(temp1, d, d+1./max_rate)
                    temp2 = comp2(selectedRates,d,max_rate)
                    grad_MinAge[idx] = grad_MinAge[idx] + (result[0]+temp2)*selectedPartitionsArea[p]/np.sum(selectedPartitionsArea)
                else:
                    temp1 = comp1(np.delete(selectedRates,s),d,selectedRates[s])
                    result = quad(temp1, d, d+1./max_rate)
                    grad_MinAge[idx] = grad_MinAge[idx] + (result[0])*selectedPartitionsArea[p]/np.sum(selectedPartitionsArea)
                             
    return grad_MinAge
    

def frank_wolfe(N,ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea, t, capacity,d):
    # update x (your code here)
    # We implement a method we found in some book that describes FW update method
    grad = gradient_obj_fn_MinAge(N,ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea,d)
    idx = np.argmax(np.abs(grad))
    s = np.zeros(len(ratePerSensor))
    s[idx] = -capacity * np.sign(grad[idx])
    eta = 2./(t+2.)    
    ratePerSensor = ratePerSensor + eta*(s-ratePerSensor)
    ratePerSensor[ratePerSensor<=0]=1./d
    
    return ratePerSensor


def descent(N,update, d, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea, capacity, T=int(250)):
    ratePerSensor =  1.*np.ones(int(numSelectedSensors))
    obj_fn = []
    l1 = []
    
    for t in range(T):
        # update A (either subgradient or frank-wolfe)
        ratePerSensor = update(N,ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea, t, capacity, d)

        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            l1.append(np.sum(abs(ratePerSensor)))
            obj_fn.append(objective_function_MinAge(N,d, ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea))
            #assert not np.isnan(obj_fn[-1]) 
    
    return ratePerSensor, obj_fn, l1


def generatePixelsCenters(xPosCenterPixel1, yPosCenterPixel1, pixelLength, pixelWidth, numSquaresperLength, numSquaresperWidth):
    coordPixels = []
    
    #coordPixels.append([xPosCenterPixel1,yPosCenterPixel1])
    xCoord = xPosCenterPixel1
    yCoord = yPosCenterPixel1
    
    for i in range(numSquaresperLength):
        if i != 0:
            xCoord = xCoord + pixelLength
            
        yCoord = yPosCenterPixel1
        for j in range(numSquaresperWidth):
            if j != 0:
                yCoord = yCoord + pixelWidth
                
            newrow = np.array([xCoord,yCoord])
            if i == 0 | j == 0:
                coordPixels.append([xPosCenterPixel1,yPosCenterPixel1])
            else:
                coordPixels = np.vstack([coordPixels, newrow])
    
    return coordPixels


def pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors):
     isInCircle = 0
     
     if np.sqrt( (coordPixels[pixel][0]-coordSensors[sensor][0])**2 + (coordPixels[pixel][1]-coordSensors[sensor][1])**2 ) <= sensorRadius:
         isInCircle = 1
         
     return isInCircle


def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 

def findPartitionsAreas(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,N):
    tempPartitionsPixels = np.zeros(2**N-1)
    partitionsPixels = np.zeros(2**N-1)
    temp = np.zeros(2**N-1)
    temp1 = []
    allPossibleSets = []

    for ii in range(1,N+1):
        hello = findsubsets(np.arange(1,N+1,1),ii) 
        #hello1 = (np.asarray(hello))
        for jj in range(len(hello)):
            allPossibleSets.append(list(hello[jj]))
          
    for pixel in range(len(coordPixels)):
        for sensor in range(N):
            if pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors) == 1:
               tempPartitionsPixels[sensor] = tempPartitionsPixels[sensor] + 1 
        
        if np.sum(tempPartitionsPixels) > 1:
            idxOnes = np.nonzero(tempPartitionsPixels)
            for ii in range(idxOnes[0].size):
                temp1.append(idxOnes[0][ii]+1)        
            idxPartition = allPossibleSets.index(temp1)
            temp[idxPartition] = 1
        else:
            temp = tempPartitionsPixels
            
        partitionsPixels = partitionsPixels + temp
        
        tempPartitionsPixels = np.zeros(2**N-1)
        temp = np.zeros(2**N-1)
        temp1 = []
 
    return partitionsPixels*pixelLength*pixelWidth, allPossibleSets



################         OLD MODEL       #########################
def  baselineModel(ratePerSensor , d, partitionsArea , allPossibleSets, scalingFactor):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    AgePerPartition = []
    for ii in range(len(partitionsArea)):
        n = len(allPossibleSets[ii])
        tempAge = d + (1./(n+1.))*(1/ratePerSensor) 
        #tempAge = (n+2.)/(n+1.)*(1/ratePerSensor)
        AgePerPartition.append(tempAge)
    
    areaWeightedAge = np.sum(partitionsArea*AgePerPartition)/coverageArea
    
    return coverageArea, areaWeightedAge







####### END OF OLD MODEL   #######################
    

##### NEW MODEL (WITH SAMPLING) ################
def sampleIsinBounds(coordSample,coordBox):
    isInBox = 0
    
    if (coordSample[0][0] >= coordBox[0] and coordSample[0][0] <= coordBox[1] and coordSample[0][1] >= coordBox[2] and coordSample[0][1] <= coordBox[3]) :
        isInBox = 1
    
    return isInBox



def SamplesPerSensor(coordSensor,sensorRadius,coordBox,num_samples_per_sensor):
    coordSamplesPerSensor = []
    numIsInCircle = 0
                  
    
    for ii in range(num_samples_per_sensor):
        theta = 2*np.pi*np.random.rand(1,1)
        r = sensorRadius*np.sqrt(np.random.rand(1,1))
        x, y = r * np.cos(theta) + coordSensor[0]*np.ones([1,1]), r * np.sin(theta) + coordSensor[1]*np.ones([1,1])
        temp = np.concatenate((x,y),axis=1)
        if sampleIsinBounds(temp,coordBox) == 1:
           coordSamplesPerSensor.append([])
           coordSamplesPerSensor[numIsInCircle].append(temp)
           numIsInCircle = numIsInCircle + 1
           
    
    return numIsInCircle, coordSamplesPerSensor

def sampleisInCircle(sensorRadius,samplePoint,coordSamplesPerSensor,coordSensors):
     isInCircle = 0
     
     if np.sqrt( (coordSamplesPerSensor[0][0][0]-coordSensors[0])**2 + (coordSamplesPerSensor[0][0][1]-coordSensors[1])**2 ) <= sensorRadius:
         isInCircle = 1
         
     return isInCircle



def computeAgeandArea(N,sensorRadius,coordSamplesPerSensor,numIsInCircle,coordSensors):
    allPartitions = []
    samplesPerPartition = []
    
    for ii in range(len(coordSamplesPerSensor)):
        templistofPartitions = []
        for jj in range(N):
            if sampleisInCircle(sensorRadius,ii,coordSamplesPerSensor[ii],coordSensors[jj,:]):
                templistofPartitions.append(jj+1)
        
        #creating list of non-empty partitions   
        if templistofPartitions not in allPartitions:
            allPartitions.append(templistofPartitions)
    
            l = len(samplesPerPartition)
            samplesPerPartition.append(0) 
            samplesPerPartition[l] = samplesPerPartition[l] + 1
            
        else:
            # Find where does this partition fall and add a sample to the sample tube
           temp = allPartitions.index(templistofPartitions)
           samplesPerPartition[temp] = samplesPerPartition[temp] + 1
           
    return allPartitions, samplesPerPartition


def newbaselineModel(capacity, mu, N, d, coordSensors, sensorRadius, coordBox, num_samples_per_sensor, scalingFactor):
    ratePerSensor = capacity/(mu*d*N)
    areaWeightedAge = 0.
    coverageArea = 0.
    allPartitions = []
    samplesPerPartition = []
    appearanceOfaPartition = []
    agePerPartition = []
    percentageSamplesPerPartition = []
    
    for ii in range(N):
        #Step 1: for each sensor, sample "samples_per_sensor" points and check how many partitions do they cover
        numIsInCircle, coordSamplesPerSensor = SamplesPerSensor(coordSensors[ii,:],sensorRadius,coordBox,num_samples_per_sensor)
        
        #Step 2: check where does each sample fall
        tempallPartitions, tempsamplesPerPartition = computeAgeandArea(N,sensorRadius,coordSamplesPerSensor,numIsInCircle,coordSensors)
        
        for jj in range(len(tempallPartitions)):
            if tempallPartitions[jj] not in allPartitions:
                allPartitions.append(tempallPartitions[jj])
                samplesPerPartition.append(tempsamplesPerPartition[jj])
                #percentageSamplesPerPartition.append(tempsamplesPerPartition[jj]/numIsInCircle)
                appearanceOfaPartition.append(1)
            else:
                temp = allPartitions.index(tempallPartitions[jj])
                samplesPerPartition[temp] = samplesPerPartition[temp] + tempsamplesPerPartition[jj]
                #percentageSamplesPerPartition[temp] = percentageSamplesPerPartition[temp] + tempsamplesPerPartition[jj]/numIsInCircle
                appearanceOfaPartition[temp] = appearanceOfaPartition[temp] + 1
                
        
    percentageSamplesPerPartition = np.array(samplesPerPartition)/np.array(appearanceOfaPartition)/num_samples_per_sensor
    areaPerPartition = percentageSamplesPerPartition*np.pi*sensorRadius**2*scalingFactor**2
    coverageArea = np.sum(areaPerPartition)
    
    for ii in range(len(allPartitions)):
        n = len(allPartitions[ii])
        tempAge = d + (1./(n+1.))*(1/ratePerSensor)
        agePerPartition.append(tempAge)
        
    areaWeightedAge = np.sum(areaPerPartition*agePerPartition)/coverageArea        

    
    return coverageArea, areaWeightedAge


def newSensSelecModel(N, d, capacity, mu, coordSensors, sensorRadius, coordBox, num_samples_per_sensor, scalingFactor, lam, areaR, thresh = 2.):
    areaWeightedAge = 0.  
    coverageArea = 0.
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
    allPartitions = []
    samplesPerPartition = []
    appearanceOfaPartition = []
    agePerPartition = []
    percentageSamplesPerPartition = []  
    
    
    k = 4.
    #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = (k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+1./2.*numSelectedSensors)
    
    new_max = 0.

    for ii in range(int(numSelectedSensors)):
        new_max = 0.
        for jj in range(N):
            if jj+1 not in setofSelectedSensors:
                #Step 1: for each sensor, sample "samples_per_sensor" points and check how many partitions do they cover
                numIsInCircle, coordSamplesPerSensor = SamplesPerSensor(coordSensors[jj,:],sensorRadius,coordBox,num_samples_per_sensor)
                
                #Step 2: compute the 'b' function
                #b_new, tempcoverageArea, tempareaWeightedAge = new_compute_b(N, d, mu, coordSensors, setofSelectedSensors, setofSensors, ratePerSensor, jj+1, sensorRadius, num_samples_per_sensor, numIsInCircle, coordSamplesPerSensor, oldcoverageArea, oldareaWeightedAge, scalingFactor, areaR, lam)
                
                # Key step: Compute delta_b = b_new - b_old
                delta_b = compute_delta_b(N, d, mu, coordSensors, setofSelectedSensors, setofSensors, ratePerSensor, jj+1, sensorRadius, num_samples_per_sensor, numIsInCircle, coordSamplesPerSensor, scalingFactor, areaR, lam)

                if delta_b >= new_max:
                    new_max = delta_b
                    selectedSensor = jj+1
        
        setofSelectedSensors.append(selectedSensor)
    
    for ii in range(len(setofSelectedSensors)):
        #Step 1: for each sensor, sample "samples_per_sensor" points and check how many partitions do they cover
        numIsInCircle, coordSamplesPerSensor = SamplesPerSensor(coordSensors[setofSelectedSensors[ii]-1,:],sensorRadius,coordBox,num_samples_per_sensor)
        
        #Step 2: check where does each sample fall
        tempallPartitions, tempsamplesPerPartition = computeAgeandArea(N,sensorRadius,coordSamplesPerSensor,numIsInCircle,coordSensors)
        
        for jj in range(len(tempallPartitions)):
            if tempallPartitions[jj] not in allPartitions:
                allPartitions.append(tempallPartitions[jj])
                samplesPerPartition.append(tempsamplesPerPartition[jj])
                #percentageSamplesPerPartition.append(tempsamplesPerPartition[jj]/numIsInCircle)
                appearanceOfaPartition.append(1)
            else:
                temp = allPartitions.index(tempallPartitions[jj])
                samplesPerPartition[temp] = samplesPerPartition[temp] + tempsamplesPerPartition[jj]
                #percentageSamplesPerPartition[temp] = percentageSamplesPerPartition[temp] + tempsamplesPerPartition[jj]/numIsInCircle
                appearanceOfaPartition[temp] = appearanceOfaPartition[temp] + 1
                
        
    percentageSamplesPerPartition = np.array(samplesPerPartition)/np.array(appearanceOfaPartition)/num_samples_per_sensor
    areaPerPartition = percentageSamplesPerPartition*np.pi*sensorRadius**2*scalingFactor**2
    coverageArea = np.sum(areaPerPartition)
    
    for ii in range(len(allPartitions)):
        n = len(allPartitions[ii])
        tempAge = d + (1./(n+1.))*(1/ratePerSensor)
        agePerPartition.append(tempAge)
        
    areaWeightedAge = np.sum(areaPerPartition*agePerPartition)/coverageArea


    
    return coverageArea , areaWeightedAge/(coverageArea) , setofSelectedSensors

def compute_delta_b(N, d, mu, coordSensors, setofSelectedSensors, setofSensors, ratePerSensor, currSensor, sensorRadius, num_samples_per_sensor, numIsInCircle, coordSamplesPerSensor, scalingFactor, areaR, lam):
    delta_b = 0.
    allPartitions = []
    samplesPerPartition = []
    appearanceOfaPartition = []
    coordSelectedSensors = []
        
    if not setofSelectedSensors:
        currSensors = currSensor
        coordSelectedSensors.append([])
        coordSelectedSensors[0].append(coordSensors[currSensors-1,:])
        
        #coordSelectedSensors = np.array(coordSelectedSensors)        
        tempallPartitions, tempsamplesPerPartition = newcomputeAgeandArea(currSensors,coordSelectedSensors,sensorRadius,coordSamplesPerSensor,numIsInCircle)
        
        for jj in range(len(tempallPartitions)):
            if tempallPartitions[jj] not in allPartitions:
                allPartitions.append(tempallPartitions[jj])
                samplesPerPartition.append(tempsamplesPerPartition[jj])
                #percentageSamplesPerPartition.append(tempsamplesPerPartition[jj]/numIsInCircle)
                appearanceOfaPartition.append(1)
            else:
                temp = allPartitions.index(tempallPartitions[jj])
                samplesPerPartition[temp] = samplesPerPartition[temp] + tempsamplesPerPartition[jj]
                #percentageSamplesPerPartition[temp] = percentageSamplesPerPartition[temp] + tempsamplesPerPartition[jj]/numIsInCircle
                appearanceOfaPartition[temp] = appearanceOfaPartition[temp] + 1
        
        percentageSamplesPerPartition = np.array(samplesPerPartition)/np.array(appearanceOfaPartition)/num_samples_per_sensor
        areaPerPartition = percentageSamplesPerPartition*np.pi*sensorRadius**2*scalingFactor**2
    
            
    else:
        currSensors = copy.copy(setofSelectedSensors)
        currSensors.append(currSensor)
        currSensors = np.sort(currSensors)
        #Step 2: check where does each sample fall
        for ii in range(len(currSensors)):
            coordSelectedSensors.append([])
            coordSelectedSensors[ii].append(coordSensors[currSensors[ii]-1,:])
        
        #coordSelectedSensors = np.array(coordSelectedSensors)        
        tempallPartitions, tempsamplesPerPartition = newcomputeAgeandArea(currSensors,coordSelectedSensors,sensorRadius,coordSamplesPerSensor,numIsInCircle)
        
        for jj in range(len(tempallPartitions)):
            if tempallPartitions[jj] not in allPartitions:
                allPartitions.append(tempallPartitions[jj])
                samplesPerPartition.append(tempsamplesPerPartition[jj])
                #percentageSamplesPerPartition.append(tempsamplesPerPartition[jj]/numIsInCircle)
                appearanceOfaPartition.append(1)
            else:
                temp = allPartitions.index(tempallPartitions[jj])
                samplesPerPartition[temp] = samplesPerPartition[temp] + tempsamplesPerPartition[jj]
                #percentageSamplesPerPartition[temp] = percentageSamplesPerPartition[temp] + tempsamplesPerPartition[jj]/numIsInCircle
                appearanceOfaPartition[temp] = appearanceOfaPartition[temp] + 1
                
        
        percentageSamplesPerPartition = np.array(samplesPerPartition)/np.array(appearanceOfaPartition)/num_samples_per_sensor
        areaPerPartition = percentageSamplesPerPartition*np.pi*sensorRadius**2*scalingFactor**2
    
    deltaCoverageArea = 0.    
    deltaAreaWeightedAge = 0.    
    
    for ii in range(len(allPartitions)):
        if len(allPartitions[ii]) == 1:
            deltaCoverageArea = deltaCoverageArea + lam*areaPerPartition[ii]
            deltaAreaWeightedAge = deltaAreaWeightedAge-areaPerPartition[ii]*(d+1./2.*1./ratePerSensor)
        else:
            l = len(allPartitions[ii])
            deltaAreaWeightedAge = deltaAreaWeightedAge + areaPerPartition[ii]*(-1./((l+1)*(l)))*1./ratePerSensor
    
    delta_b = deltaCoverageArea + deltaAreaWeightedAge 
      
    return delta_b






def newcomputeAgeandArea(currSensors,coordSelectedSensors,sensorRadius,coordSamplesPerSensor,numIsInCircle):
    allPartitions = []
    samplesPerPartition = []
    
    if np.size(currSensors) > 1:
        currSensors = np.sort(currSensors)
    #coordSelectedSensors = np.array(coordSelectedSensors)
    for ii in range(len(coordSamplesPerSensor)):
        templistofPartitions = []
        for jj in range(len(coordSelectedSensors)):
            if sampleisInCircle(sensorRadius,ii,coordSamplesPerSensor[ii],coordSelectedSensors[jj][0]):
                if np.size(currSensors) == 1:
                    templistofPartitions.append(currSensors)
                else:
                    templistofPartitions.append(currSensors[jj])
        
        #creating list of non-empty partitions   
        if templistofPartitions not in allPartitions:
            allPartitions.append(templistofPartitions)
    
            l = len(samplesPerPartition)
            samplesPerPartition.append(0) 
            samplesPerPartition[l] = samplesPerPartition[l] + 1
            
        else:
            # Find where does this partition fall and add a sample to the sample tube
           temp = allPartitions.index(templistofPartitions)
           samplesPerPartition[temp] = samplesPerPartition[temp] + 1
           
    return allPartitions, samplesPerPartition


######################################################


def compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, currSensor, allPossibleSets, areaR, lam):
    b = 0.
    AgePerPartition = []
    coveredArea = []
    tempP = np.zeros(2**N-1)
    newPartitionArea = np.zeros(2**N-1)
    if not setofSelectedSensors:
        currSensors = np.array(currSensor)
        for ii in range(len(partitionsArea)):
            if currSensors in allPossibleSets[ii]:    
                tempP[ii] = tempP[ii] + 1 #check how many sensors cover a particular partition
                newPartitionArea[ii] = partitionsArea[ii] 
    else:
        currSensors = copy.copy(setofSelectedSensors)
        currSensors.append(currSensor)
        for s in range(len(currSensors)):
            for ii in range(len(partitionsArea)):
                if currSensors[s] in allPossibleSets[ii]:    
                    tempP[ii] = tempP[ii] + 1 #check how many sensors cover a particular partition
                    newPartitionArea[ii] = partitionsArea[ii]                    
                
                
    for ii in range(len(partitionsArea)):
        n = tempP[ii]
        if n!=0:
            tempAge = d + (1./(n+1.))*(1./ratePerSensor) 
            #tempAge = (n+2.)/(n+1.)*(1./ratePerSensor)
            if np.isnan(tempAge):
               AgePerPartition.append(0.)
            else:
               AgePerPartition.append(tempAge)
            coveredArea.append(newPartitionArea[ii])
        else:
            AgePerPartition.append(0.)
            coveredArea.append(0.)
            
    totalCoveredArea = np.sum(coveredArea)        
    areaWeightedAge = np.sum(np.array(coveredArea)*np.array(AgePerPartition))        
    selectedPartitionsArea = copy.copy(newPartitionArea)
    
    a = areaWeightedAge + lam*(areaR-totalCoveredArea)     
    a_empty = lam*areaR
    b = a_empty-a
    
    return b, totalCoveredArea, areaWeightedAge, selectedPartitionsArea


def SensSelecModel(N, d, capacity, mu, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, areaR, thresh = 2.):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
    
    k = 4.
    #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = (k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+1./2.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        for jj in range(N):
            if jj+1 not in setofSelectedSensors:
                b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, jj+1, allPossibleSets, areaR, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = jj+1
                    coverageArea = tempcoverageArea
                    areaWeightedAge = tempareaWeightedAge
        setofSelectedSensors.append(selectedSensor)
                
    #setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return coverageArea , areaWeightedAge/(coverageArea) , setofSelectedSensors


def AgeMinModel(N, d, mu, capacity , partitionsArea , allPossibleSets, rectangleLength , rectangleWidth , sensorRadius, scalingFactor , T, lam, thresh = 2.):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
    
    k = 4.
    #np.ceil((rectangleLength/sensorRadius)*1.)
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+1./2.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        for jj in range(N):
            if jj+1 not in setofSelectedSensors:
                b_new, tempcoverageArea , tempareaWeightedAge,selectedPartitionsArea = compute_b(N, d ,mu, partitionsArea, setofSelectedSensors, setofSensors ,ratePerSensor, jj+1, allPossibleSets, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = jj+1
                    coverageArea = tempcoverageArea
                    areaWeightedAge = tempareaWeightedAge
        setofSelectedSensors.append(selectedSensor)
                
    setofSelectedSensors = np.sort(setofSelectedSensors)
    
    
    
    newallPossibleSets = []
    
    for ii in range(1,int(numSelectedSensors)+1):
        hello = findsubsets(setofSelectedSensors,ii) 
        #hello1 = (np.asarray(hello))
        for jj in range(len(hello)):
            newallPossibleSets.append(list(hello[jj]))
    
    newselectedPartitionsArea = np.zeros(2**(numSelectedSensors)-1)
        
    for ii in range(len(allPossibleSets)):
        temp = []
        for jj in range(len(allPossibleSets[ii])):
            if allPossibleSets[ii][jj] in setofSelectedSensors:
                temp.append(allPossibleSets[ii][jj])
        if temp:
            #temp = np.sort(temp)
            idx = newallPossibleSets.index(temp)
            newselectedPartitionsArea[idx] = newselectedPartitionsArea[idx] + partitionsArea[ii]    
            
    # Compute new rate allocation and new ageWeightedArea
    rate_fw_agemin,  obj_fn, l1_fw_agemin = descent(N,frank_wolfe, d, numSelectedSensors, setofSelectedSensors, newallPossibleSets, np.array(newselectedPartitionsArea), capacity/(mu*d), T=T)
    
    return coverageArea, obj_fn[-1], setofSelectedSensors


##################################################################


def main(T=int(5e2)): 
    scalingFactor = 50
    N = np.arange(2,9,1) # number of sensors
    num_samples_per_sensor = 1000
    
    lam = 1.
    sensorRadius = np.array(100/scalingFactor)#coverage radius per sensor
    #sensorRadius = []
    #sensorRadius = np.array([1.,1.,1.,1.,1.,2.,2.,2.,2.,2.])    
    capacity = 1.
    d = 0.5e-3 #transmission delay
    mu = 1. #packet size
    
    rectangleLength = 500/scalingFactor
    rectangleWidth = 10/scalingFactor
    areaR = rectangleLength*rectangleWidth*scalingFactor**2

    numSquaresperLength = int(rectangleLength*10)
    numSquaresperWidth = int(rectangleWidth*10)
    
    pixelLength = rectangleLength/numSquaresperLength
    pixelWidth = rectangleWidth/numSquaresperWidth
    
    xPosCenterPixel1 = pixelLength/2
    yPosCenterPixel1 = pixelWidth/2
    
    coordPixels = generatePixelsCenters(xPosCenterPixel1, yPosCenterPixel1, pixelLength, pixelWidth, numSquaresperLength, numSquaresperWidth)

    
    # the coordinates of the box are: x_min, x_max, y_min, y_max
    coordBox = np.array([0.,rectangleLength,0.,rectangleWidth])

    coverageAreaBaseline = []
    areaWeightedAgeBaseline = []
    totalTimeBaseline = []
    
    coverageAreaSensSelec = []
    areaWeightedAgeSensSelec = []
    selectedSensorsSensSelec = []
    totalTimeSensSelec = []
    
    coverageAreaAgeMin = []
    areaWeightedAgeAgeMin =[]
    selectedSensorsAgeMin =[]
 
    
    newcoverageAreaBaseline = []
    newareaWeightedAgeBaseline = []
    newtotalTimeBaseline = []
    
    newcoverageAreaSensSelec = []
    newareaWeightedAgeSensSelec = []
    newselectedSensorsSensSelec = []
    newtotalTimeSensSelec = []
    
    newcoverageAreaAgeMin = []
    newareaWeightedAgeAgeMin =[]
    newselectedSensorsAgeMin =[]    

    numIter = 20        

    for ii in range(len(N)):
         temp1coverageAreaBaseline = []
         temp1areaWeightedAgeBaseline = []
         temp1TotalTimeBaseline = []
         
         temp1coverageAreaSensSelec = []
         temp1areaWeightedAgeSensSelec = []
         temp1selectedSensorsSensSelec = []
         temp1TotalTimeSensSelec = []
         
         temp1coverageAreaAgeMin = []
         temp1areaWeightedAgeAgeMin =[]
         temp1selectedSensorsAgeMin =[]
         
         for jj in range(numIter):
             xcoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleLength-0) 
             ycoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleWidth-0)
             coordSensors = np.concatenate((xcoordSensors,ycoordSensors),axis=1)

             startTimeFindPartitions = time.time()
             partitionsArea , allPossibleSets = findPartitionsAreas(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,N[ii])
             endTimeFindPartitions = time.time()
             
             startTimeBaseline = time.time()
             tempcoverageAreaBaseline , tempareaWeightedAgeBaseline = baselineModel(capacity/(N[ii]*mu*d), d, partitionsArea*scalingFactor**2 , allPossibleSets, scalingFactor)
             endTimeBaseline  = time.time()
            
             startTimeSensSelec = time.time()
             tempcoverageAreaSensSelec , tempareaWeightedAgeSensSelec , tempselectedSensorsSensSelec = SensSelecModel(N[ii], d, capacity , mu, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, lam, areaR, thresh = 2.)
             endTimeSensSelec = time.time()
             
#             tempcoverageAreaAgeMin , tempareaWeightedAgeAgeMin , tempselectedSensorsAgeMin = AgeMinModel(N[ii], d, mu, capacity , partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor , rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, T, lam ,thresh = 2.)
   
             temp1coverageAreaBaseline.append(tempcoverageAreaBaseline)
             temp1areaWeightedAgeBaseline.append(tempareaWeightedAgeBaseline)
             temp1TotalTimeBaseline.append((endTimeFindPartitions-startTimeFindPartitions)+(endTimeBaseline-startTimeBaseline))
             
             temp1coverageAreaSensSelec.append(tempcoverageAreaSensSelec)
             temp1areaWeightedAgeSensSelec.append(tempareaWeightedAgeSensSelec)
             temp1selectedSensorsSensSelec.append(len(tempselectedSensorsSensSelec))
             temp1TotalTimeSensSelec.append((endTimeFindPartitions-startTimeFindPartitions)+(endTimeSensSelec-startTimeSensSelec))
        
     
#             temp1coverageAreaAgeMin.append(tempcoverageAreaAgeMin)
#             temp1areaWeightedAgeAgeMin.append(tempareaWeightedAgeAgeMin)
#             temp1selectedSensorsAgeMin.append(len(tempselectedSensorsAgeMin))
     
             
         coverageAreaBaseline.append(np.sum(temp1coverageAreaBaseline)/numIter/areaR)
         areaWeightedAgeBaseline.append(np.sum(temp1areaWeightedAgeBaseline)/numIter*1000.)
         totalTimeBaseline.append(np.sum(temp1TotalTimeBaseline)/numIter) 
         
         coverageAreaSensSelec.append(np.sum(temp1coverageAreaSensSelec)/numIter/areaR)
         areaWeightedAgeSensSelec.append(np.sum(temp1areaWeightedAgeSensSelec)/numIter*1000.)
         selectedSensorsSensSelec.append(np.sum(temp1selectedSensorsSensSelec)/numIter)
         totalTimeSensSelec.append(np.sum(temp1TotalTimeSensSelec)/numIter) 

     
#         coverageAreaAgeMin.append(np.sum(temp1coverageAreaAgeMin)/numIter/areaR)
#         areaWeightedAgeAgeMin.append(np.sum(temp1areaWeightedAgeAgeMin)/numIter*1000.)
#         selectedSensorsAgeMin.append(np.sum(temp1selectedSensorsAgeMin)/numIter)        


    for ii in range(len(N)):
         temp1coverageAreaBaseline = []
         temp1areaWeightedAgeBaseline = []
         temp1TotalTimeBaseline = []
         
         temp1coverageAreaSensSelec = []
         temp1areaWeightedAgeSensSelec = []
         temp1selectedSensorsSensSelec = []
         temp1TotalTimeSensSel = []
         
         temp1coverageAreaAgeMin = []
         temp1areaWeightedAgeAgeMin =[]
         temp1selectedSensorsAgeMin =[]
         
         
         for jj in range(numIter):
             xcoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleLength-0) 
             ycoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleWidth-0)
             coordSensors = np.concatenate((xcoordSensors,ycoordSensors),axis=1)
             
             startTimeBaseline = time.time()
             tempcoverageAreaBaseline , tempareaWeightedAgeBaseline = newbaselineModel(capacity, mu, N[ii], d, coordSensors, sensorRadius, coordBox, num_samples_per_sensor, scalingFactor)
             endTimeBaseline = time.time()
             
             startTimeSensSelec = time.time()
             tempcoverageAreaSensSelec, tempareaWeightedAgeSensSelec, tempselectedSensorsSensSelec = newSensSelecModel(N[ii], d, capacity , mu, coordSensors, sensorRadius, coordBox, num_samples_per_sensor, scalingFactor, lam, areaR, thresh = 2.) 
             endTimeSensSelec = time.time()
             
             #tempcoverageAreaAgeMin , tempareaWeightedAgeAgeMin , tempselectedSensorsAgeMin = AgeMinModel(N[ii], d, mu, capacity , partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor , rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, T, lam ,thresh = 2.)
             
             
             temp1coverageAreaBaseline.append(tempcoverageAreaBaseline)
             temp1areaWeightedAgeBaseline.append(tempareaWeightedAgeBaseline)
             temp1TotalTimeBaseline.append(endTimeBaseline-startTimeBaseline)

             
             temp1coverageAreaSensSelec.append(tempcoverageAreaSensSelec)
             temp1areaWeightedAgeSensSelec.append(tempareaWeightedAgeSensSelec)
             temp1selectedSensorsSensSelec.append(len(tempselectedSensorsSensSelec))
             temp1TotalTimeSensSelec.append(endTimeSensSelec-startTimeSensSelec)
        
     
#             temp1coverageAreaAgeMin.append(tempcoverageAreaAgeMin)
#             temp1areaWeightedAgeAgeMin.append(tempareaWeightedAgeAgeMin)
#             temp1selectedSensorsAgeMin.append(len(tempselectedSensorsAgeMin))
     
             
         newcoverageAreaBaseline.append(np.sum(temp1coverageAreaBaseline)/numIter/areaR)
         newareaWeightedAgeBaseline.append(np.sum(temp1areaWeightedAgeBaseline)/numIter*1000.)
         newtotalTimeBaseline.append(np.sum(temp1TotalTimeBaseline)/numIter)  
         
         newcoverageAreaSensSelec.append(np.sum(temp1coverageAreaSensSelec)/numIter/areaR)
         newareaWeightedAgeSensSelec.append(np.sum(temp1areaWeightedAgeSensSelec)/numIter*1000.)
         newselectedSensorsSensSelec.append(np.sum(temp1selectedSensorsSensSelec)/numIter)
         newtotalTimeSensSelec.append(np.sum(temp1TotalTimeSensSelec)/numIter) 
     
         
#     
#         newcoverageAreaAgeMin.append(np.sum(temp1coverageAreaAgeMin)/numIter/areaR)
#         newareaWeightedAgeAgeMin.append(np.sum(temp1areaWeightedAgeAgeMin)/numIter*1000.)
#         newselectedSensorsAgeMin.append(np.sum(temp1selectedSensorsAgeMin)/numIter)        
     
    plt.clf()
    plt.plot(N , areaWeightedAgeBaseline, 'b' , label='Baseline')
    plt.plot(N , newareaWeightedAgeBaseline, 'b--', label='Sampling Baseline')
    plt.plot(N , areaWeightedAgeSensSelec, 'r',label='Sensor Selection')
    plt.plot(N , newareaWeightedAgeSensSelec, 'r--',label='Sampling Sensor Selection')
    #plt.plot(N , areaWeightedAgeAgeMin, label='Age Minimization')
     #plt.title('Area weighted age as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
      #plt.yscale('log')
    plt.xlabel('Number of available sensors N', fontsize=12)
    plt.ylabel('Normalized average weighted age [msec]', fontsize=10)
    plt.savefig('SampledAge_N=1_10.eps')
    plt.savefig('SampledAge_N=1_10.pdf')
      
    plt.clf()
    plt.plot(N , coverageAreaBaseline, 'b',  label='Baseline')
    plt.plot(N , newcoverageAreaBaseline, 'b--', label='Sampling Baseline')
    plt.plot(N , coverageAreaSensSelec, 'r',label='Sensor Selection')
    plt.plot(N , newcoverageAreaSensSelec, 'r--',label='Sampling Sensor Selection')
    #plt.plot(N , coverageAreaAgeMin, label='Age Minimization')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xlabel('Number of available sensors N', fontsize=12)
    plt.ylabel('Coverage Area [%]', fontsize=10)
    plt.savefig('SampledCov_N=1_10.eps')
    plt.savefig('SampledCov_N=1_10.pdf')
    
    plt.clf()
    plt.plot(N , totalTimeBaseline, 'b',  label='Baseline')
    plt.plot(N , newtotalTimeBaseline, 'b--', label='Sampling Baseline')
    plt.plot(N , totalTimeSensSelec, 'r',label='Sensor Selection')
    plt.plot(N , newtotalTimeSensSelec, 'r--',label='Sampling Sensor Selection')
    #plt.plot(N , coverageAreaAgeMin, label='Age Minimization')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of available sensors N', fontsize=12)
    plt.ylabel('Algorithm running time [sec]', fontsize=10)
    plt.savefig('SampledAlgoRunTime_N=1_10.eps')
    plt.savefig('SampledAlgoRunTime_N=1_10.pdf')    

    
if __name__ == "__main__":
    main()