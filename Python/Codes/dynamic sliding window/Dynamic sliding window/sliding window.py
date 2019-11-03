#### during each window, 1 sensor is added, 1 sensor leaves  ######

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
import itertools 
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.optimize import brentq
import os
from datetime import date
from numba import jit, cuda 
from tqdm import tqdm
    
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


def compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, currSensor, allPossibleSets, lam):
    b = 0.
    AgePerPartition = []
    coveredArea = []
    tempP = np.zeros(2**N-1)
    newPartitionArea = np.zeros(2**N-1)
    if not list(setofSelectedSensors):
        currSensors = np.array(currSensor)
        for ii in range(len(partitionsArea)):
            if currSensors in allPossibleSets[ii]:    
                tempP[ii] = tempP[ii] + 1 #check how many sensors cover a particular partition
                newPartitionArea[ii] = partitionsArea[ii] 
    else:
        currSensors = copy.copy(list(setofSelectedSensors))
        if not (not currSensor):
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
    
    a = areaWeightedAge + lam*(np.sum(partitionsArea)-totalCoveredArea)     
    a_empty = lam*np.sum(partitionsArea)
    b = a_empty-a
    
    return b, totalCoveredArea, areaWeightedAge, selectedPartitionsArea

#@jit(target ="cuda")  
def GreedySensSelecModel(N, k, d, capacity, mu, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, coordSensors, lam):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    numSelectedSensors = N
    setofSelectedSensors = []
    coordSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
        #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = (k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+2./3.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        for jj in range(N):
            if jj+1 not in setofSelectedSensors:
                b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, jj+1, allPossibleSets, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = jj+1
                    coverageArea = tempcoverageArea
                    areaWeightedAge = tempareaWeightedAge
                    max_b_new = b_new
        setofSelectedSensors.append(selectedSensor)
    
    setofSelectedSensors = np.sort(setofSelectedSensors)
    for ii in range(int(numSelectedSensors)):
        coordSelectedSensors.append(coordSensors[setofSelectedSensors[ii]-1,:])
    
    return coverageArea , areaWeightedAge/(coverageArea) , max_b_new, setofSelectedSensors, coordSelectedSensors



def ModifiedStreamSensSelecModel(N, k, d, capacity, mu, coordprevSelectedSensors, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, coordSensors, lam):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    coordSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
        #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    
    numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+2./3.*numSelectedSensors)
    
    if len(coordprevSelectedSensors) < k:
        # check which of the non-selected sensors has the biggest value function
        coordSelectedSensors = coordprevSelectedSensors
        #coordSelectedSensors.append(coordSensors[-1])
        #setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,-1)
        #new_max, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, len(coordSensors), allPossibleSets, lam)
        coordSelectedSensors.append(coordSensors[-1]) 
    else:
        new_max = 0.
        for ii in range(int(numSelectedSensors)):
            setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,ii) #remove one sensor in each iteration, sensor 'ii'
            b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, len(coordSensors), allPossibleSets, lam) #we add the last sensor cuz he's the freshest
            if  b_new >= new_max:
                new_max = b_new
                removedSensorCoord = []
                removedSensorCoord.append(coordprevSelectedSensors[ii:][0])
                coverageArea = tempcoverageArea
                areaWeightedAge = tempareaWeightedAge
        
        
        for ii in range(len(coordSensors)):
            if (coordSensors[ii,:]) in np.array(coordprevSelectedSensors):
                if np.sum((coordSensors[ii,:]) != removedSensorCoord) != 0:
                    coordSelectedSensors.append(coordSensors[ii,:]) 
         # Append the new sensor at the end:
        coordSelectedSensors.append(coordSensors[-1])           
        
    selectedSensors = findsetofSelecSensors(coordSelectedSensors,coordSensors,-1)
    new_max, coverageArea, areaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, selectedSensors, setofSensors, ratePerSensor, [], allPossibleSets, lam)

    
    return coverageArea , areaWeightedAge/(coverageArea) , new_max, selectedSensors, coordSelectedSensors #new_max is b_new



def StreamSensSelecModel(N, k, d, capacity, mu, coordprevSelectedSensors, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, coordSensors, lam):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    coordSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
        #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    
    numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+2./3.*numSelectedSensors)
    
    if len(coordprevSelectedSensors) < k:
        # check which of the non-selected sensors has the biggest value function
        coordSelectedSensors = coordprevSelectedSensors
        #coordSelectedSensors.append(coordSensors[-1])
        setofSelectedSensors = findsetofSelecSensors(coordSelectedSensors,coordSensors,-1)

        new_max = 0.
        for ii in range(len(setofSensors)):
            if setofSensors[ii] not in setofSelectedSensors:
                b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, setofSensors[ii], allPossibleSets, lam) #we add the last sensor cuz he's the freshest
                if  b_new >= new_max:
                    new_max = b_new
                    addedSensorCoord = []
                    addedSensorCoord.append(coordSensors[ii:][0])
       
        coordSelectedSensors.append(addedSensorCoord[0])             
                    
    else:
        setofprevSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,-1)
        if len(setofprevSelectedSensors)<k:
            print(setofprevSelectedSensors)
            time.sleep(100)
        
        new_max = 0.
        for ii in range(int(numSelectedSensors)):
            removedSensor = setofprevSelectedSensors[ii]
            setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,ii) #remove one sensor in each iteration, sensor 'ii'
            for jj in range(len(setofSensors)):
                if (setofSensors[jj] not in setofSelectedSensors) :#& (setofSensors[jj] != removedSensor):
                    b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, setofSensors[jj], allPossibleSets, lam) #we add the last sensor cuz he's the freshest
                    if  b_new >= new_max:
                        new_max = b_new
                        removedSensorCoord = []
                        removedSensorCoord.append(coordprevSelectedSensors[ii:][0])
                        addedSensorCoord = []
                        addedSensorCoord.append(coordSensors[jj:][0])

        
        
        for ii in range(len(coordSensors)):
            if (coordSensors[ii,:]) in np.array(coordprevSelectedSensors):
                if np.sum((coordSensors[ii,:]) != removedSensorCoord) != 0:
                    coordSelectedSensors.append(coordSensors[ii,:]) 
                    
            elif (coordSensors[ii,:]) in addedSensorCoord[0]:
                coordSelectedSensors.append(coordSensors[ii,:])
                          
        
    selectedSensors = findsetofSelecSensors(coordSelectedSensors,coordSensors,-1)
    new_max, coverageArea, areaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, selectedSensors, setofSensors, ratePerSensor, [], allPossibleSets, lam)

    
    return coverageArea , areaWeightedAge/(coverageArea) , new_max, selectedSensors, coordSelectedSensors #new_max is b_new


def ModifiedThreshSensSelecModel(N, k, d, capacity, mu, coordprevSelectedSensors, partitionsArea, allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, coordSensors, lam, counter, thresh):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    coordSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
    startTime = 0.
    endTime = 0.
        #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    
    numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+2./3.*numSelectedSensors)
    
    if len(coordprevSelectedSensors) < k:
        # check which of the non-selected sensors has the biggest value function
        coordSelectedSensors = coordprevSelectedSensors
        #coordSelectedSensors.append(coordSensors[-1])
        #setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,-1)
        #new_max, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, len(coordSensors), allPossibleSets, lam)
        coordSelectedSensors.append(coordSensors[-1]) 
    else:
        b_of_previous_sensors = []
        startTime = time.time()
        setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,-1)
        # Compute b of the previous sensors        
        for ii in range(len(setofSelectedSensors)):
            #a = []
            old_b, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, [], setofSensors, ratePerSensor, setofSelectedSensors[ii], allPossibleSets, lam)
            #temp_b, temp1, temp2, s1 = compute_b(N, d, mu, partitionsArea, (setofSelectedSensors[ii]), setofSensors, ratePerSensor, [], allPossibleSets, lam)
            b_of_previous_sensors.append(old_b)
        endTime = time.time()    
        # Compute b of the newly added sensor
        b_new_sensor, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, [], setofSensors, ratePerSensor, len(coordSensors), allPossibleSets, lam)
        if b_new_sensor > thresh*min(b_of_previous_sensors):  
            idxMin = b_of_previous_sensors.index(min(b_of_previous_sensors))
            b_of_previous_sensors.remove(min(b_of_previous_sensors))
            #list(setofSelectedSensors).remove(setofSelectedSensors[idxMin])

            coordSelectedSensors = coordprevSelectedSensors
            coordSelectedSensors = list(np.delete(np.array(coordSelectedSensors),idxMin,0))            

            #setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,-1)
            #new_max, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, len(coordSensors), allPossibleSets, lam)

            coordSelectedSensors.append(coordSensors[-1])
            
            counter = counter + 1

        else:
            #no change in previous sensors
            coordSelectedSensors = coordprevSelectedSensors
            #coordSelectedSensors.append(coordSensors[-1])
            #setofSelectedSensors = findsetofSelecSensors(coordprevSelectedSensors,coordSensors,-1)
            #new_max, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, [], allPossibleSets, lam)
                        
                            
    selectedSensors = findsetofSelecSensors(coordSelectedSensors,coordSensors,-1)
    new_max, coverageArea, areaWeightedAge, selectedPartitionsArea = compute_b(N, d, mu, partitionsArea, selectedSensors, setofSensors, ratePerSensor, [], allPossibleSets, lam)
        
    return coverageArea , areaWeightedAge/(coverageArea) , new_max, selectedSensors, coordSelectedSensors, (endTime-startTime), counter #new_max is b_new



def findsetofSelecSensors(coordprevSelectedSensors,coordSensors,removedSensorIdx):
    setofSelectedSensors = []
    
    for ii in range(len(coordSensors)):
        if coordSensors[ii,:] in np.array(coordprevSelectedSensors):
            if removedSensorIdx < 0:
                setofSelectedSensors.append(ii+1) 
            elif np.sum(coordSensors[ii,:] != np.array(coordprevSelectedSensors)[removedSensorIdx,:])>0:
                setofSelectedSensors.append(ii+1) 
    
    setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return setofSelectedSensors

def check_k(selectedSensors,CoordSelectedSensors):
    coords = CoordSelectedSensors
    
    if selectedSensors[0] == 1:
        coords = CoordSelectedSensors[:][1:]
    
    return coords

def check_k1(selectedSensors,CoordSelectedSensors,coordSensors):
    coords = CoordSelectedSensors
    
    if selectedSensors[0] == 1:
        coords = []
        for ii in range(len(CoordSelectedSensors)):
            if np.sum(CoordSelectedSensors[ii:][0] != coordSensors[0,:])>0: 
                coords.append(CoordSelectedSensors[ii:][0])
        #coords = np.array(coords)
    
    return coords
    

def main(T=int(5e2)): 
    scalingFactor = 50
    N = 80 # number of sensors
    k = 4 # number of selected sensors
    W = 10 # window size

    sensorRadius = np.array(100/scalingFactor)#coverage radius per sensor
    #sensorRadius = []
    #sensorRadius = np.array([1.,1.,1.,1.,1.,2.,2.,2.,2.,2.])    
    capacity = 1.
    d = 0.5e-3 #transmission delay
    mu = 1. #packet size

    #lam = 1.
    lam = d*(1.+2./3.*k)
    
    thresh = 0.75
    
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


    coverageAreaGreedySensSelec = []
    areaWeightedAgeGreedySensSelec = []
    Greedysubmodfunc = []
    AvgGreedysubmodfunc = []
    GreedyTime = []
    
    coverageAreaStreamSensSelec = []
    areaWeightedAgeStreamSensSelec = []
    Streamsubmodfunc = []
    AvgStreamsubmodfunc = []
    StreamTime = []    
    
    coverageAreaModifiedStreamSensSelec = []
    areaWeightedAgeModifiedStreamSensSelec = []
    ModifiedStreamsubmodfunc = []
    AvgModifiedStreamsubmodfunc = []
    ModifiedStreamTime = []

    coverageAreaModifiedThreshSensSelec = []
    areaWeightedAgeModifiedThreshSensSelec = []
    ModifiedThreshsubmodfunc = []
    AvgModifiedThreshsubmodfunc = []
    ModifiedThreshTime = []
    counterVec = []
    
    numIter = 20
        
     
    for jj in tqdm(range(numIter)):
        xcoordSensors = 0 + np.random.rand(N,1)*(rectangleLength-0) 
        ycoordSensors = 0 + np.random.rand(N,1)*(rectangleWidth-0)
        coordSensors = np.concatenate((xcoordSensors,ycoordSensors),axis=1)
        #coordSensors  = np.array([[0.346256,0.794008],[17.6222,1.67842],[1.60685,1.52488],[17.6952,0.376898],[14.8532,1.3532],[5.21618,1.56915],[17.8326,0.501913],[13.8915,0.141149],[0.0616458,0.807074],[12.3948,0.727091]])
     
        temp1coverageAreaGreedySensSelec = []
        temp1areaWeightedAgeGreedySensSelec = []
        temp1Greedysubmodfunc = []
        temp2Greedysubmodfunc = []
        temp1TotalTimeGreedySensSelec = []

        temp1coverageAreaStreamSensSelec = []
        temp1areaWeightedAgeStreamSensSelec = []
        temp1Streamsubmodfunc = []            
        temp2Streamsubmodfunc = []
        temp1TotalTimeStreamSensSelec = []
        
        temp1coverageAreaModifiedStreamSensSelec = []
        temp1areaWeightedAgeModifiedStreamSensSelec = []
        temp1ModifiedStreamsubmodfunc = []            
        temp2ModifiedStreamsubmodfunc = []
        temp1TotalTimeModifiedStreamSensSelec = []

        temp1coverageAreaModifiedThreshSensSelec = []
        temp1areaWeightedAgeModifiedThreshSensSelec = []
        temp1ModifiedThreshsubmodfunc = []            
        temp2ModifiedThreshsubmodfunc = []
        temp1TotalTimeModifiedThreshSensSelec = []

        counter = 0.
        
        for ii in range(len(coordSensors)-W+1):
            partitionsArea, allPossibleSets = findPartitionsAreas(pixelLength, pixelWidth, coordPixels, coordSensors[ii:ii+W], sensorRadius,W)

            # Greedy 
            startTimeGreedySensSelec = time.time()
            tempcoverageAreaGreedySensSelec , tempareaWeightedAgeGreedySensSelec , b_new_SensSelec , tempselectedSensorsGreedySensSelec, tempcoordSelectedSensorsGreedy = GreedySensSelecModel(W, k, d, capacity , mu, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, coordSensors[ii:ii+W], lam)
            endTimeGreedySensSelec = time.time()
            
            if ii==0:
                # Stream
                tempCoordSelectedSensorsStream = tempcoordSelectedSensorsGreedy
                tempcoverageAreaStreamSensSelec = tempcoverageAreaGreedySensSelec
                tempareaWeightedAgeStreamSensSelec = tempareaWeightedAgeGreedySensSelec
                b_new_Stream = b_new_SensSelec
                tempselectedSensorsStreamSensSelec = tempselectedSensorsGreedySensSelec
                startTimeStreamSensSelec = startTimeGreedySensSelec
                endTimeStreamSensSelec = endTimeGreedySensSelec

                # Modified Stream
                tempCoordSelectedSensorsModifiedStream = tempcoordSelectedSensorsGreedy
                tempcoverageAreaModifiedStreamSensSelec = tempcoverageAreaGreedySensSelec
                tempareaWeightedAgeModifiedStreamSensSelec = tempareaWeightedAgeGreedySensSelec
                b_new_ModifiedStream = b_new_SensSelec
                tempselectedSensorsModifiedStreamSensSelec = tempselectedSensorsGreedySensSelec
                startTimeModifiedStreamSensSelec = startTimeGreedySensSelec
                endTimeModifiedStreamSensSelec = endTimeGreedySensSelec
                
                # Thresh
                tempCoordSelectedSensorsModifiedThresh = tempcoordSelectedSensorsGreedy
                tempcoverageAreaModifiedThreshSensSelec = tempcoverageAreaGreedySensSelec
                tempareaWeightedAgeModifiedThreshSensSelec = tempareaWeightedAgeGreedySensSelec
                b_new_ModifiedThresh = b_new_SensSelec
                tempselectedSensorsModifiedThreshSensSelec = tempselectedSensorsGreedySensSelec   
                startTimeModifiedThreshSensSelec = startTimeGreedySensSelec
                endTimeModifiedThreshSensSelec = endTimeGreedySensSelec
                extraTimeModifiedThresh = 0
            else:
                # Stream
                tempCoordSelectedSensorsStream = check_k1(tempselectedSensorsStreamSensSelec,tempCoordSelectedSensorsStream,coordSensors[ii-1:ii+W-1])                
                startTimeStreamSensSelec = time.time()
                tempcoverageAreaStreamSensSelec , tempareaWeightedAgeStreamSensSelec , b_new_Stream , tempselectedSensorsStreamSensSelec, tempCoordSelectedSensorsStream = StreamSensSelecModel(W, k, d, capacity , mu, tempCoordSelectedSensorsStream, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, coordSensors[ii:ii+W], lam)
                endTimeStreamSensSelec = time.time()
                
                # Modified Stream Algo
                #Check if a sensor in the 'k' selected sensors has been removed: return coords of selected sensors without the sensor that left the window
                tempCoordSelectedSensorsModifiedStream = check_k(tempselectedSensorsModifiedStreamSensSelec,tempCoordSelectedSensorsModifiedStream)                
                startTimeModifiedStreamSensSelec = time.time()
                tempcoverageAreaModifiedStreamSensSelec , tempareaWeightedAgeModifiedStreamSensSelec , b_new_ModifiedStream , tempselectedSensorsModifiedStreamSensSelec, tempCoordSelectedSensorsModifiedStream = ModifiedStreamSensSelecModel(W, k, d, capacity , mu, tempCoordSelectedSensorsModifiedStream, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, coordSensors[ii:ii+W], lam)
                endTimeModifiedStreamSensSelec = time.time()
                
                # Thresh Algo
                #Check if a sensor in the 'k' selected sensors has been removed: return coords of selected sensors without the sensor that left the window
                tempCoordSelectedSensorsModifiedThresh = check_k(tempselectedSensorsModifiedThreshSensSelec,tempCoordSelectedSensorsModifiedThresh)
                startTimeModifiedThreshSensSelec = time.time()
                tempcoverageAreaModifiedThreshSensSelec , tempareaWeightedAgeModifiedThreshSensSelec , b_new_ModifiedThresh , tempselectedSensorsModifiedThreshSensSelec, tempCoordSelectedSensorsModifiedThresh, extraTimeModifiedThresh, counter = ModifiedThreshSensSelecModel(W, k, d, capacity , mu, tempCoordSelectedSensorsModifiedThresh, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, coordSensors[ii:ii+W], lam, counter, thresh)
                endTimeModifiedThreshSensSelec = time.time()
            
            # Greedy
            temp1coverageAreaGreedySensSelec.append(tempcoverageAreaGreedySensSelec/areaR*100.)
            temp1areaWeightedAgeGreedySensSelec.append(tempareaWeightedAgeGreedySensSelec*1000.)
            temp1Greedysubmodfunc.append(b_new_SensSelec)
            temp2Greedysubmodfunc.append((1./(ii+1))*np.sum(temp1Greedysubmodfunc))
            temp1TotalTimeGreedySensSelec.append(endTimeGreedySensSelec-startTimeGreedySensSelec)
            
            # Stream
            temp1coverageAreaStreamSensSelec.append(tempcoverageAreaStreamSensSelec/areaR*100.)
            temp1areaWeightedAgeStreamSensSelec.append(tempareaWeightedAgeStreamSensSelec*1000.)
            temp1Streamsubmodfunc.append(b_new_Stream)
            temp2Streamsubmodfunc.append((1./(ii+1))*np.sum(temp1Streamsubmodfunc))
            temp1TotalTimeStreamSensSelec.append(endTimeStreamSensSelec-startTimeStreamSensSelec)
            
            #Modified Stream
            temp1coverageAreaModifiedStreamSensSelec.append(tempcoverageAreaModifiedStreamSensSelec/areaR*100.)
            temp1areaWeightedAgeModifiedStreamSensSelec.append(tempareaWeightedAgeModifiedStreamSensSelec*1000.)
            temp1ModifiedStreamsubmodfunc.append(b_new_ModifiedStream)
            temp2ModifiedStreamsubmodfunc.append((1./(ii+1))*np.sum(temp1ModifiedStreamsubmodfunc))
            temp1TotalTimeModifiedStreamSensSelec.append(endTimeModifiedStreamSensSelec-startTimeModifiedStreamSensSelec)
            
            #Thresh
            temp1coverageAreaModifiedThreshSensSelec.append(tempcoverageAreaModifiedThreshSensSelec/areaR*100.)
            temp1areaWeightedAgeModifiedThreshSensSelec.append(tempareaWeightedAgeModifiedThreshSensSelec*1000.)
            temp1ModifiedThreshsubmodfunc.append(b_new_ModifiedThresh)
            temp2ModifiedThreshsubmodfunc.append((1./(ii+1))*np.sum(temp1ModifiedThreshsubmodfunc))            
            temp1TotalTimeModifiedThreshSensSelec.append(endTimeModifiedThreshSensSelec-startTimeModifiedThreshSensSelec-extraTimeModifiedThresh)

        coverageAreaGreedySensSelec.append(temp1coverageAreaGreedySensSelec)
        areaWeightedAgeGreedySensSelec.append(temp1areaWeightedAgeGreedySensSelec)
        Greedysubmodfunc.append(temp1Greedysubmodfunc)
        AvgGreedysubmodfunc.append(temp2Greedysubmodfunc)
        GreedyTime.append(temp1TotalTimeGreedySensSelec)

        coverageAreaStreamSensSelec.append(temp1coverageAreaStreamSensSelec)
        areaWeightedAgeStreamSensSelec.append(temp1areaWeightedAgeStreamSensSelec)
        Streamsubmodfunc.append(temp1Streamsubmodfunc)        
        AvgStreamsubmodfunc.append(temp2Streamsubmodfunc)
        StreamTime.append(temp1TotalTimeStreamSensSelec)
        
        coverageAreaModifiedStreamSensSelec.append(temp1coverageAreaModifiedStreamSensSelec)
        areaWeightedAgeModifiedStreamSensSelec.append(temp1areaWeightedAgeModifiedStreamSensSelec)
        ModifiedStreamsubmodfunc.append(temp1ModifiedStreamsubmodfunc)        
        AvgModifiedStreamsubmodfunc.append(temp2ModifiedStreamsubmodfunc)
        ModifiedStreamTime.append(temp1TotalTimeModifiedStreamSensSelec)
    
        coverageAreaModifiedThreshSensSelec.append(temp1coverageAreaModifiedThreshSensSelec)
        areaWeightedAgeModifiedThreshSensSelec.append(temp1areaWeightedAgeModifiedThreshSensSelec)
        ModifiedThreshsubmodfunc.append(temp1ModifiedThreshsubmodfunc)        
        AvgModifiedThreshsubmodfunc.append(temp2ModifiedThreshsubmodfunc)
        ModifiedThreshTime.append(temp1TotalTimeModifiedThreshSensSelec)
        counterVec.append(counter)

    # Final averaged output
    coverageAreaGreedySensSelec = np.sum(coverageAreaGreedySensSelec,0)/numIter
    areaWeightedAgeGreedySensSelec = np.sum(areaWeightedAgeGreedySensSelec,0)/numIter
    Greedysubmodfunc = np.sum(Greedysubmodfunc,0)/numIter
    AvgGreedysubmodfunc = np.sum(AvgGreedysubmodfunc,0)/numIter
    GreedyTime = np.sum(GreedyTime,0)/numIter

    coverageAreaStreamSensSelec = np.sum(coverageAreaStreamSensSelec,0)/numIter
    areaWeightedAgeStreamSensSelec = np.sum(areaWeightedAgeStreamSensSelec,0)/numIter
    Streamsubmodfunc = np.sum(Streamsubmodfunc,0)/numIter
    AvgStreamsubmodfunc = np.sum(AvgStreamsubmodfunc,0)/numIter
    StreamTime = np.sum(StreamTime,0)/numIter
    
    coverageAreaModifiedStreamSensSelec = np.sum(coverageAreaModifiedStreamSensSelec,0)/numIter
    areaWeightedAgeModifiedStreamSensSelec = np.sum(areaWeightedAgeModifiedStreamSensSelec,0)/numIter
    ModifiedStreamsubmodfunc = np.sum(ModifiedStreamsubmodfunc,0)/numIter
    AvgModifiedStreamsubmodfunc = np.sum(AvgModifiedStreamsubmodfunc,0)/numIter
    ModifiedStreamTime = np.sum(ModifiedStreamTime,0)/numIter

    coverageAreaModifiedThreshSensSelec = np.sum(coverageAreaModifiedThreshSensSelec,0)/numIter
    areaWeightedAgeModifiedThreshSensSelec = np.sum(areaWeightedAgeModifiedThreshSensSelec,0)/numIter
    ModifiedThreshsubmodfunc = np.sum(ModifiedThreshsubmodfunc,0)/numIter
    AvgModifiedThreshsubmodfunc = np.sum(AvgModifiedThreshsubmodfunc,0)/numIter    
    ModifiedThreshTime = np.sum(ModifiedThreshTime,0)/numIter
    counterVec = np.sum(counterVec,0)
    
    #################  Plots  ######################################
    todayDate = date.today()

    path = os.getcwd() + '\Figures' + '\\' + str(todayDate)
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    
    plt.clf()
    plt.plot(areaWeightedAgeGreedySensSelec, '.-',label='Greedy Sensor Selection')
    plt.plot(areaWeightedAgeStreamSensSelec, 'o-',label='Stream Sensor Selection')
    plt.plot(areaWeightedAgeModifiedStreamSensSelec, '*-',label='Modified Stream Sensor Selection')
    plt.plot(areaWeightedAgeModifiedThreshSensSelec, '--',label='Thresh Sensor Selection')
     #plt.title('Area weighted age as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
      #plt.yscale('log')
    plt.xlabel('Window W', fontsize=12)
    plt.ylabel('Normalized average weighted age [msec]', fontsize=10)
    plt.savefig(os.path.join(path,'Age' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_thresh=' + str(thresh) + '_' + 'lam=' + str(lam)+'.eps'))
    plt.savefig(os.path.join(path,'Age' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_thresh=' + str(thresh) + '_' + 'lam=' + str(lam)+'.pdf'))
      
    plt.clf()
    plt.plot(coverageAreaGreedySensSelec, '.-',label='Greedy Sensor Selection')
    plt.plot(coverageAreaStreamSensSelec, 'o-',label='Stream Sensor Selection')
    plt.plot(coverageAreaModifiedStreamSensSelec, '*-',label='Modified Stream Sensor Selection')
    plt.plot(coverageAreaModifiedThreshSensSelec, '--',label='Tresh Sensor Selection')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xlabel('Window W', fontsize=12)
    plt.ylabel('Coverage [%]', fontsize=10)
    plt.savefig(os.path.join(path,'Coverage' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + 'thresh=' + str(thresh) + '_lam=' + str(lam)+'.eps'))
    plt.savefig(os.path.join(path,'Coverage' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + 'thresh=' + str(thresh) + '_lam=' + str(lam)+'.pdf'))


    plt.clf()
    plt.plot(Greedysubmodfunc, '.-',label='Greedy Sensor Selection')
    plt.plot(Streamsubmodfunc, 'o-',label='Stream Sensor Selection')
    plt.plot(ModifiedStreamsubmodfunc, '*-',label='Modified Stream Sensor Selection')
    plt.plot(ModifiedThreshsubmodfunc, '--',label='Thresh Sensor Selection')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xlabel('Window', fontsize=12)
    plt.ylabel('Submodular function value', fontsize=10)
    plt.savefig(os.path.join(path,'SubmodFn' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + '_thresh=' + str(thresh) + '_lam=' + str(lam)+' 2.eps'))
    plt.savefig(os.path.join(path,'SubmodFn' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + '_thresh=' + str(thresh) + '_lam=' + str(lam)+' 2.pdf'))


    plt.clf()
    plt.plot(AvgGreedysubmodfunc, '.-',label='Greedy Sensor Selection')
    plt.plot(AvgStreamsubmodfunc, 'o-',label='Stream Sensor Selection')
    plt.plot(AvgModifiedStreamsubmodfunc, '*-',label='Modified Stream Sensor Selection')
    plt.plot(AvgModifiedThreshsubmodfunc, '--',label='Thresh Sensor Selection')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xlabel('Window', fontsize=12)
    plt.ylabel('Submodular avg function value', fontsize=10)
    plt.savefig(os.path.join(path,'AVGSubmodFn' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + '_thresh=' + str(thresh) + '_lam=' + str(lam)+'.eps'))
    plt.savefig(os.path.join(path,'AVGSubmodFn' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + '_thresh=' + str(thresh) + '_lam=' + str(lam)+'.pdf'))



    plt.clf()
    plt.plot(GreedyTime, '.-',label='Greedy Sensor Selection')
    plt.plot(StreamTime, 'o-',label='Stream Sensor Selection')
    plt.plot(ModifiedStreamTime, '*-',label='Modified Stream Sensor Selection')
    plt.plot(ModifiedThreshTime, '--',label='Thresh Sensor Selection')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xlabel('Window', fontsize=12)
    plt.ylabel('Average time [sec]', fontsize=10)
    plt.savefig(os.path.join(path,'AlgoTime' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + '_thresh=' + str(thresh) + '_lam=' + str(lam)+'.eps'))
    plt.savefig(os.path.join(path,'AlgoTime' + '_N=' + str(N) + '_k=' + str(k) + '_W=' + str(W) + '_' + '_thresh=' + str(thresh) + '_lam=' + str(lam)+'.pdf'))



    
if __name__ == "__main__":
    main()