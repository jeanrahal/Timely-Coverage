import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools 
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.optimize import brentq


# Function that computes the age of the region of intersection 
    
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
        ratePerSensor = update(N,ratePerSensor, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea, t, capacity,d)

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


def compute_b(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors ,ratePerSensor, currSensor, allPossibleSets, lam):
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
            tempAge = d + (1./(n+1.))*(1/ratePerSensor) 
            #tempAge = (n+2.)/(n+1.)*(1./ratePerSensor)
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
    
    return b,totalCoveredArea,areaWeightedAge,selectedPartitionsArea


def SensSelecModel(N, d, capacity, mu, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, numSelectedSensors, thresh = 2.):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(1,N+1,1)
    
    
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    lam = d*(1.+2./3.*numSelectedSensors)
    
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
        setofSelectedSensors.append(selectedSensor)
                
    #setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return coverageArea , areaWeightedAge/(coverageArea) , setofSelectedSensors


def main(T=int(5e2)): 
    scalingFactor = 50
    N = 8 # number of sensors
    k = 4 # number of selected sensors
    
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    sensorRadius = 100/scalingFactor #coverage radius per sensor
    capacity = 1.
    d = 0.5e-3 #transmission delay
    mu = 1. #packet size
    
    lam_min = d*(1.+2./3.*numSelectedSensors)
    #lam = np.arange(lam_min,2.*lam_min,0.00005)
    lam = np.arange(0.01,3.,0.05)
    
    rectangleLength = 1000/scalingFactor
    rectangleWidth = 10/scalingFactor
    #areaR = rectangleLength*rectangleWidth
    numSquaresperLength = int(rectangleLength*10)
    numSquaresperWidth = int(rectangleWidth*10)
    
    pixelLength = rectangleLength/numSquaresperLength
    pixelWidth = rectangleWidth/numSquaresperWidth
    
    xPosCenterPixel1 = pixelLength/2
    yPosCenterPixel1 = pixelWidth/2
    
    coordPixels = generatePixelsCenters(xPosCenterPixel1, yPosCenterPixel1, pixelLength, pixelWidth, numSquaresperLength, numSquaresperWidth)


    coverageAreaSensSelec = []
    areaWeightedAgeSensSelec = []
    selectedSensorsSensSelec = []

    numIter = 30

    for ii in range(len(lam)):
         temp1coverageAreaSensSelec = []
         temp1areaWeightedAgeSensSelec = []
         temp1selectedSensorsSensSelec = []
         
         for jj in range(numIter):
             xcoordSensors = 0 + np.random.rand(N,1)*(rectangleLength-0) 
             ycoordSensors = 0 + np.random.rand(N,1)*(rectangleWidth-0)
             coordSensors = np.concatenate((xcoordSensors,ycoordSensors),axis=1)
             
             partitionsArea , allPossibleSets = findPartitionsAreas(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,N)

             tempcoverageAreaSensSelec , tempareaWeightedAgeSensSelec , tempselectedSensorsSensSelec = SensSelecModel(N, d, capacity , mu, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor, sensorRadius*scalingFactor, scalingFactor, lam[ii], numSelectedSensors, thresh = 2.)
             
             temp1coverageAreaSensSelec.append(tempcoverageAreaSensSelec)
             temp1areaWeightedAgeSensSelec.append(tempareaWeightedAgeSensSelec)
             temp1selectedSensorsSensSelec.append(len(tempselectedSensorsSensSelec))
     
          
         coverageAreaSensSelec.append(np.sum(temp1coverageAreaSensSelec)/numIter)
         areaWeightedAgeSensSelec.append(np.sum(temp1areaWeightedAgeSensSelec)/numIter)
         selectedSensorsSensSelec.append(np.sum(temp1selectedSensorsSensSelec)/numIter)
    
    
    plt.clf()
    plt.scatter(coverageAreaSensSelec, areaWeightedAgeSensSelec, label='Sensor Selection')
    plt.legend()
    plt.grid()
    plt.ylim(min(areaWeightedAgeSensSelec), max(areaWeightedAgeSensSelec))
    #plt.yscale('log')
    plt.xlabel('Coverage area ($m^2$)', fontsize=12)
    plt.ylabel('Normalized area weighted age (seconds)', fontsize=5)
    plt.savefig('AgevsCo1.eps')
    plt.savefig('AgevsCo1.pdf')







    
if __name__ == "__main__":
    main()