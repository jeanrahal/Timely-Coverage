import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import matplotlib.patches as patches
from tqdm import tqdm 
import random
import math
import copy
import time
import itertools 
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.optimize import brentq
import os
from datetime import date
import seaborn as sns; sns.set()
import scipy.io as sio 


todayDate = date.today()

path = os.getcwd() + '\Figures' + '\\' + str(todayDate)
    
if not os.path.exists(path):
    os.mkdir(path)

    
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
    ratePerSensor[ratePerSensor<=0]= d
    ratePerSensor[ratePerSensor<=5]= 5
    #ratePerSensor[ratePerSensor>=30]= 30
    
    return ratePerSensor


def descent(N,update, d, numSelectedSensors, setofSelectedSensors, allPossibleSets, selectedPartitionsArea, capacity, T=int(250)):
    ratePerSensor =  d*np.ones(int(numSelectedSensors))
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
    #ratePerSensor[ratePerSensor<=5] = 5
    #ratePerSensor[ratePerSensor>=30] = 30
    
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

def findCarCoords(coordSensors,carDimensions, boxDim, N):
    '''
    input: cars coordinates, car dimensions
    output: 4 cars corners coordinates
    '''
    
    carsCoords = []
    
    for car in range(N):
        a = coordSensors[car][0]+carDimensions[0]/2
        b = coordSensors[car][1]+carDimensions[1]/2
        c = coordSensors[car][1]-carDimensions[1]/2
        d = coordSensors[car][0]-carDimensions[0]/2
        
        carsCoords.append([[a if a<= boxDim[0] else boxDim[0], b if b<= boxDim[1] else boxDim[1]],
                           [a if a<= boxDim[0] else boxDim[0], c if c>= 0 else 0],
                           [d if d>= 0 else 0,b if b<= boxDim[1] else boxDim[1]],
                           [d if d>= 0 else 0,c if c>= 0 else 0]])
        
    return carsCoords


def plotObstructions(coordPixels, coordSensors, carsCoords, carDimensions, obstructions, boxDim, pixelLength = 1., pixelWidth = 1.):
    
    plt.clf()
    fig,ax = plt.subplots(1)
    rect = patches.Rectangle((0,0),boxDim[0],boxDim[1],linewidth=1,edgecolor='black',facecolor='none')
    ax.add_patch(rect)

    plt.xlim(0, boxDim[0])
    plt.ylim(0, boxDim[1])
    
    edgecol = ['red','blue','green','orange','yellow','purple','black','gray','silver','gold','maroon','coral']
    facecol = ['red','blue','green','orange','yellow','purple','black','gray','silver','gold','maroon','coral']
    
    for ii in range(len(coordPixels)):
        rect = patches.Rectangle((coordPixels[ii][0]-pixelLength/2,coordPixels[ii][1]-pixelWidth/2),pixelLength,pixelWidth,linewidth=1,edgecolor='black',facecolor='none')
        ax.add_patch(rect)
    
    for sensor in range(len(coordSensors)):
        rect = patches.Rectangle((carsCoords[sensor][-1][0],carsCoords[sensor][-1][1]),carDimensions[0],carDimensions[1],linewidth=1,edgecolor=edgecol[sensor],facecolor='none')
        ax.add_patch(rect)
        circle = plt.Circle((coordSensors[sensor][0], coordSensors[sensor][1]), 20, edgecolor=edgecol[sensor],facecolor='none')
        ax.add_patch(circle)
        for obs in range(len(obstructions[sensor])):
            temp = obstructions[sensor][obs]
            rect = patches.Rectangle(((coordPixels[temp][0]-pixelLength/2),(coordPixels[temp][1]-pixelWidth/2)),pixelLength,pixelWidth,linewidth=1,edgecolor=edgecol[sensor],facecolor=facecol[sensor])
            ax.add_patch(rect)
            
        
    plt.savefig(os.path.join(path, 'Obstructed Views.eps'))
    plt.savefig(os.path.join(path, 'Obstructed Views.pdf'))
            
    
    return None


#################     PER CAR REGION OF INTEREST   ###########################
def findCoordregionOfInterestPerSensor_1(coordPixels,coordSensors,N,length_box_per_car,width_box_per_car,boxDim):
    coordRegionofInterest = []
    
    for car in range(N):
        a = coordSensors[car][0]+length_box_per_car/2
        b = coordSensors[car][1]+width_box_per_car
        c = coordSensors[car][1]-width_box_per_car
        d = coordSensors[car][0]-length_box_per_car/2
        
        coordRegionofInterest.append([[a if a<= boxDim[0] else boxDim[0], b if b<= boxDim[1] else boxDim[1]],
                                      [a if a<= boxDim[0] else boxDim[0], c if c>= 0 else 0],
                                      [d if d>= 0 else 0, b if b<= boxDim[1] else boxDim[1]],
                                      [d if d>= 0 else 0, c if c>= 0 else 0]])
        
    return coordRegionofInterest
    

def findobstructedPixelsinBox_1(N, pixelsPerBoxPerSensor, coordSensors, coordPixels, sensorRadius, carDimensions, carRoI, boxDim, plot):
    obstructedPixelsinBox = []   
    ''' 
    Goal: Find the coord of obstructed pixels for each sensor in box of interest
    
    Inputs: coord Pixels, coord sensors, sensor radius, # of sensors, car dimensions: W x L
    Output: list of lists, each list is the set of obstructed pixels per sensor per box
    '''    
    carsCoords = findCarCoords(coordSensors,carDimensions, boxDim, N)
    
    for sensor in range(N):
        ## Find obstructing other sensors
        obstructingSensors = []
        for otherSensor in range(N):
            if otherSensor != sensor:
                currSensorCoord = coordSensors[otherSensor]
                if pixelisInCircle(sensor,sensorRadius,otherSensor,coordSensors,coordSensors) and currSensorCoord[0] >= coordSensors[sensor][0] - carRoI[0] and currSensorCoord[0] <= coordSensors[sensor][0] + carRoI[0] and currSensorCoord[1] >= coordSensors[sensor][1] - carRoI[1] and currSensorCoord[1] <= coordSensors[sensor][1] + carRoI[1]:
                    obstructingSensors.append(otherSensor)
        ########
        obstructedPixelsinBox.append([])
        for pixel in range(len(pixelsPerBoxPerSensor[sensor])):
            # Step 1: check if the selected pixel is first within the range of observability of the selected sensor
            if pixelisInCircle(sensor,sensorRadius,pixelsPerBoxPerSensor[sensor][pixel],coordPixels,coordSensors) == 1:
                # Step 2: check if the pixel is obstructed
                if obstructingSensors:
                    for otherSensor in range(len(obstructingSensors)):
                        # Compute all the slopes between the selected sensor and other sensors
                        slope = []
                        for ii in range(len(carsCoords[obstructingSensors[otherSensor]])):
                            slope.append((coordSensors[sensor][1]-carsCoords[obstructingSensors[otherSensor]][ii][1])/(coordSensors[sensor][0]-carsCoords[obstructingSensors[otherSensor]][ii][0]))
                        
                        pickedSlopes = np.array([min(slope),max(slope)])
                        distSensors = np.linalg.norm(coordSensors[sensor]-coordSensors[obstructingSensors[otherSensor]])
                        distSensorToPixel = np.linalg.norm(coordSensors[sensor]-coordPixels[pixel])
                        distPixelTootherSens = np.linalg.norm(coordPixels[pixel]-coordSensors[obstructingSensors[otherSensor]])
                        slopeSensorPixel = (coordSensors[sensor][1]-coordPixels[pixel][1])/(coordSensors[sensor][0]-coordPixels[pixel][0])
                        
                        if distSensorToPixel > distSensors and distSensorToPixel > distPixelTootherSens and slopeSensorPixel >= pickedSlopes[0] and slopeSensorPixel <= pickedSlopes[1]:
                            obstructedPixelsinBox[sensor].append(pixelsPerBoxPerSensor[sensor][pixel])
                            break
            elif pixelisInCircle(sensor,sensorRadius,pixelsPerBoxPerSensor[sensor][pixel],coordPixels,coordSensors) == 0:
                obstructedPixelsinBox[sensor].append(pixelsPerBoxPerSensor[sensor][pixel])
                           
    if plot == 1:
        plotObstructions(coordPixels, coordSensors, carsCoords, carDimensions, obstructedPixelsinBox, boxDim)    
    
    return obstructedPixelsinBox


def sortObstructedPixelsperSensorinBox_1(N, obstructedPixelsinBox, labeledMatrixPixel,labeledPixels,regionOfInterestPerSensor, numSquaresPerLength, numSquaresPerWidth):
    obstructedRegionsPerSensor = []
    
    for sensor in range(N):
        ## Form the box around the sensors: Find the number of pixels per length and width
        currRoI = regionOfInterestPerSensor[sensor] #
        numPixelsperLength = (currRoI[0][0] - currRoI[2][0])*numSquaresPerLength
        numPixelsperWidth = (currRoI[0][1] - currRoI[1][1])*numSquaresPerWidth
        
        # Store the pixels of the current sensor below
        obstructedRegionsPerSensor.append([])
        selectedSensorObstructedPixels = obstructedPixelsinBox[sensor]
        regionsPerCurrentSensor = []
        # Check if the set is non-empty
        if selectedSensorObstructedPixels:
            for currentPixel in range(len(selectedSensorObstructedPixels)):
                # We now need to split the pixels:
                #   regionsPerPixel: Store the obstructed regions per pixel = [Sensor1: [ Region 1 ], [ Region 2 ] Sensor2: [ Region 1 ], [ Region 2 ] ,[ Region 3 ]]
                
                # Find if the current pixel belongs to an older region or to a new one:
                # If it belongs to an old region, return the region ID and add the pixel to it. Else append new region.
                newRegionForCurrentPixel, regionID = detectRegion(selectedSensorObstructedPixels[currentPixel],regionsPerCurrentSensor,labeledMatrixPixel,labeledPixels,numPixelsperLength,numPixelsperWidth)
                if newRegionForCurrentPixel:
                    # We need to create a new region for the current sensor
                    regionsPerCurrentSensor.append(())
                    regionsPerCurrentSensor[-1] = (selectedSensorObstructedPixels[currentPixel],)
                
                else:
                    regionsPerCurrentSensor[regionID] = regionsPerCurrentSensor[regionID] + (selectedSensorObstructedPixels[currentPixel],)
        
        obstructedRegionsPerSensor[sensor].append(regionsPerCurrentSensor)                    
                    
    return obstructedRegionsPerSensor
    

def findPixelsinRegionOfInterest_1(N,coordPixels,coordSensors,length_box_per_car,width_box_per_car,labeledPixels):
    pixelsPerBoxPerSensor = []
    
    for sensor in range(N):
        pixelsPerBoxPerSensor.append(())
        for pixel in range(len(coordPixels)):
            currPixelCoord = coordPixels[pixel]
            if currPixelCoord[0] >= coordSensors[sensor][0] - length_box_per_car and currPixelCoord[0] <= coordSensors[sensor][0] + length_box_per_car and currPixelCoord[1] >= coordSensors[sensor][1] - width_box_per_car and currPixelCoord[1] <= coordSensors[sensor][1] + width_box_per_car:
                pixelsPerBoxPerSensor[sensor] = pixelsPerBoxPerSensor[sensor] + (labeledPixels[pixel],) 


    return pixelsPerBoxPerSensor    


###############################################################################


def findObstructions(coordPixels, coordSensors, sensorRadius, labeledPixels, N, carDimensions, boxDim, plot):
    obstructions = []
    
    ''' 
    Find the coord of obstructed pixels for each sensor
    
    input: coord Pixels, coord sensors, sensor radius, # of sensors, car dimensions: W x L
    output: list of lists, each list is the set of obstructed pixels per sensor
    '''
    
    carsCoords = findCarCoords(coordSensors,carDimensions, boxDim, N)
    
    for sensor in range(N):
        obstructions.append([])
        for pixel in range(len(coordPixels)):
            # Step 1: check if the selected pixel is first within the range of observability of the selected sensor
            if pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors) == 1:
                # Step 2: check if the pixel is obstructed
                for otherSensor in range(N):
                    if otherSensor != sensor:
                        # Compute all the slopes between the selected sensor and other sensors
                        slope = []
                        for ii in range(len(carsCoords[otherSensor])):
                            slope.append((coordSensors[sensor][1]-carsCoords[otherSensor][ii][1])/(coordSensors[sensor][0]-carsCoords[otherSensor][ii][0]))
                        
                        pickedSlopes = np.array([min(slope),max(slope)])
                        distSensors = np.linalg.norm(coordSensors[sensor]-coordSensors[otherSensor])
                        distSensorToPixel = np.linalg.norm(coordSensors[sensor]-coordPixels[pixel])
                        distPixelTootherSens = np.linalg.norm(coordPixels[pixel]-coordSensors[otherSensor])
                        slopeSensorPixel = (coordSensors[sensor][1]-coordPixels[pixel][1])/(coordSensors[sensor][0]-coordPixels[pixel][0])
                        
                        if distSensorToPixel > distSensors and distSensorToPixel > distPixelTootherSens and slopeSensorPixel >= pickedSlopes[0] and slopeSensorPixel <= pickedSlopes[1]:
                            obstructions[sensor].append(labeledPixels[pixel])
                            break
    
    if plot == 1:
        plotObstructions(coordPixels, coordSensors, carsCoords, carDimensions, obstructions, boxDim)
                        
    return obstructions


def findneighborPixels(index, pixel,numPixelsperLength,numPixelsperWidth):
    if index[0] == 0 and index[1] == 0:     # Upper-Left corner
        neighbourPixels = np.array([pixel+1, pixel+numPixelsperLength, pixel+numPixelsperLength+1])
    
    elif index[0] == 0 and index[1] == numPixelsperLength-1: # Upper-Right corner
        neighbourPixels = np.array([pixel-1, pixel+numPixelsperLength-1, pixel+numPixelsperLength])
        
    elif index[0] == numPixelsperWidth-1 and index[1] == 0: #Lower-Left corner
        neighbourPixels = np.array([pixel-numPixelsperLength, pixel-numPixelsperLength+1, pixel+1])
        
    elif index[0] == numPixelsperWidth-1 and index[1] == numPixelsperLength-1: #Lower-right corner
        neighbourPixels = np.array([pixel-numPixelsperLength-1, pixel-numPixelsperLength, pixel-1])
        
    elif index[0] == 0: #first row without the corners
        neighbourPixels = np.array([pixel-1, pixel+1, pixel+numPixelsperLength-1, pixel+numPixelsperLength, pixel+numPixelsperLength+1])
        
    elif index[0] == numPixelsperWidth-1: #last row without the corners
        neighbourPixels = np.array([pixel-1, pixel+1, pixel-numPixelsperLength-1, pixel-numPixelsperLength, pixel-numPixelsperLength+1])
    
    elif index[1] == 0: #first col without the corners
        neighbourPixels = np.array([pixel-numPixelsperLength, pixel-numPixelsperLength+1, pixel+1, pixel+numPixelsperLength+1, pixel+numPixelsperLength])
        
    elif index[1] == numPixelsperLength-1: #last col without the corners    
        neighbourPixels = np.array([pixel-numPixelsperLength, pixel-numPixelsperLength-1, pixel-1, pixel+numPixelsperLength-1, pixel+numPixelsperLength])
        
    else:
        neighbourPixels = np.array([pixel-numPixelsperLength-1,pixel-numPixelsperLength,pixel-numPixelsperLength+1,
                                    pixel - 1 ,       pixel + 1,
                                    pixel + numPixelsperLength-1,pixel+numPixelsperLength,pixel+numPixelsperLength+1])    
    return neighbourPixels


def detectRegion(currentPixel,regionsPerCurrentSensor,labeledMatrixPixel,labeledPixels,numPixelsperLength,numPixelsperWidth):
    newRegionForCurrentPixel = 1
    regionID = 0
    
    currentPixelLabel = labeledPixels[currentPixel]
    index = np.where(labeledMatrixPixel == currentPixel)
    # Find the neighbour pixels of the current pixel(~): [1,2,3
                                                    #    4,~,5
                                                    #    6,7,8]

    neighbourPixels = findneighborPixels(index, currentPixelLabel,numPixelsperLength,numPixelsperWidth)
    
    for nn in range(len(neighbourPixels)):
        for ii in range(len(regionsPerCurrentSensor)):
            if neighbourPixels[nn] in regionsPerCurrentSensor[ii]:
                regionID = ii
                newRegionForCurrentPixel = 0
                break
        if newRegionForCurrentPixel == 0:
            break
        
    return newRegionForCurrentPixel , regionID 


def sortObstructedPixelPerSensor(labeledMatrixPixel,labeledPixels,obstructedLabeledPixelsperSensor,numPixelsperLength,numPixelsperWidth,N):
    
    obstructedRegionsPerSensor = []
    
    for sensor in range(N):
        # Store the pixels of the current sensor below
        obstructedRegionsPerSensor.append([])
        selectedSensorObstructedPixels = obstructedLabeledPixelsperSensor[sensor]
        regionsPerCurrentSensor = []
        # Check if the set is non-empty
        if selectedSensorObstructedPixels:
            for currentPixel in range(len(selectedSensorObstructedPixels)):
                # We now need to split the pixels:
                #   regionsPerPixel: Store the obstructed regions per pixel = [Sensor1: [ Region 1 ], [ Region 2 ] Sensor2: [ Region 1 ], [ Region 2 ] ,[ Region 3 ]]
                
                # Find if the current pixel belongs to an older region or to a new one:
                # If it belongs to an old region, return the region ID and add the pixel to it. Else append new region.
                newRegionForCurrentPixel , regionID = detectRegion(selectedSensorObstructedPixels[currentPixel],regionsPerCurrentSensor,labeledMatrixPixel,labeledPixels,numPixelsperLength,numPixelsperWidth)
                if newRegionForCurrentPixel:
                    # We need to create a new region for the current sensor
                    regionsPerCurrentSensor.append(())
                    regionsPerCurrentSensor[-1] = (selectedSensorObstructedPixels[currentPixel],)
                
                else:
                    regionsPerCurrentSensor[regionID] = regionsPerCurrentSensor[regionID] + (selectedSensorObstructedPixels[currentPixel],)
        
        obstructedRegionsPerSensor[sensor].append(regionsPerCurrentSensor)                    
                    
    return obstructedRegionsPerSensor 
    

def putWeightonRegions(sortedObstructedPixelsperSensor,N):
    weightedPixelsPerSensor = []
    
    for sensor in range(N):
        currentObstructedPixels = sortedObstructedPixelsperSensor[sensor]
        weightedPixelsPerSensor.append([])
        temp = []
        
        for ii in range(len(currentObstructedPixels[0])): # Loop over the regions
            if currentObstructedPixels[0]:
                temp.append(len(currentObstructedPixels[0][ii]))
            
        # divide by total area
        weightedPixelsPerSensor[sensor] = 1./np.sum(temp) 

    return weightedPixelsPerSensor



def weightPixels(labeledMatrixPixel,weightedRegionsPerSensor,sortedObstructedPixelsperSensor,N):
    weightedMap = np.zeros(np.shape(labeledMatrixPixel))
    
    for sensor in range(N):
        #for ii in range(len(weightedRegionsPerSensor[sensor])):
        currentWeight = weightedRegionsPerSensor[sensor]#[ii]
        for jj in range(len(sortedObstructedPixelsperSensor[sensor][0])):
            for kk in range(len(sortedObstructedPixelsperSensor[sensor][0][jj])):
                index = np.where(labeledMatrixPixel == sortedObstructedPixelsperSensor[sensor][0][jj][kk])
                weightedMap[index[0][0]][index[1][0]] = weightedMap[index[0][0]][index[1][0]] + currentWeight

    return weightedMap 


def divideMapintoRegions(weightedMap, labeledMatrixPixel, labeledPixels, numPixelsperLength, numPixelsperWidth):
    '''
    input: weighted map with all the pixels labels
    output: region ID: each row corresponds to ta different region: each row has the first input = weight, then all the 
            rest correspond to the pixels of this region
    '''
    
    
    regionID = []
    IDMap = -1*np.ones(np.shape(labeledMatrixPixel))
    
    ID = -1
    for pixel in range(len(labeledPixels)):
        index = np.where(labeledMatrixPixel == pixel)
        # Find the neighbour pixels of the current pixel(~): [1,2,3
                                                        #    4,~,5
                                                        #    6,7,8]
    
        neighbourPixels = findneighborPixels(index, pixel,numPixelsperLength,numPixelsperWidth)

        
        
        # Remove the negative indices (for the pixels on the borders of the box)
        #new_neighbourPixels = neighbourPixels[neighbourPixels >= 0]
        
        
        for nn in range(len(neighbourPixels)):
            check = 0
            index_neighbour = np.where(labeledMatrixPixel == neighbourPixels[nn])
            if  weightedMap[index[0][0]][index[1][0]] == weightedMap[index_neighbour[0][0]][index_neighbour[1][0]]:
                if IDMap[index_neighbour[0][0]][index_neighbour[1][0]] >= 0:
                    IDMap[index[0][0]][index[1][0]] = IDMap[index_neighbour[0][0]][index_neighbour[1][0]]
                    temp_ID = int(IDMap[index[0][0]][index[1][0]])
                    regionID[temp_ID] = regionID[temp_ID] + (pixel,)
                    check = 1
                    break
        if check == 0:
            ID = ID + 1 #new ID
            IDMap[index[0][0]][index[1][0]] = ID
            regionID.append(())
            regionID[-1] = (weightedMap[index[0][0]][index[1][0]],) + (pixel,) # new pixel is added as follows: weight + pixels
    
    
    return regionID, IDMap


def findSensorsPerPixel(selectedSensors, labeledMatrixPixel, sensorRadius, coordPixels, coordSensors):
    numSensorsPerPixelMap = np.zeros(np.shape(labeledMatrixPixel)) 
    
    for pixel in range(np.prod(np.shape(labeledMatrixPixel))):
        index = np.where(labeledMatrixPixel == pixel)
        
        for ss in range(len(selectedSensors)):
            selectedSensor = selectedSensors[ss]
            if pixelisInCircle(selectedSensor-1,sensorRadius,pixel,coordPixels,coordSensors) == 1:
                numSensorsPerPixelMap[index[0][0]][index[1][0]] += 1
    
    return numSensorsPerPixelMap

def findPartitionsAreas(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,labeledPixels,labeledMatrixPixel,N,carDimensions,boxDim,obstructedLabeledPixelsperSensor):
    tempPartitionsPixels = np.zeros(2**N-1)
    partitionsPixels = np.zeros(2**N-1)
    temp = np.zeros(2**N-1)
    temp1 = []
    allPossibleSets = []
    
    for ii in range(1,N+1):
        hello = findsubsets(np.arange(0,N,1),ii) 
        #hello1 = (np.asarray(hello))
        for jj in range(len(hello)):
            allPossibleSets.append(list(hello[jj]))
        
    
    for pixel in range(len(coordPixels)):
        for sensor in range(N):
            if pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors) == 1 and labeledPixels[pixel] not in obstructedLabeledPixelsperSensor[sensor]:
               tempPartitionsPixels[sensor] = tempPartitionsPixels[sensor] + 1 
        
        if np.sum(tempPartitionsPixels) > 1:
            idxOnes = np.nonzero(tempPartitionsPixels)
            for ii in range(idxOnes[0].size):
                temp1.append(idxOnes[0][ii])        
            idxPartition = allPossibleSets.index(temp1)
            temp[idxPartition] = 1
        else:
            temp = tempPartitionsPixels
            
        partitionsPixels = partitionsPixels + temp
        
        tempPartitionsPixels = np.zeros(2**N-1)
        temp = np.zeros(2**N-1)
        temp1 = []
        
        
    partitionsArea = partitionsPixels*pixelLength*pixelWidth
    
    
    # Find indices of non-zero weights
    idxNonZeroArea = np.nonzero(partitionsArea)
    
    # Remove the 0 weights entries from newPartitionWeight, as well as allPossibleSets
    partitionsArea = partitionsArea[idxNonZeroArea]
    allPossibleSets = list(np.array(allPossibleSets)[idxNonZeroArea])
    
    return partitionsArea, allPossibleSets


def findPartitionsAreasFaster(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,labeledPixels,labeledMatrixPixel,N,carDimensions,boxDim,obstructedLabeledPixelsperSensor):
   ### Note: 'allPossibleSets' and 'partitionsWeights' have the same ordering of the partitions as well as their assigned weights
    allPossibleSets = []
    partitionsArea = []
          
    # Step 1: go over all the pixels in the map
    for pixel in range(len(coordPixels)):
        sensorsSeePixel = []
        #tempSensorsSeePixel = []
        
        for sensor in range(N):                            
            if pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors) == 1 and labeledPixels[pixel] not in obstructedLabeledPixelsperSensor[sensor]:
                #Append the sensors that see the current pixel
                sensorsSeePixel.append(sensor)

                
        # Set of partitions is not empty
        if sensorsSeePixel:  
            # Check if allPossibleSets already has the set of sensors we're looking for
            # 1- if the set is already available, add the partition
            if sensorsSeePixel in allPossibleSets:
                idxSet = allPossibleSets.index(sensorsSeePixel)
                partitionsArea[idxSet] += 1
                
            # 2- if the set is not available, add it    
            else:
                allPossibleSets.append(sensorsSeePixel)
                partitionsArea.append(1)
            
    partitionsArea = np.array(partitionsArea)
  
    return partitionsArea, allPossibleSets




def findPartitionsWeights(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,labeledPixels,labeledMatrixPixel,N,carDimensions,boxDim,obstructedLabeledPixelsperSensor,weightedMap):
    tempPartitionsPixels = np.zeros(2**N-1)
    partitionsPixels = np.zeros(2**N-1)
    temp = np.zeros(2**N-1)
    temp1 = []
    allPossibleSets = []
    
    for ii in range(1,N+1):
        hello = findsubsets(np.arange(0,N,1),ii) 
        #hello1 = (np.asarray(hello))
        for jj in range(len(hello)):
            allPossibleSets.append(list(hello[jj]))
        
    
    for pixel in range(len(coordPixels)):
        index = np.where(labeledMatrixPixel == pixel)
        weightPixel = weightedMap[index[0][0]][index[1][0]]
        
        if weightPixel > 0:
            for sensor in range(N):            
                
                if pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors) == 1 and labeledPixels[pixel] not in obstructedLabeledPixelsperSensor[sensor]:
                   tempPartitionsPixels[sensor] = tempPartitionsPixels[sensor] + 1 
            
            if np.sum(tempPartitionsPixels) > 1: #pixel is seen by more than one sensor ?
                idxOnes = np.nonzero(tempPartitionsPixels)
                for ii in range(idxOnes[0].size):
                    temp1.append(idxOnes[0][ii])        
                idxPartition = allPossibleSets.index(temp1)
                temp[idxPartition] = weightPixel
            else:
                temp = tempPartitionsPixels*weightPixel
                
            partitionsPixels += temp
            
            tempPartitionsPixels = np.zeros(2**N-1)
            temp = np.zeros(2**N-1)
            temp1 = []
    
    partitionsWeights = partitionsPixels
        
    
    #######  Optimize the code by decreasing the size of the partitionWeights array 
    # Find indices of non-zero weights
    idxNonZeroWeight = np.nonzero(partitionsWeights)
    
    # Remove the 0 weights entries from newPartitionWeight, as well as allPossibleSets
    partitionsWeights = partitionsWeights[idxNonZeroWeight]
    allPossibleSets = list(np.array(allPossibleSets)[idxNonZeroWeight])
    
    return partitionsWeights, allPossibleSets



def findPartitionsWeightsFaster(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,labeledPixels,labeledMatrixPixel,N,carDimensions,boxDim,obstructedLabeledPixelsperSensor,weightedMap):
    ### Note: 'allPossibleSets' and 'partitionsWeights' have the same ordering of the partitions as well as their assigned weights
    allPossibleSets = []
    partitionsWeights = []
          
    # Step 1: go over all the pixels in the map
    for pixel in range(len(coordPixels)):
        sensorsSeePixel = []
        #tempSensorsSeePixel = []
        index = np.where(labeledMatrixPixel == pixel)
        weightPixel = weightedMap[index[0][0]][index[1][0]]
        
        if weightPixel > 0:
            for sensor in range(N):                            
                if pixelisInCircle(sensor,sensorRadius,pixel,coordPixels,coordSensors) == 1 and labeledPixels[pixel] not in obstructedLabeledPixelsperSensor[sensor]:
                    #Append the sensors that see the current pixel
                    sensorsSeePixel.append(sensor)
            #sensorsSeePixel.append(tempSensorsSeePixel) 
            
#            # If the set of partitions is empty, append the first one
#            if not sensorsSeePixel:
#                allPossibleSets.append(sensorsSeePixel)
#                partitionsWeights.append(weightPixel)
                    
            # Set of partitions is not empty
            if sensorsSeePixel:  
                # Check if allPossibleSets already has the set of sensors we're looking for
                # 1- if the set is already available, add the partition
                if sensorsSeePixel in allPossibleSets:
                    idxSet = allPossibleSets.index(sensorsSeePixel)
                    partitionsWeights[idxSet] += weightPixel
                    
                # 2- if the set is not available, add it    
                else:
                    allPossibleSets.append(sensorsSeePixel)
                    partitionsWeights.append(weightPixel)
            
    partitionsWeights = np.array(partitionsWeights)
    
    return partitionsWeights, allPossibleSets



def baselineModel(ratePerSensor , d, partitionsWeights , allPossibleSets, scalingFactor):
    weightedAge = 0.
    coverageWeights = np.sum(partitionsWeights)
    percentageCoverageWeight = np.sum(partitionsWeights)/np.sum(partitionsWeights)*100.
    AgePerPartition = []
    for ii in range(len(partitionsWeights)):
        n = len(allPossibleSets[ii])
        tempAge = d + (1./(n+1.))*(1/ratePerSensor) 
        #tempAge = (n+2.)/(n+1.)*(1/ratePerSensor)
        AgePerPartition.append(tempAge)
    
    weightedAge = np.sum(partitionsWeights*AgePerPartition)/coverageWeights
    
    return percentageCoverageWeight, weightedAge



def compute_b_1(N, d, mu, partitionsWeight, setofSelectedSensors, setofSensors ,ratePerSensor, currSensor, allPossibleSets,weightedMap,lam):
    b = 0.
    AgePerPartition = []
    coveredWeight = []
    tempP = np.zeros(len(partitionsWeight))
    newPartitionWeight = np.zeros(len(partitionsWeight))
    
    startTime = time.time()
    if not setofSelectedSensors:
        currSensors = np.array(currSensor)
        for ii in range(len(partitionsWeight)):
            if currSensors in allPossibleSets[ii]:    
                tempP[ii] = tempP[ii] + 1 #check how many sensors cover a particular partition
                newPartitionWeight[ii] = partitionsWeight[ii] 
    else:
        currSensors = copy.copy(setofSelectedSensors)
        currSensors.append(currSensor)
        for s in range(len(currSensors)):
            for ii in range(len(partitionsWeight)):
                if currSensors[s] in allPossibleSets[ii]:    
                    tempP[ii] = tempP[ii] + 1 #check how many sensors cover a particular partition
                    newPartitionWeight[ii] = partitionsWeight[ii]                    
                
    
    endTime = time.time()
    #print("--- %s seconds : Sens select 1: Compute new partitions' weights" % (endTime - startTime))    
    
    # Find indices of non-zero weights
    idxNonZeroWeight = np.nonzero(newPartitionWeight)
    
    # Remove the 0 weights entries from newPartitionWeight, as well as allPossibleSets
    newPartitionWeight = newPartitionWeight[idxNonZeroWeight]
    tempP = tempP[idxNonZeroWeight]    
    
    
    startTime = time.time()
    for ii in range(len(newPartitionWeight)):
        n = tempP[ii]
        if n!=0:
            tempAge = d + (1./(n+1.))*(1./ratePerSensor) 
            #tempAge = (n+2.)/(n+1.)*(1./ratePerSensor)
            if np.isnan(tempAge):
               AgePerPartition.append(0.)
            else:
               AgePerPartition.append(tempAge)
            coveredWeight.append(newPartitionWeight[ii])
        else:
            AgePerPartition.append(0.)
            coveredWeight.append(0.)
            
    totalCoveredWeight = np.sum(coveredWeight)        
    weightedAge = np.sum(np.array(coveredWeight)*np.array(AgePerPartition))        
    selectedPartitionsArea = copy.copy(newPartitionWeight)
    
    a = weightedAge + lam*(np.sum(partitionsWeight)-totalCoveredWeight )     
    a_empty = lam*np.sum(partitionsWeight)
    b = a_empty-a
    
    percentageCoveredWeight = totalCoveredWeight/np.sum(np.sum(weightedMap))*100.
    
    endTime = time.time()
    
    #print("--- %s seconds : Sens Select 1: Compute b" % (endTime - startTime))
    
    return b, percentageCoveredWeight , weightedAge/totalCoveredWeight , selectedPartitionsArea


def compute_b_2(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors ,ratePerSensor, currSensor, allPossibleSets, lam):
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
                
    # Find indices of non-zero weights
    idxNonZeroWeight = np.nonzero(newPartitionArea)
    
    # Remove the 0 weights entries from newPartitionWeight, as well as allPossibleSets
    newPartitionArea = newPartitionArea[idxNonZeroWeight]
    tempP = tempP[idxNonZeroWeight]   

                
    for ii in range(len(newPartitionArea)):
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
def SensSelecModel_1_StochGreedy(N, d, capacity, mu, weightedMap, partitionsWeight , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, k, thresh = 2.):
    #coverageArea = np.sum(partitionsWeight)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(0,N,1)
    
    #k = 5.
    #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    lam = d*(1.+2./3.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    
    eps = 0.1
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        
        #### Apply Stochastic Greedy ++ from : deVeciana, Abolfazl, Vikalo paper
        # Step 1: Determine how many sensors to sample from the set of available sensors 
        if ii < int(k - np.log(1./eps)):
            r_i = int((N-ii)/(k-ii)*np.log(1./eps))
        else:
            r_i = N-ii
        
        # Step 2: Randomly select r_i sensors
        idxOfRandomGeneratedSensors = []
        qq = 0
        print('start')
        while qq < (r_i):
            tempPickedIdx = np.random.randint(0,N) #Generate numbers within the range [0,N-ii]
            if tempPickedIdx not in idxOfRandomGeneratedSensors and setofSensors[tempPickedIdx] not in setofSelectedSensors: #Check that the newly generated number hasn't been already picked. If not, add it to the list AND that the newly picked sensor wasn't previously picked
                qq += 1
                idxOfRandomGeneratedSensors.append(tempPickedIdx)
     
        setOfRandomGeneratedSensors = setofSensors[idxOfRandomGeneratedSensors]
        print('end')
        
        for jj in range(len(setOfRandomGeneratedSensors)):
            if setOfRandomGeneratedSensors[jj] not in setofSelectedSensors:
                b_new, temp_percentageCoveredWeight , temp_weightedAge , selectedPartitionsArea = compute_b_1(N, d, mu, partitionsWeight, setofSelectedSensors, setofSensors, ratePerSensor, setOfRandomGeneratedSensors[jj], allPossibleSets,weightedMap, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = setOfRandomGeneratedSensors[jj]
                    coverageWeight = temp_percentageCoveredWeight
                    weightedAge = temp_weightedAge 
        setofSelectedSensors.append(selectedSensor)
    
    #setofSelectedSensors = setofSelectedSensors - np.ones(len(setofSelectedSensors))             
    #setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return setofSelectedSensors



def SensSelecModel_2_StochGreedy(N, d, capacity, mu, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, k, thresh = 2.):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(0,N,1)
    
    #k = 5.
    #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    lam = d*(1.+2./3.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    
    eps = 0.1
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        
        #### Apply Stochastic Greedy ++ from : deVeciana, Abolfazl, Vikalo paper
        # Step 1: Determine how many sensors to sample from the set of available sensors 
        if ii < int(k - np.log(1./eps))-1:
            r_i = int((N-ii)/(k-ii)*np.log(1./eps))
        else:
            r_i = N-ii
        
        # Step 2: Randomly select r_i sensors
        idxOfRandomGeneratedSensors = []
        for qq in range(r_i):
            tempPickedIdx = np.random.randint(0,N) #Generate numbers within the range [0,N-ii]
            if tempPickedIdx not in idxOfRandomGeneratedSensors: #Check that the newly generated number hasn't been already picked. If not, add it to the list
                idxOfRandomGeneratedSensors.append(tempPickedIdx)
     
        setOfRandomGeneratedSensors = setofSensors[idxOfRandomGeneratedSensors]
        
        for jj in range(len(setOfRandomGeneratedSensors)):
            if setOfRandomGeneratedSensors[jj] not in setofSelectedSensors:
                b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b_2(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, setOfRandomGeneratedSensors[jj], allPossibleSets, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = setOfRandomGeneratedSensors[jj]
                    coverageArea = tempcoverageArea
                    areaWeightedAge = tempareaWeightedAge
        setofSelectedSensors.append(selectedSensor)
    
    #setofSelectedSensors = setofSelectedSensors - np.ones(len(setofSelectedSensors))            
    #setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return setofSelectedSensors



def SensSelecModel_1_Greedy(N, d, capacity, mu, weightedMap, partitionsWeight , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, k, thresh = 2.):
    #coverageArea = np.sum(partitionsWeight)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(0,N,1)
    
    #k = 5.
    #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    lam = d*(1.+2./3.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        for jj in range(N):
            if jj not in setofSelectedSensors:
                b_new, temp_percentageCoveredWeight , temp_weightedAge , selectedPartitionsArea = compute_b_1(N, d, mu, partitionsWeight, setofSelectedSensors, setofSensors, ratePerSensor, jj, allPossibleSets,weightedMap, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = jj
                    coverageWeight = temp_percentageCoveredWeight
                    weightedAge = temp_weightedAge 
        setofSelectedSensors.append(selectedSensor)
    
    #setofSelectedSensors = setofSelectedSensors - np.ones(len(setofSelectedSensors))             
    #setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return setofSelectedSensors



def SensSelecModel_2_Greedy(N, d, capacity, mu, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, k, thresh = 2.):
    areaWeightedAge = 0.
    coverageArea = np.sum(partitionsArea)
    numSelectedSensors = N
    setofSelectedSensors = []
    setofSensors = np.arange(0,N,1)
    
    #k = 5.
    #np.ceil((rectangleLength/sensorRadius)*1.) - 5.
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    lam = d*(1.+2./3.*numSelectedSensors)
    
    new_max = 0.
    temp_b_old = 0.
    for ii in range(int(numSelectedSensors)):
        b_old = temp_b_old
        new_max = 0.
        for jj in range(N):
            if jj not in setofSelectedSensors:
                b_new, tempcoverageArea, tempareaWeightedAge, selectedPartitionsArea = compute_b_2(N, d, mu, partitionsArea, setofSelectedSensors, setofSensors, ratePerSensor, jj, allPossibleSets, lam)
                if np.abs(b_new - b_old) >= new_max:
                    new_max = (b_new - b_old)
                    temp_b_old = b_new
                    selectedSensor = jj
                    coverageArea = tempcoverageArea
                    areaWeightedAge = tempareaWeightedAge
        setofSelectedSensors.append(selectedSensor)
    
    #setofSelectedSensors = setofSelectedSensors - np.ones(len(setofSelectedSensors))            
    #setofSelectedSensors = np.sort(setofSelectedSensors)
    
    return setofSelectedSensors






def computeCoveredAreaOfinterest(selectedSensors,weightedMap,sensorRadius,pixelWidth,pixelLength,labeledPixelMatrix,coordPixels,coordSensors):
    #totalnumPixels = np.prod(np.shape(weightedMap))
    totalnumStricPositivePixels = np.count_nonzero(weightedMap>0)
    coveredPixels = 0
    
    for pixel_0 in range(np.shape(weightedMap)[0]):
        for pixel_1 in range(np.shape(weightedMap)[1]):
            if weightedMap[pixel_0][pixel_1] > 0:
                for sensor in range(len(selectedSensors)):
                    if pixelisInCircle(selectedSensors[sensor],sensorRadius,int(labeledPixelMatrix[pixel_0][pixel_1]),coordPixels,coordSensors):
                        coveredPixels += 1
                        break
    coverage = coveredPixels/totalnumStricPositivePixels*100
                    
    return coverage


def computeCoveredAreaofTypicalSensor(selectedSensors, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor, obstructedPixelsinBox, labeledPixelMatrix):
    numTotalPixels = np.prod(np.shape(pixelsPerBoxPerSensor))
    numCoveredPixels = numTotalPixels - len(obstructedPixelsinBox)
    
    numAddedPixels = 0
    for pixel in range(len(obstructedPixelsinBox)):
        for sensor in range(len(selectedSensors)):
            if pixelisInCircle(selectedSensors[sensor],sensorRadius,int(obstructedPixelsinBox[pixel]),coordPixels,coordSensors):
                numAddedPixels += 1
                break
            
    numTotalCoveredPixels = numCoveredPixels + numAddedPixels
    coverageOfTypicalSensor = numTotalCoveredPixels/numTotalPixels*100
    
    return coverageOfTypicalSensor


def computeCoveredAreaofObstructedRegionTypicalSensor(selectedSensors, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor, obstructedPixelsinBox, labeledPixelMatrix):
    numTotalPixels = np.prod(np.shape(pixelsPerBoxPerSensor))
    numCoveredPixels = 0
    
    for pixel in range(len(obstructedPixelsinBox)):
        for sensor in range(len(selectedSensors)):
            if pixelisInCircle(selectedSensors[sensor],sensorRadius,int(obstructedPixelsinBox[pixel]),coordPixels,coordSensors):
                numCoveredPixels += 1
                break
            
    coverageOfObstructedRegionofTypicalSensor = numCoveredPixels/numTotalPixels*100
    
    return coverageOfObstructedRegionofTypicalSensor


def computeWeightedAgeofTypicalSensor_1(selectedSensors, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor, obstructedPixelsinBox, labeledMatrixPixel, weightedMap, weightedRegionsPerSensor, lam, d, N, capacity, mu, k):
    #totalWeight = np.sum(weightedMap)
    totalWeight = 0.
    
    numSelectedSensors = N
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+2./3.*numSelectedSensors)
    #lam = d*(1.+2./3.*20)
    
    weightedAgeOfTypicalSensor = 0
    #tempCoveredWeights = []
    
    totalWeight = len(obstructedPixelsinBox)*weightedRegionsPerSensor
    
    for pixel in range(len(obstructedPixelsinBox)):
        obstructedPixel = obstructedPixelsinBox[pixel]
        index = np.where(labeledMatrixPixel == obstructedPixel)
        weight = weightedMap[index[0][0]][index[1][0]]
        #totalWeight += weightedRegionsPerSensor
        
        numSensors = 0
        for sensor in range(len(selectedSensors)):
            if pixelisInCircle(selectedSensors[sensor],sensorRadius,int(obstructedPixel),coordPixels,coordSensors):
                numSensors += 1
                
        if numSensors == 0:
            tempAge = 0    
        else:
            #tempCoveredWeights.append(weight)
            tempAge = d + (1./(numSensors+1.))*(1./ratePerSensor)
            totalWeight += weight
            
        weightedAgeOfTypicalSensor += tempAge*weightedRegionsPerSensor
        
#    if totalWeight == 0:
#        weightedAgeOfTypicalSensor = 0.
#    else:
#        weightedAgeOfTypicalSensor = weightedAgeOfTypicalSensor/totalWeight
   
    weightedAgeOfTypicalSensor = weightedAgeOfTypicalSensor/totalWeight
     
    return weightedAgeOfTypicalSensor

    
    
def computeWeightedAgeofTypicalSensor_AgeMin_1(selectedSensors, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor,obstructedPixelsinBox, labeledMatrixPixel, weightedMap, weightedRegionsPerSensor, lam, d, N, capacity, mu, ratesofselectedSensorsAgeMin, k):
# =============================================================================
#     numSelectedSensors = N
#     if int(N)>int(k):
#        numSelectedSensors = int(k) 
# =============================================================================
    
    #ratePerSensor = capacity/(numSelectedSensors*mu*d)
    #lam = d*(1.+2./3.*numSelectedSensors)
    
    totalWeight = 0.
    #lam = d*(1.+2./3.*20)
    
    weightedMinAgeOfTypicalSensor = 0
    #tempCoveredWeights = []

    totalWeight = len(obstructedPixelsinBox)*weightedRegionsPerSensor
    
    selectedSensors = np.sort(selectedSensors)
    
    for pixel in range(len(obstructedPixelsinBox)):
        obstructedPixel = obstructedPixelsinBox[pixel]
        index = np.where(labeledMatrixPixel == obstructedPixel)
        weight = weightedMap[index[0][0]][index[1][0]]
        
        #totalWeight += weightedRegionsPerSensor
        selectedRates = []
        numSensors = 0
        for sensor in range(len(selectedSensors)):
            if pixelisInCircle(selectedSensors[sensor],sensorRadius,int(obstructedPixel),coordPixels,coordSensors):
                selectedRates.append(ratesofselectedSensorsAgeMin[sensor])
                numSensors += 1
                
        if numSensors == 0:
            tempAge = 0
        else:
            r_max = max(selectedRates)
            result = quad(return_zj(selectedRates,d), d, d + 1./r_max)
            tempAge = d+result[0]
            totalWeight += weight
            #tempCoveredWeights.append(weight)
        
        weightedMinAgeOfTypicalSensor += tempAge*weightedRegionsPerSensor
        
#    if totalWeight == 0:
#        weightedMinAgeOfTypicalSensor = 0.
#    else:
#        weightedMinAgeOfTypicalSensor = weightedMinAgeOfTypicalSensor/totalWeight
    
    weightedMinAgeOfTypicalSensor = weightedMinAgeOfTypicalSensor/totalWeight
    
    return weightedMinAgeOfTypicalSensor



def computeNoCollabCoverage(pixelsPerBoxPerSensor, obstructedPixelsinBox):
    numTotalPixels = np.prod(np.shape(pixelsPerBoxPerSensor))
    numCoveredPixels = numTotalPixels - len(obstructedPixelsinBox)

    noCollabCoverage = numCoveredPixels/numTotalPixels*100

    return noCollabCoverage

def computeNoCollabAge(pixelsPerBoxPerSensor, obstructedPixelsinBox, weightedMap, weightedRegionsPerSensor, labeledPixelMatrix, lam, d, N, k, capacity, mu):
    totalWeight = 0
    
    numSelectedSensors = N
    if int(N)>int(k):
       numSelectedSensors = int(k) 
    
    ratePerSensor = capacity/(numSelectedSensors*mu*d)

    lam = d*(1.+2./3.*numSelectedSensors)
    
    noCollabWeightedAge = 0
    
    for pixel in range(len(obstructedPixelsinBox)):
        #obstructedPixel = obstructedPixelsinBox[pixel]
        #index = np.where(labeledPixelMatrix == obstructedPixel)
        #weight = weightedMap[index[0][0]][index[1][0]]
        noCollabWeightedAge += lam*weightedRegionsPerSensor
        totalWeight += weightedRegionsPerSensor
    
    noCollabWeightedAge = noCollabWeightedAge/totalWeight
    
    return noCollabWeightedAge
    
#@jit(target ="cuda")  
def AgeMinModel_1(N, d, capacity, mu, setofSelectedSensors, weightedMap, partitionsWeight , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, T, lam, k, thresh = 2.):
    numSelectedSensors = N    
    if int(N)>int(k):
       numSelectedSensors = int(k) 

    setofSelectedSensors = np.sort(setofSelectedSensors)
    
    newallPossibleSets = []
    
    for ii in range(1,int(numSelectedSensors)+1):
        hello = findsubsets(setofSelectedSensors,ii) 
        #hello1 = (np.asarray(hello))
        for jj in range(len(hello)):
            newallPossibleSets.append(list(hello[jj]))
    
    newselectedPartitionsWeight = np.zeros(2**(numSelectedSensors)-1)
    
    
    for ii in range(len(allPossibleSets)):
        temp = []
        for jj in range(len(allPossibleSets[ii])):
            if allPossibleSets[ii][jj] in setofSelectedSensors:
                temp.append(allPossibleSets[ii][jj])
        if temp:
            #temp = np.sort(temp)
            idx = newallPossibleSets.index(temp)
            newselectedPartitionsWeight[idx] += partitionsWeight[ii]    
            
    # Compute new rate allocation and new ageWeightedArea
    rate_fw_agemin, obj_fn, l1_fw_agemin = descent(N,frank_wolfe, d, numSelectedSensors, setofSelectedSensors, newallPossibleSets, np.array(newselectedPartitionsWeight), capacity/(mu*d), T=T)
    
    return rate_fw_agemin



def main(T=int(5e2)): 
    startTotalTime = time.time()
    scalingFactor = 1
    scale = 1
    N = np.arange(8,20) # number of sensors
    k = 8
    lam = 1.
    sensorRadius = np.array(50/scalingFactor)/scale#coverage radius per sensor
    #sensorRadius = []
    #sensorRadius = np.array([1.,1.,1.,1.,1.,2.,2.,2.,2.,2.])    
    capacity = 1.
    d = 4.2e-3 # transmission delay
    mu = 1. # packet size 
    
    plot = 0
    plot2 = 0
    
    rectangleLength = 500/scalingFactor#/scale
    rectangleWidth = 12/scalingFactor#/scale
    numLanes = 3
    boxDim = np.array([rectangleLength,rectangleWidth])
    areaR = rectangleLength*rectangleWidth*scalingFactor**2


    #### Region of interest of car  #####
    
    t_interest = 5 #seconds
    speed = 50./3 # 60 km/hour
    #speed = 19.44
    
    length_box_per_car = 2*t_interest*speed/scale
    #length_box_per_car = 300.
    width_box_per_car = rectangleWidth/scale
    carRoI = np.array([length_box_per_car,width_box_per_car])
    
    carLength = 4.48#/scale
    carWidth = 1.83#/scale
    carLengthScaled = carLength/scalingFactor
    carWidthScaled = carWidth/scalingFactor
    
    inter_car_dist = 10/scale
    carDimensions = np.array([carLength,carWidth])/scalingFactor # Length X Width

    
    numSquaresperLength = int(rectangleLength)
    numSquaresperWidth =  int(rectangleWidth)

        
    pixelLength = rectangleLength/numSquaresperLength
    pixelWidth = rectangleWidth/numSquaresperWidth
    
    xPosCenterPixel1 = pixelLength/2
    yPosCenterPixel1 = pixelWidth/2
    
    coordPixels = generatePixelsCenters(xPosCenterPixel1, yPosCenterPixel1, pixelLength, pixelWidth, numSquaresperLength, numSquaresperWidth)
    labeledPixels = np.arange(0,numSquaresperLength*numSquaresperWidth)
    labeledMatrixPixel = np.zeros((numSquaresperWidth,numSquaresperLength))
        
    countPixel = 0
    for ww in range(numSquaresperWidth):
        for ll in range(numSquaresperLength):
            labeledMatrixPixel[ww][ll] = countPixel
            countPixel += 1

    coverageAreaObstructions_1 = []
    coverageAreaObstructions_2 = []
    
    coverageAreaBaseline = []
    weightedAgeBaseline = []
    
    noCollabCoverageTypicalSensor = []
    noCollabWeightedAgeTypicalSensor = []
    
    coverageObstructedPixelsSensSelec_1 = []
    coverageTypicalSensorSensSelec_1 = []
    weightedAgeSensSelec_1 = []
    weightedAgeMinAge_1 = []
    
    coverageObstructedPixelsSensSelec_2 = []
    coverageTypicalSensorSensSelec_2 = []
    weightedAgeSensSelec_2 = []    
    
    
    
    #### STD for confidence intervals  ####
    stdcoverageAreaObstructions_1 = []
    stdcoverageAreaObstructions_2 = []
    
    stdcoverageAreaBaseline = []
    stdweightedAgeBaseline = []
    
    stdnoCollabCoverageTypicalSensor = []
    stdnoCollabWeightedAgeTypicalSensor = []
    
    stdcoverageObstructedPixelsSensSelec_1 = []
    stdcoverageTypicalSensorSensSelec_1 = []
    stdweightedAgeSensSelec_1 = []
    stdweightedAgeMinAge_1 = []
    
    stdcoverageObstructedPixelsSensSelec_2 = []
    stdcoverageTypicalSensorSensSelec_2 = []
    stdweightedAgeSensSelec_2 = []  


    numIter = 5
    
    for ii in tqdm(range(len(N))):
         
         temp1coverageAreaObstructions_1 = []
         temp1coverageAreaObstructions_2 = []
         
         temp1coverageAreaBaseline = []
         temp1WeightedAgeBaseline = []
         
         temp1NoCollabCoverage = []
         temp1NoCollabWeightedAge = []
         
         temp1coverageObstructedPixelsSensSelec_1 = []
         temp1coverageTypicalSensorSensSelec_1 = []
         temp1WeightedAgeSensSelec_1 = []
         
         temp1WeightedAgeMinAge_1 = []
         
         temp1coverageObstructedPixelsSensSelec_2 = []
         temp1coverageTypicalSensorSensSelec_2 = []         
         temp1WeightedAgeSensSelec_2 = []

         
         for jj in range(numIter):
             
             print('...start')
             #####  We check if the newly generated vehicle doesn't overlap with any previously generated vehicle  ############ 
             nn = 0
             coordSensors = []
             while nn < N[ii]:
                 check = 0
                 temp_x = np.random.rand(1,1)*(rectangleLength-0) 
                 temp_y = rectangleWidth/numLanes/2 + np.random.randint(0,numLanes-1,(1,1))*rectangleWidth/numLanes
                 #temp_y = np.random.rand(1,1)*(rectangleWidth-0)
                 
                 temp_newSensor = np.concatenate((temp_x,temp_y),axis=1)
                 # Check first that the new sensor's coordinates have not been previously selected
                 check1 = 0
                 for mm in range(len(coordSensors)):
                    if temp_newSensor[0][0] == coordSensors[mm][0] and temp_newSensor[0][1] == coordSensors[mm][1]:
                        check1 = 1
                        break
                        # Check that the dist between the new sensor and other sensors is at least: sqrt(L^2+W^2)
                 if check1 == 0: 
                    if not list(coordSensors):
                        coordSensors = np.concatenate((temp_x,temp_y),axis=1)
                        nn += 1
                        coordSensors = list(coordSensors)
                    else:
                        for mm in range(len(coordSensors)):
                            if temp_newSensor[0][1] == coordSensors[mm][1]:
                                if np.linalg.norm(temp_newSensor[0][0] - coordSensors[mm][0]) < inter_car_dist: #np.sqrt((carLengthScaled)**2+(carWidthScaled)**2):
                                    check = 1
                                    break                        
                        if check == 0:
                            coordSensors.append(np.concatenate((temp_x,temp_y),axis=1)[0])   
                            nn += 1
                 
             
             print('...end')
             
#             xcoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleLength-0) 
#             ycoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleWidth-0)
#             coordSensors = np.concatenate((xcoordSensors,ycoordSensors),axis=1)             
             
             #######################################################################################################################
             
             #########################     PER SENSOR REGION OF INTEREST      ##############################
             
             startTime = time.time()
             # Step 1: Find 4 corners of box of region of interest per car
             regionOfInterestPerSensor_1 = findCoordregionOfInterestPerSensor_1(coordPixels,coordSensors,N[ii],length_box_per_car,width_box_per_car,boxDim)
             
             # Step 2: Find pixels in the region of interest
             pixelsPerBoxPerSensor_1 = findPixelsinRegionOfInterest_1(N[ii],coordPixels,coordSensors,length_box_per_car,width_box_per_car,labeledPixels)
             
             # Step 3: Find obstructed pixels and non-observed pixels in region of interest
             obstructedPixelsinBox_1 = findobstructedPixelsinBox_1(N[ii], pixelsPerBoxPerSensor_1, coordSensors, coordPixels, sensorRadius, carDimensions, carRoI, boxDim, plot)
             
             # Step 4: Sort the obstructed pixels per sensor into different regions
             sortedObstructedPixelsperSensorinBox_1 = sortObstructedPixelsperSensorinBox_1(N[ii], obstructedPixelsinBox_1, labeledMatrixPixel,labeledPixels,regionOfInterestPerSensor_1, numSquaresperLength, numSquaresperWidth)
             
             # Step 5: Put weight on pixels depending on obstructed region area 
             weightedRegionsPerSensor_1 = putWeightonRegions(sortedObstructedPixelsperSensorinBox_1,N[ii])
             
             # Step 6: Weight the pixels by summing the weights we got from the previous function
             weightedMap_1 = weightPixels(labeledMatrixPixel,weightedRegionsPerSensor_1,sortedObstructedPixelsperSensorinBox_1,N[ii])
             
             endTime = time.time()

             print("--- %s seconds : Per sensor region of interest" % (endTime - startTime))

            ################################################################################################
            
             #########################    REGION OF INTEREST IS THE WHOLE BOX  ############################ 
             startTime = time.time()
             # Step 1: Find the labels of the obstructed pixels per sensor, for all available sensors in the network   
             obstructedLabeledPixelsperSensor_2 = findObstructions(coordPixels, coordSensors, sensorRadius, labeledPixels, N[ii], carDimensions, boxDim, plot)
                
             endTime = time.time()
             print("--- %s seconds : Obstructions" % (endTime - startTime))
             # Step 2: Sort the obstructed pixels per sensor into different regions
             #sortedObstructedPixelsperSensor_2 = sortObstructedPixelPerSensor(labeledMatrixPixel,labeledPixels,obstructedLabeledPixelsperSensor_2,numSquaresperLength,numSquaresperWidth,N[ii])
                
             # Step 3: Put weight on pixels depending on obstructed region area
             #weightedRegionsPerSensor_2 = putWeightonRegions(sortedObstructedPixelsperSensor_2,N[ii])
             
             #Step 4: Weight the pixels by summing the weights we got from the previous function
             #weightedMap_2 = weightPixels(labeledMatrixPixel, weightedRegionsPerSensor_2, sortedObstructedPixelsperSensor_2, N[ii])
             
             # Plot the heat map
             if plot == 1:
                 plt.figure()
                 ax = sns.heatmap(weightedMap_1)
                 plt.savefig(os.path.join(path, 'heat-Weighted-Map.pdf'))
            
            ################################################################################################
            
            #########   Compare both approaches  #################################################
            
             # Step: Divide the map into regions based on different weights
             
             # Technique 1: Region of interest of each vehicle
             #regionID_1, IDmap_1 = divideMapintoRegions(weightedMap_1, labeledMatrixPixel, labeledPixels, numSquaresperLength, numSquaresperWidth)
             
             # Technique 2: Region of interest of each vehicle
             #regionID_2, IDmap_2 = divideMapintoRegions(weightedMap_2, labeledMatrixPixel, labeledPixels, numSquaresperLength, numSquaresperWidth)
                        
            # Step : Compute the different partitions areas
             #partitionsArea , allPossibleSets = findPartitionsAreas(pixelLength, pixelWidth, coordPixels, coordSensors, sensorRadius, labeledPixels, labeledMatrixPixel, N[ii], carDimensions, boxDim, obstructedLabeledPixelsperSensor)
             
             startTime = time.time()
             partitionsWeights_1 , allPossibleSets_1 = findPartitionsWeightsFaster(pixelLength, pixelWidth, coordPixels, coordSensors, sensorRadius, labeledPixels, labeledMatrixPixel, N[ii], carDimensions, boxDim, obstructedPixelsinBox_1, weightedMap_1)
#             partitionsWeights_2 , allPossibleSets_2 = findPartitionsWeights(pixelLength, pixelWidth, coordPixels, coordSensors, sensorRadius, labeledPixels, labeledMatrixPixel, N[ii], carDimensions, boxDim, obstructedLabeledPixelsperSensor_2, regionID_1, weightedMap_1, IDmap_1)
             endTime = time.time()
             print("--- %s seconds : Partitions Weight" % (endTime - startTime))
             
             startTime = time.time()
             partitionsArea_2 , allPossibleSets_2 = findPartitionsAreasFaster(pixelLength, pixelWidth, coordPixels, coordSensors, sensorRadius, labeledPixels, labeledMatrixPixel, N[ii], carDimensions, boxDim, obstructedLabeledPixelsperSensor_2)
             endTime = time.time()
             print("--- %s seconds : Partitions Area" % (endTime - startTime))
             ##########################   Baseline   ##############################
             
             #tempcoverageAreaBaseline , tempareaWeightedAgeBaseline = baselineModel(capacity/(N[ii]*mu*d), d, partitionsArea_2 , allPossibleSets_2, scalingFactor)
             
             ##########################  Sensor Selection  ########################
             # Step 1 - Find the selected sensors for each technique
             
             # Technique 1: Region of interest
             # a - No age minimization
             startTime = time.time()             
             tempselectedSensorsSensSelec_1 = SensSelecModel_1_StochGreedy(N[ii], d, capacity , mu, weightedMap_1, partitionsWeights_1 , allPossibleSets_1, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor,lam, k,thresh = 2.)
             endTime = time.time()
             print("--- %s seconds : Sens Select 1" % (endTime - startTime))
             
             startTime = time.time()
             tempRatesofselectedSensorsAgeMin_1 = AgeMinModel_1(N[ii], d, capacity, mu, tempselectedSensorsSensSelec_1, weightedMap_1, partitionsWeights_1 , allPossibleSets_1, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, T, lam, k, thresh = 2.)
             endTime = time.time()
             print("--- %s seconds : Age minimization" % (endTime - startTime))
             
             # Technique 2: All box
             startTime = time.time()
             tempselectedSensorsSensSelec_2 = SensSelecModel_2_StochGreedy(N[ii], d, capacity , mu, partitionsArea_2*scalingFactor**2 , allPossibleSets_2, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor,lam, k, thresh = 2.)
             endTime = time.time()
             print("--- %s seconds : Sens Select 2" % (endTime - startTime))
             # Step 2 - Compute the covered area of the region of interest
             
# =============================================================================
#              # Technique 1: Region of interest
#              #coverageObstrucedPixels_1 = computeCoveredAreaOfinterest(tempselectedSensorsSensSelec_1,weightedMap_1,sensorRadius,pixelWidth,pixelLength,labeledMatrixPixel,coordPixels,coordSensors)
#              
#              # Technique 2: Whole box
#              #coverageObstrucedPixels_2 = computeCoveredAreaOfinterest(tempselectedSensorsSensSelec_2,weightedMap_1,sensorRadius,pixelWidth,pixelLength,labeledMatrixPixel,coordPixels,coordSensors)
#              
# =============================================================================
             # Step 3 - Compute a typical sensor's coverage
            

             temp2coverageObstrucedPixels_1 = []
             temp2coverageObstrucedPixels_2 = []
             temp2noCollabCoverage = []
             temp2noCollabWeightedAge  = []
             temp2coverageTypicalSensor_Baseline = []
             temp2weightedAgeTypicalSensor_Baseline = []
             temp2coverageTypicalSensor_1 = []
             temp2weightedAgeTypicalSensor_1 = []
             temp2weightedAgeTypicalSensorAgeMin_1 = []
             temp2coverageTypicalSensor_2 = []
             temp2weightedAgeTypicalSensor_2 = []
             
             startTime = time.time()
             
             num = int(N[ii])
             for mm in range(num):
                #Compute the covered area of the region of interest
                # Technique 1: Region of interest
                coverageObstrucedPixels_1 = computeCoveredAreaofObstructedRegionTypicalSensor(tempselectedSensorsSensSelec_1, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel)
                
                # Technique 2: Whole box 
                coverageObstrucedPixels_2 = computeCoveredAreaofObstructedRegionTypicalSensor(tempselectedSensorsSensSelec_2, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel)
                
                # No collaboration
                noCollabCoverage = computeNoCollabCoverage(pixelsPerBoxPerSensor_1[mm], obstructedPixelsinBox_1[mm])
                #noCollabWeightedAge = computeNoCollabAge(pixelsPerBoxPerSensor_1[mm], obstructedPixelsinBox_1[mm], weightedMap_1, weightedRegionsPerSensor_1[mm], labeledMatrixPixel, lam, d, N[ii], k, capacity, mu)
            
                # Sensor Selection
                # Baseline: Select all sensors
                coverageTypicalSensor_Baseline = computeCoveredAreaofTypicalSensor(np.arange(0,N[ii],1), sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm], obstructedPixelsinBox_1[mm], labeledMatrixPixel)
                weightedAgeTypicalSensor_Baseline = computeWeightedAgeofTypicalSensor_1(np.arange(0,N[ii],1), sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel, weightedMap_1, weightedRegionsPerSensor_1[mm], lam, d, N[ii], capacity, mu, k=N[ii])
             
                # Technique 1: Region of Interest
                # a - No age minimization
                coverageTypicalSensor_1 = computeCoveredAreaofTypicalSensor(tempselectedSensorsSensSelec_1, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel)
                weightedAgeTypicalSensor_1 = computeWeightedAgeofTypicalSensor_1(tempselectedSensorsSensSelec_1, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel, weightedMap_1, weightedRegionsPerSensor_1[mm], lam, d, N[ii], capacity, mu, k)
             
                # b - With age minimization
                weightedAgeTypicalSensorAgeMin_1 = computeWeightedAgeofTypicalSensor_AgeMin_1(tempselectedSensorsSensSelec_1, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel, weightedMap_1, weightedRegionsPerSensor_1[mm], lam, d, N[ii], capacity, mu, tempRatesofselectedSensorsAgeMin_1, k)
             
                # Technique 2: All box
                coverageTypicalSensor_2 = computeCoveredAreaofTypicalSensor(tempselectedSensorsSensSelec_2, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel)
                weightedAgeTypicalSensor_2 = computeWeightedAgeofTypicalSensor_1(tempselectedSensorsSensSelec_2, sensorRadius, coordSensors, coordPixels, pixelsPerBoxPerSensor_1[mm],obstructedPixelsinBox_1[mm], labeledMatrixPixel, weightedMap_1, weightedRegionsPerSensor_1[mm], lam, d, N[ii], capacity, mu, k)
             
                
                temp2coverageObstrucedPixels_1.append(coverageObstrucedPixels_1)
                temp2coverageObstrucedPixels_2.append(coverageObstrucedPixels_2)
                temp2noCollabCoverage.append(noCollabCoverage)
#                temp2noCollabWeightedAge.append(noCollabWeightedAge)
                temp2coverageTypicalSensor_Baseline.append(coverageTypicalSensor_Baseline)
                temp2weightedAgeTypicalSensor_Baseline.append(weightedAgeTypicalSensor_Baseline)
                temp2coverageTypicalSensor_1.append(coverageTypicalSensor_1)
                temp2weightedAgeTypicalSensor_1.append(weightedAgeTypicalSensor_1) 
                temp2weightedAgeTypicalSensorAgeMin_1.append(weightedAgeTypicalSensorAgeMin_1)
                temp2coverageTypicalSensor_2.append(coverageTypicalSensor_2)
                temp2weightedAgeTypicalSensor_2.append(weightedAgeTypicalSensor_2)                
            
             coverageObstrucedPixels_1 = np.sum(temp2coverageObstrucedPixels_1)/num
             coverageObstrucedPixels_2 = np.sum(temp2coverageObstrucedPixels_2)/num
             noCollabCoverage = np.sum(temp2noCollabCoverage)/num
             noCollabWeightedAge = np.sum(temp2noCollabWeightedAge)/num
             coverageTypicalSensor_Baseline = np.sum(temp2coverageTypicalSensor_Baseline)/num
             weightedAgeTypicalSensor_Baseline = np.sum(temp2weightedAgeTypicalSensor_Baseline)/num
             coverageTypicalSensor_1 = np.sum(temp2coverageTypicalSensor_1)/num
             weightedAgeTypicalSensor_1  = np.sum(temp2weightedAgeTypicalSensor_1)/num
             weightedAgeTypicalSensorAgeMin_1 = np.sum(temp2weightedAgeTypicalSensorAgeMin_1)/num
             coverageTypicalSensor_2 = np.sum(temp2coverageTypicalSensor_2)/num
             weightedAgeTypicalSensor_2 = np.sum(temp2weightedAgeTypicalSensor_2)/num
             
             
             endTime = time.time()
             print("--- %s seconds : Age-Cov computations" % (endTime - startTime))
             
             # 3 - Compute the age of the regions of interest

             #tempcoverageAreaAgeMin , tempareaWeightedAgeAgeMin , tempselectedSensorsAgeMin = AgeMinModel(N[ii], d, capacity, mu, weightedMap, partitionsWeights , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, T, lam, k, thresh = 2.)
             
#             if plot2 == 1:
#                 numSensorsPerPixelMap = findSensorsPerPixel(tempselectedSensorsSensSelec, labeledMatrixPixel, sensorRadius, coordPixels, coordSensors)
#                 plt.figure()
#                 ax = sns.heatmap(numSensorsPerPixelMap)
#                 plt.savefig(os.path.join(path, 'heat-Sensors-Map.pdf'))


             # Obstruction coverage
             temp1coverageAreaObstructions_1.append(coverageObstrucedPixels_1)
             temp1coverageAreaObstructions_2.append(coverageObstrucedPixels_2)
             
             # Baseline
             temp1coverageAreaBaseline.append(coverageTypicalSensor_Baseline)
             temp1WeightedAgeBaseline.append(weightedAgeTypicalSensor_Baseline)
             
             # Naive Approach: No collaboration
             temp1NoCollabCoverage.append(noCollabCoverage)
             temp1NoCollabWeightedAge.append(noCollabWeightedAge)
             
             # Technique 1: Region of Interest
             temp1coverageObstructedPixelsSensSelec_1.append(coverageObstrucedPixels_1)
             temp1coverageTypicalSensorSensSelec_1.append(coverageTypicalSensor_1)
             temp1WeightedAgeSensSelec_1.append(weightedAgeTypicalSensor_1)
             
             temp1WeightedAgeMinAge_1.append(weightedAgeTypicalSensorAgeMin_1)
             
#             temp1WeightedAgeSensSelec.append(tempWeightedAgeSensSelec)
#             temp1selectedSensorsSensSelec.append(len(tempselectedSensorsSensSelec))
             
             
             # Technique 2: All box
             temp1coverageObstructedPixelsSensSelec_2.append(coverageObstrucedPixels_2)
             temp1coverageTypicalSensorSensSelec_2.append(coverageTypicalSensor_2)
             temp1WeightedAgeSensSelec_2.append(weightedAgeTypicalSensor_2)
        
#             temp1coverageSensSelec1.append(tempcoverageSensSelec1)
#             temp1WeightedAgeSensSelec1.append(tempWeightedAgeSensSelec1)
#             temp1selectedSensorsSensSelec1.append(len(tempselectedSensorsSensSelec1))         
     
#             temp1coverageAreaAgeMin.append(tempcoverageAreaAgeMin)
#             temp1areaWeightedAgeAgeMin.append(tempareaWeightedAgeAgeMin)
#             temp1selectedSensorsAgeMin.append(len(tempselectedSensorsAgeMin))
     
         coverageAreaObstructions_1.append(np.sum(temp1coverageAreaObstructions_1)/numIter)
         coverageAreaObstructions_2.append(np.sum(temp1coverageAreaObstructions_2)/numIter)  
         
         ##
         stdcoverageAreaObstructions_1.append(np.std(temp1coverageAreaObstructions_1))
         stdcoverageAreaObstructions_2.append(np.std(temp1coverageAreaObstructions_2))
         ##
         
         coverageAreaBaseline.append(np.sum(temp1coverageAreaBaseline)/numIter)
         weightedAgeBaseline.append(np.sum(temp1WeightedAgeBaseline)/numIter*1000.)
         
         ##
         stdcoverageAreaBaseline.append(np.std(temp1coverageAreaBaseline))
         stdweightedAgeBaseline.append(np.std(temp1WeightedAgeBaseline)*1000.)
         ##
         
         noCollabCoverageTypicalSensor.append(np.sum(temp1NoCollabCoverage)/numIter)
         noCollabWeightedAgeTypicalSensor.append(np.sum(temp1NoCollabWeightedAge)/numIter*1000.)
         
         ##
         stdnoCollabCoverageTypicalSensor.append(np.std(temp1NoCollabCoverage))
         stdnoCollabWeightedAgeTypicalSensor.append(np.std(temp1NoCollabWeightedAge)*1000.)         
         ##
         
         
         coverageObstructedPixelsSensSelec_1.append(np.sum(temp1coverageObstructedPixelsSensSelec_1)/numIter)
         coverageTypicalSensorSensSelec_1.append(np.sum(temp1coverageTypicalSensorSensSelec_1)/numIter)
         weightedAgeSensSelec_1.append(np.sum(temp1WeightedAgeSensSelec_1)/numIter*1000.)
         #weightedAgeSensSelec.append(np.sum(temp1WeightedAgeSensSelec)/numIter*1000.)
         #selectedSensorsSensSelec.append(np.sum(temp1selectedSensorsSensSelec)/numIter)
         
         weightedAgeMinAge_1.append(np.sum(temp1WeightedAgeMinAge_1)/numIter*1000.)
         
         ##
         stdcoverageObstructedPixelsSensSelec_1.append(np.std(temp1coverageObstructedPixelsSensSelec_1))
         stdcoverageTypicalSensorSensSelec_1.append(np.std(temp1coverageTypicalSensorSensSelec_1))
         stdweightedAgeSensSelec_1.append(np.std(temp1WeightedAgeSensSelec_1)*1000.)
         #weightedAgeSensSelec.append(np.sum(temp1WeightedAgeSensSelec)/numIter*1000.)
         #selectedSensorsSensSelec.append(np.sum(temp1selectedSensorsSensSelec)/numIter)
         
         stdweightedAgeMinAge_1.append(np.std(temp1WeightedAgeMinAge_1)*1000.)
         ##
         
         
         coverageObstructedPixelsSensSelec_2.append(np.sum(temp1coverageObstructedPixelsSensSelec_2)/numIter)
         coverageTypicalSensorSensSelec_2.append(np.sum(temp1coverageTypicalSensorSensSelec_2)/numIter)
         weightedAgeSensSelec_2.append(np.sum(temp1WeightedAgeSensSelec_2)/numIter*1000.)
#         coverageSensSelec1.append(np.sum(temp1coverageSensSelec1)/numIter)
#         weightedAgeSensSelec1.append(np.sum(temp1WeightedAgeSensSelec1)/numIter*1000.)
#         selectedSensorsSensSelec1.append(np.sum(temp1selectedSensorsSensSelec1)/numIter)         
     
#         coverageAreaAgeMin.append(np.sum(temp1coverageAreaAgeMin)/numIter)
#         areaWeightedAgeAgeMin.append(np.sum(temp1areaWeightedAgeAgeMin)/numIter*1000.)
#         selectedSensorsAgeMin.append(np.sum(temp1selectedSensorsAgeMin)/numIter)        
    
          ##
         stdcoverageObstructedPixelsSensSelec_2.append(np.std(temp1coverageObstructedPixelsSensSelec_2))
         stdcoverageTypicalSensorSensSelec_2.append(np.std(temp1coverageTypicalSensorSensSelec_2))
         stdweightedAgeSensSelec_2.append(np.std(temp1WeightedAgeSensSelec_2)*1000.)          
          ##
    
    
    endTotalTime = time.time()
    print("--- %s seconds : Total time" %(endTotalTime - startTotalTime))
    #################  Plots  ######################################
  
    #####################         COVERAGE OF OBSTRUCTIONS  ##########################
    plt.clf()
    #plt.plot(N , coverageAreaBaseline, '--', label='Baseline')
    plt.plot(N , coverageAreaObstructions_1, '.-',label='Aggregate regional interest - Sensor Selection')
    plt.plot(N , coverageAreaObstructions_2, '^-',label='Spatially uniform interest - Sensor Selection')
    #plt.plot(N , coverageSensSelec1, '.',label='New Sensor Selection')
    #plt.plot(N , coverageAreaAgeMin, label='Age Minimization')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.xlabel('Number of available sensors N', fontsize=12)
    plt.ylabel('Coverage of Regions of Interest [%]', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()
    plt.xlim(min(N),max(N),1)
    
    sio.savemat('coverageAreaObstructions_1.mat', {'coverageAreaObstructions_1':coverageAreaObstructions_1})
    sio.savemat('coverageAreaObstructions_2.mat', {'coverageAreaObstructions_2':coverageAreaObstructions_2})

    sio.savemat('stdcoverageAreaObstructions_1.mat', {'stdcoverageAreaObstructions_1':stdcoverageAreaObstructions_1})
    sio.savemat('stdcoverageAreaObstructions_2.mat', {'stdcoverageAreaObstructions_2':stdcoverageAreaObstructions_2})
    
    plt.savefig(os.path.join(path,'Coverage Obstructions' + '_N=' + str(min(N)) +'_'+ str(max(N)) + '_' + 'lam=' + 'lam_min' + '_obstructions_' + '.eps'),dpi=300, transparent=True)
    plt.savefig(os.path.join(path,'Coverage Obstructions' + '_N=' + str(min(N)) +'_'+ str(max(N)) + '_' + 'lam=' + 'lam_min' + '_obstructions_' + '.pdf'),dpi=300, transparent=True)

    ################################################################################

    #####################         COVERAGE OF TYPICAL SENSOR  ##########################
    plt.clf()
    plt.plot(N , coverageAreaBaseline, '>-', label='Baseline')
    plt.plot(N , noCollabCoverageTypicalSensor, '.-',label='No collaboration')
    plt.plot(N , coverageTypicalSensorSensSelec_1, '*-',label='Aggregate regional interest - Sensor Selection')
    plt.plot(N , coverageTypicalSensorSensSelec_2, '^-',label='Spatially uniform interest - Sensor Selection')
    #plt.plot(N , coverageSensSelec1, '.',label='New Sensor Selection')
    #plt.plot(N , coverageAreaAgeMin, label='Age Minimization')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.xlabel('Number of available sensors N', fontsize=12)
    plt.ylabel('Coverage of typical sensor [%]', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()
    plt.xlim(min(N),max(N),1)
    
    sio.savemat('coverageAreaBaseline.mat', {'coverageAreaBaseline':coverageAreaBaseline})
    sio.savemat('noCollabCoverageTypicalSensor.mat', {'noCollabCoverageTypicalSensor':noCollabCoverageTypicalSensor})
    sio.savemat('coverageTypicalSensorSensSelec_1.mat', {'coverageTypicalSensorSensSelec_1':coverageTypicalSensorSensSelec_1})
    sio.savemat('coverageTypicalSensorSensSelec_2.mat', {'coverageTypicalSensorSensSelec_2':coverageTypicalSensorSensSelec_2})

    sio.savemat('stdcoverageAreaBaseline.mat', {'stdcoverageAreaBaseline':stdcoverageAreaBaseline})
    sio.savemat('stdnoCollabCoverageTypicalSensor.mat', {'stdnoCollabCoverageTypicalSensor':stdnoCollabCoverageTypicalSensor})
    sio.savemat('stdcoverageTypicalSensorSensSelec_1.mat', {'stdcoverageTypicalSensorSensSelec_1':stdcoverageTypicalSensorSensSelec_1})
    sio.savemat('stdcoverageTypicalSensorSensSelec_2.mat', {'stdcoverageTypicalSensorSensSelec_2':stdcoverageTypicalSensorSensSelec_2})

    
    plt.savefig(os.path.join(path,'Coverage Typical Sensor' + '_N=' + str(min(N)) +'_'+ str(max(N)) + '_' + 'lam=' + 'lam_min' + '_obstructions_' + '.eps'),dpi=300, transparent=True)
    plt.savefig(os.path.join(path,'Coverage Typical Sensor' + '_N=' + str(min(N)) +'_'+ str(max(N)) + '_' + 'lam=' + 'lam_min' + '_obstructions_' + '.pdf'),dpi=300, transparent=True)

    ################################################################################
    #####################         Age OF RoI of TYPICAL SENSOR  ##########################
    plt.clf()
    plt.plot(N , weightedAgeBaseline, '>-', label='Baseline')
    #plt.plot(N , noCollabWeightedAgeTypicalSensor, '.-',label='No collaboration')
    plt.plot(N , weightedAgeSensSelec_1, '.-',label='Aggregate regional interest - Sensor Selection')
    plt.plot(N , weightedAgeMinAge_1, '*-',label='Aggregate regional interest - Age Minimization')
    plt.plot(N , weightedAgeSensSelec_2, '^-',label='Spatially uniform interest - Sensor Selection')
    #plt.plot(N , coverageSensSelec1, '.',label='New Sensor Selection')
    #plt.plot(N , coverageAreaAgeMin, label='Age Minimization')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.xlabel('Number of available sensors N', fontsize=12)
    plt.ylabel('Normalized weighted average age [msec]', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()
    plt.xlim(min(N),max(N),1)
    
    sio.savemat('weightedAgeBaseline.mat', {'weightedAgeBaseline':weightedAgeBaseline})
    sio.savemat('weightedAgeSensSelec_1.mat', {'weightedAgeSensSelec_1':weightedAgeSensSelec_1})
    sio.savemat('weightedAgeMinAge_1.mat', {'weightedAgeMinAge_1':weightedAgeMinAge_1})
    sio.savemat('weightedAgeSensSelec_2.mat', {'weightedAgeSensSelec_2':weightedAgeSensSelec_2})

    sio.savemat('stdweightedAgeBaseline.mat', {'stdweightedAgeBaseline':stdweightedAgeBaseline})
    sio.savemat('stdweightedAgeSensSelec_1.mat', {'stdweightedAgeSensSelec_1':stdweightedAgeSensSelec_1})
    sio.savemat('stdweightedAgeMinAge_1.mat', {'stdweightedAgeMinAge_1':stdweightedAgeMinAge_1})
    sio.savemat('stdweightedAgeSensSelec_2.mat', {'stdweightedAgeSensSelec_2':stdweightedAgeSensSelec_2})
    
    plt.savefig(os.path.join(path,'Age Typical Sensor' + '_N=' + str(min(N)) +'_'+ str(max(N)) + '_' + 'lam=' + 'lam_min' + '_obstructions_' + '.eps'),dpi=300, transparent=True)
    plt.savefig(os.path.join(path,'Age Typical Sensor' + '_N=' + str(min(N)) +'_'+ str(max(N)) + '_' + 'lam=' + 'lam_min' + '_obstructions_' + '.pdf'),dpi=300, transparent=True)

    ################################################################################
    
    
    
if __name__ == "__main__":
    main()