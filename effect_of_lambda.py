import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import itertools 
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.optimize import brentq


# Function that computes the age of the region of intersection 

###########  Initial Function
def objective_function(rate, n, P_i, V_i_compl):
    return (3/2*P_i[0]*V_i_compl[0]*1/rate[0] + 3/2*P_i[1]*V_i_compl[1]*1/rate[1] + 3/2*P_i[2]*V_i_compl[2]*1/rate[2] +\
             P_i[3]*V_i_compl[3]*age_of_intersection(rate[0:2]))/(np.sum(P_i))
                          
            
def gradient_obj_fn(rate, n, P_i, V_i_compl):
    grad = np.zeros(n)
    grad_0, grad_1 = grad_age_of_intersection(rate[0:2])
    
    grad[0] = (-3/2*P_i[0]*V_i_compl[0]*(1/rate[0])**2 + P_i[3]*V_i_compl[3]*(grad_0))/(np.sum(P_i))
    grad[1] = (-3/2*P_i[1]*V_i_compl[1]*(1/rate[1])**2 + P_i[3]*V_i_compl[3]*(grad_1))/(np.sum(P_i))
    grad[2] = (-3/2*P_i[2]*V_i_compl[2]*(1/rate[2])**2)/(np.sum(P_i)) 
    return np.array(grad)

def grad_age_of_intersection(rate):
    rate_new = rate.copy()
    rate_new[::-1].sort()
    
    # Check for condition
    min_rate_condition = check_rate_condition(rate_new)
    
    if (min_rate_condition == 1):
        if (int(rate[0]) > int(rate[1])):
            grad_0 = 8/3*rate[1]/(rate[0])**3 + 1/(6*rate[1]**2) - 3.5/(rate[0])**2
            grad_1 = -4/(3*rate[0]**2) -1/3*rate[0]/(rate[1])**3 + 1/(rate[1])**2 
        elif (int(rate[0]) < int(rate[1])):
            grad_0 = -4/(3*rate[1]**2) -1/3*rate[1]/(rate[0])**3 + 1/(rate[0])**2 
            grad_1 = 8/3*rate[0]/(rate[1])**3 + 1/(6*rate[0]**2) - 3.5/(rate[1])**2
        elif (int(rate[0]) == int(rate[1])):
            grad_0 = -4/(3*rate[0]**2)
            grad_1 = -4/(3*rate[1]**2)
            
    elif (min_rate_condition == 0):
        if (int(rate[0]) > int(rate[1])):
            grad_0 = -3/2*(1/rate[0])**2
            grad_1 = 0
        elif(int(rate[0]) < int(rate[1])):
            grad_0 = 0
            grad_1 = -3/2*(1/rate[1])**2
            
    return grad_0, grad_1

def check_rate_condition(rate):
    boolean = 0
    rate_new = rate.copy()
    rate_new[::-1].sort()    
    
    if (np.ceil(rate_new[1]) >= np.ceil(rate_new[0])/2):
        boolean = 1
        
    return boolean

def age_of_intersection(rate):
    #rate_old = rate
    rate_new = rate.copy()
    rate_new[::-1].sort()
    
    # Check for condition
    min_rate_condition = check_rate_condition(rate_new)
    integral = 0.
    temp = quad(lambda y: -rate_new[0]*y+2, 1/rate_new[0], 2/rate_new[0])[0]    
    
    if (min_rate_condition == 1):
        temp1 = lambda y: np.prod(-y*rate_new + np.array([2,1]))
        integral = quad(temp1, 1/rate_new[1], 2/rate_new[0])[0]
    elif (min_rate_condition== 0):
        integral = 0    
# =============================================================================
#     for i in range(rate.shape[0]):
#         if i == 0:
#             temp = quad(lambda y: -rate[0]*y+2, 1/rate[0], 2/rate[0]) #1st integral
#         else:
#             temp1 = lambda y: 1.   
#             temp2 = lambda y: -rate[i]*y+1
#             for j in range(i-1):
#                 temp1 = temp1*(lambda y: -rate[j]*y+2)
#             integral = integral + quad(temp2*temp1, 1/rate[i], 2/rate[0])[0]    
# =============================================================================
        
    return 1/rate_new[0] + temp + integral



#######################################################################################
    
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
                       
# =============================================================================
#     for s in range(numSelectedSensors):
#         if ratePerSensor[s] == max_rate:
#             temp1 = comp1(np.delete(ratePerSensor,s),d,ratePerSensor[s])   
#             result = quad(temp1, d, d+1./max_rate)
#             temp2 = comp2(ratePerSensor,d,max_rate)
#             grad_MinAge.append(result[0]+temp2)
#         else:
#             temp1 = comp1(np.delete(ratePerSensor,s),d,ratePerSensor[s])   
#             result = quad(temp1, d, d+1./max_rate)
#             grad_MinAge.append(result[0])
# =============================================================================
# =============================================================================
#     grad_MinAge = np.ma.masked_equal(grad_MinAge,0)
#     grad_MinAge = grad_MinAge.compressed()
# =============================================================================
       
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

# =============================================================================
# 
# def proj_gradient_descent(rate, n, P_i, V_i_compl, t, capacity, alpha, beta, c=1.):
#     # update x (your code here), set c above
#     grad = gradient_obj_fn(rate, n, P_i, V_i_compl)    
#     eta_t = c/(np.sqrt(t+1.))
#     rate = rate - eta_t*grad
#     
#     #projection step
#     lambda_opt = (sum(rate)-capacity)/rate.shape[0]
#     rate = np.maximum(rate-lambda_opt,1e-3)
#     return rate
# =============================================================================

# add BTLS variants and include them in main/descent below

# =============================================================================
# def frank_wolfe_BTLS(x, A, b, t, gam, alpha, beta):
#     # update x (your code here)
#     # We implement a method we found in some book that describes FW update method
#     eta_t = 1 #step size
#     gradient_f = np.dot(np.transpose(A),(np.dot(A,x)-b)) #find the gradient
#     idx_oracle = np.argmax(np.abs(gradient_f)) #find index of the largest value in the gradient vector
#     e_i = np.zeros(len(gradient_f)) #create a vector of 0's of the same size of the gradient of f
#     e_i[idx_oracle] = 1 #set the value at the maximum value of the index to 1
#     sign_grad_f_i = np.sign(gradient_f[idx_oracle]) # find the sign of the gradient of f at location i
#     s_t = -gam*sign_grad_f_i*e_i
#     while f1_x(x+eta_t*s_t, A, b) > f1_x(x, A, b) - eta_t*alpha*np.power(la.norm(s_t,2),2):
#         eta_t = eta_t * beta
#     
#     x = x + s_t*eta_t
#     return x
# 
# 
# def subgradient_BTLS(x, A, b, t, lam, alpha, beta, c=1e-5):
#     # update x (your code here), set c above
#     sub_grad = np.dot(np.transpose(A),(np.dot(A,x)-b))+lam*np.sign(x)
#     eta_t = 1
#     while f2_x(x+eta_t*(-sub_grad), A, b, lam) > f2_x(x, A, b, lam) - eta_t*alpha*np.power(la.norm(-sub_grad,2),2):
#         eta_t = eta_t*beta
#         
#     x = x - eta_t*sub_grad
#     return x
# =============================================================================



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
   
# =============================================================================
#     file1 = open("obj_fn-FW.txt","w") 
#     L = [str(obj_fn)] 
#     file1.writelines(L)
#     file1.close()
#     
#     file1 = open("rate_vec1-FW.txt","w") 
#     L = [str(rate_vec1)] 
#     file1.writelines(L)
#     file1.close()
# 
#     file1 = open("rate_vec2-FW.txt","w") 
#     L = [str(rate_vec2)] 
#     file1.writelines(L)
#     file1.close()
# 
#     file1 = open("rate_vec3-FW.txt","w") 
#     L = [str(rate_vec3)] 
#     file1.writelines(L)
#     file1.close()   
# =============================================================================
    
    return ratePerSensor, obj_fn, l1



# =============================================================================
# def param_init():
#     return 0    
# 
# 
# def intersection_area_circles(d, R, r):
#     """Return the area of intersection of two circles.
# 
#     The circles have radii R and r, and their centres are separated by d.
# 
#     """
# 
#     if d <= abs(R-r):
#         # One circle is entirely enclosed in the other.
#         return np.pi * min(R, r)**2
#     if d >= r + R:
#         # The circles don't overlap at all.
#         return 0
# 
#     r2, R2, d2 = r**2, R**2, d**2
#     alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
#     beta = np.arccos((d2 + R2 - r2) / (2*d*R))
#     return ( r2 * alpha + R2 * beta -
#              0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
#            )
# 
# def find_d(A, R, r):
#     """
#     Find the distance between the centres of two circles giving overlap area A.
# 
#     """
# 
#     # A cannot be larger than the area of the smallest circle!
#     if A > np.pi * min(r, R)**2:
#         raise ValueError("Intersection area can't be larger than the area"
#                          " of the smallest circle")
#     if A == 0:
#         # If the circles don't overlap, place them next to each other
#         return R+r
# 
#     if A < 0:
#         raise ValueError('Negative intersection area')
# 
#     def f(d, A, R, r):
#         return intersection_area_circles(d, R, r) - A
# 
#     a, b = abs(R-r), R+r
#     d = brentq(f, a, b, args=(A, R, r))
#     return d
# =============================================================================



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


def SensSelecModel(N, d, capacity, mu, partitionsArea , allPossibleSets, rectangleLength, rectangleWidth, sensorRadius, scalingFactor, lam, thresh = 2.):
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
    #lam = d*(1.+2./3.*numSelectedSensors)
    lam = 5e-4
    
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
    #lam = d*(1.+2./3.*numSelectedSensors)
    
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
    
# =============================================================================
#     file1 = open("l1.txt","w") 
#     L = [str(l1_fw_agemin[-1])] 
#     file1.writelines(L)
#     file1.close() 
# =============================================================================
    return coverageArea , obj_fn[-1] , setofSelectedSensors





def main(T=int(5e2)): 
    scalingFactor = 50
    N = np.array([4,15]) # number of sensors
    sensorRadius = np.array(100/scalingFactor)#coverage radius per sensor
    #sensorRadius = []
    #sensorRadius = np.array([1.68,1.5,1.06,1.9,1.92,1.58,1.26,1.28,1.52,1.9])    
    capacity = 1.
    d = 0.5e-3 #transmission delay
    mu = 1. #packet size
    
    
    
    
    rectangleLength = 500/scalingFactor
    rectangleWidth = 10/scalingFactor
    #areaR = rectangleLength*rectangleWidth
    numSquaresperLength = int(rectangleLength*10)
    numSquaresperWidth = int(rectangleWidth*10)
    
    pixelLength = rectangleLength/numSquaresperLength
    pixelWidth = rectangleWidth/numSquaresperWidth
    
    xPosCenterPixel1 = pixelLength/2
    yPosCenterPixel1 = pixelWidth/2
    
    coordPixels = generatePixelsCenters(xPosCenterPixel1, yPosCenterPixel1, pixelLength, pixelWidth, numSquaresperLength, numSquaresperWidth)


    coverageAreaBaseline = []
    areaWeightedAgeBaseline = []
    coverageAreaSensSelec = []
    areaWeightedAgeSensSelec = []
    selectedSensorsSensSelec = []
    coverageAreaAgeMin = []
    areaWeightedAgeAgeMin =[]
    selectedSensorsAgeMin =[]

    numIter = 5

    for ii in range(len(N)):
         temp1coverageAreaBaseline = []
         temp1areaWeightedAgeBaseline = []
         temp1coverageAreaSensSelec = []
         temp1areaWeightedAgeSensSelec = []
         temp1selectedSensorsSensSelec = []
         temp1coverageAreaAgeMin = []
         temp1areaWeightedAgeAgeMin =[]
         temp1selectedSensorsAgeMin =[]
         
         for jj in range(numIter):
             xcoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleLength-0) 
             ycoordSensors = 0 + np.random.rand(N[ii],1)*(rectangleWidth-0)
             coordSensors = np.concatenate((xcoordSensors,ycoordSensors),axis=1)
             #coordSensors  = np.array([[8.73644,0.0203496],[0.945702,0.0954615],[7.37637,0.0560039],[8.42441,0.0585362],[9.29849,0.144649],[0.215903,0.0678067],[3.15625,0.193529],[2.48066,0.0406667],[8.45596,0.0825648],[8.08148,0.108054],[8.2672,0.155429],[0.666787,0.19286],[7.27055,0.185332],[2.8861,0.125174]])
             #coordSensors = np.array([[1.91019,0.06076],[6.94741,0.104082],[2.937,0.147354],[1.23841,0.10255],[9.39445,0.193933],[0.676381,0.118743],[7.57448,0.189689],[6.36497,0.0540557],[6.88323,0.0784535],[5.74441,0.118879]])
             partitionsArea , allPossibleSets = findPartitionsAreas(pixelLength, pixelWidth, coordPixels,coordSensors,sensorRadius,N[ii])
             
             tempcoverageAreaBaseline , tempareaWeightedAgeBaseline = baselineModel(capacity/(N[ii]*mu*d), d, partitionsArea*scalingFactor**2 , allPossibleSets, scalingFactor)
             tempcoverageAreaSensSelec , tempareaWeightedAgeSensSelec , tempselectedSensorsSensSelec = SensSelecModel(N[ii], d, capacity , mu, partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor, rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor,lam ,thresh = 2.)
             tempcoverageAreaAgeMin , tempareaWeightedAgeAgeMin , tempselectedSensorsAgeMin = AgeMinModel(N[ii], d, mu, capacity , partitionsArea*scalingFactor**2 , allPossibleSets, rectangleLength*scalingFactor , rectangleWidth*scalingFactor , sensorRadius*scalingFactor, scalingFactor, T, lam ,thresh = 2.)
             
     
             temp1coverageAreaBaseline.append(tempcoverageAreaBaseline)
             temp1areaWeightedAgeBaseline.append(tempareaWeightedAgeBaseline)
             
             temp1coverageAreaSensSelec.append(tempcoverageAreaSensSelec)
             temp1areaWeightedAgeSensSelec.append(tempareaWeightedAgeSensSelec)
             temp1selectedSensorsSensSelec.append(len(tempselectedSensorsSensSelec))
     
         
     
             temp1coverageAreaAgeMin.append(tempcoverageAreaAgeMin)
             temp1areaWeightedAgeAgeMin.append(tempareaWeightedAgeAgeMin)
             temp1selectedSensorsAgeMin.append(len(tempselectedSensorsAgeMin))
     
             
         coverageAreaBaseline.append(np.sum(temp1coverageAreaBaseline)/numIter)
         areaWeightedAgeBaseline.append(np.sum(temp1areaWeightedAgeBaseline)/numIter)
          
         coverageAreaSensSelec.append(np.sum(temp1coverageAreaSensSelec)/numIter)
         areaWeightedAgeSensSelec.append(np.sum(temp1areaWeightedAgeSensSelec)/numIter)
         selectedSensorsSensSelec.append(np.sum(temp1selectedSensorsSensSelec)/numIter)
     
         
     
         coverageAreaAgeMin.append(np.sum(temp1coverageAreaAgeMin)/numIter)
# =============================================================================
#          if int(N[ii])>4:
#              hello = areaWeightedAgeAgeMin[-1]
#              add = (-1)**N[ii]*10e-6
#              areaWeightedAgeAgeMin.append(hello+add)
#          else:
# =============================================================================
         areaWeightedAgeAgeMin.append(np.sum(temp1areaWeightedAgeAgeMin)/numIter)
         selectedSensorsAgeMin.append(np.sum(temp1selectedSensorsAgeMin)/numIter)        
     
        
# =============================================================================
#     
# =============================================================================
    plt.clf()
    plt.plot(N , areaWeightedAgeBaseline, '--', label='Baseline')
    plt.plot(N , areaWeightedAgeSensSelec, '.-',label='Sensor Selection')
    plt.plot(N , areaWeightedAgeAgeMin, label='Age Minimization')
     #plt.title('Area weighted age as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
      #plt.yscale('log')
    plt.xlabel('Number of selected sensors', fontsize=12)
    plt.ylabel('Normalized average weighted age (seconds)', fontsize=10)
    plt.savefig('newAge.eps')
    plt.savefig('newAge.pdf')
      
    plt.clf()
    plt.plot(N , coverageAreaBaseline, '--', label='Baseline')
    plt.plot(N , coverageAreaSensSelec, '.-',label='Sensor Selection')
    plt.plot(N , coverageAreaAgeMin, label='Age Minimization')
     #plt.title('Coverage Area as a function of the number of selected sensors', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xlabel('Number of selected sensors', fontsize=12)
    plt.ylabel('Coverage Area ($m^2$)', fontsize=10)
    plt.savefig('newcovArea.eps')
    plt.savefig('newcovArea.pdf')
# # =============================================================================
# =============================================================================

# =============================================================================
#    plt.clf()
#    plt.scatter(coverageAreaSensSelec , areaWeightedAgeSensSelec,label='Sensor Selection')
#    #plt.title('Area weighted age vs. coverage area as $\lambda$ increases', fontsize=12)
#    plt.legend()
#    plt.grid()
#    axes = plt.gca()
#    axes.set_ylim([13.5e-4,14.5e-4])
#     #plt.yscale('log')
#    plt.xlabel('Coverage area ($m^2$)', fontsize=12)
#    plt.ylabel('Area weighted age (seconds)', fontsize=10)
#    plt.savefig('AgevsCo2.eps')
#    plt.savefig('AgevsCo2.pdf')
# =============================================================================




    
if __name__ == "__main__":
    main()