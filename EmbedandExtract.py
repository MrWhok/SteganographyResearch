import cv2
import math
import numpy as np


def secretDataStep(Payload,k):
    #Convert to binary
    binaryPayload = np.array([format(num, '08b') for num in Payload])
    # print("New Payload:", binaryPayload)
    concenatedPayload= ''.join(binaryPayload)

    #slicing to k digit
    slicedPayload = [concenatedPayload[i:i+k] for i in range(0, len(concenatedPayload), k)]
    print("Sliced Payload:", slicedPayload)

    base10Payload = [int(binary_string, 2) for binary_string in slicedPayload]
    base10Payload = np.array(base10Payload)
    print("Base10 Payload:", base10Payload)
    return base10Payload

def centerShiftOperation(s,d,n):
    dAks1=getDAks1(s,d)
    # print("dAks1:",dAks1)
    dAks2=getDAks2(dAks1,n)
    # print("dAks2:",dAks2)

    d1=calculateD1(dAks2)
    print("d1:",d1)
    d2=calculateD2(dAks2)
    print("d2:",d2)

    return d1,d2

def getStego(cpX1,cpX2,d1,d2):
    x1=calcualateStego(cpX1,d1)
    x2=calcualateStego(cpX2,d2)
    print("x1:",x1)
    print("x2:",x2)
    return x1,x2


def getD(X,n):
    d=[]
    denom=pow(2,8-n)
    for i in range(len(X)):
        d.append(math.floor(int(X[i]) / denom ))
    
    arrD=np.array(d,dtype=int)
    # print(d)
    return arrD

def getDAks1(s,d):
    # print("len s:",len(s))
    # print("len d:",len(d))
    dAks1=[]
    for i in range(len(s)):
        dAks1.append(s[i]-d[i])
    return dAks1
        
def getDAks2(dAks1,n):
    dAks2=[]
    substractor=pow(2,n-1)
    dAks2=[dAks1[i]-substractor for i in range(len(dAks1))]
    return dAks2

def calculateD1(dAks2):
    d1=[]
    d1=[math.floor(dAks2[i]/2) for i in range(len(dAks2))]
    return d1

def calculateD2(dAks2):
    d1=[]
    d1=[math.floor(dAks2[i]/(-2)) for i in range(len(dAks2))]
    return d1

def calcualateStego(cpX,d):
    print("d:",d)
    stego=cpX
    counter=1
    
    for i in range(len(cpX)):
        if counter <= len(d):
            stego[i]=cpX[i]+d[counter-1]
            counter+=1
        else:
            stego[i]=cpX[i]
    print("stego:",stego)
    return stego

def calculateDifferent(X1, X2):
    print("X1 in calculateDifferent:", X1)
    print("X2 in calculateDifferent:", X2)
    
    # Convert to signed integers to handle negative differences
    X1_signed = X1.astype(np.int16)
    X2_signed = X2.astype(np.int16)
    
    diff = X1_signed - X2_signed
    # print("diff:", diff)
    
    return diff

def getCoverImage(X1, X2):
    X = np.zeros_like(X1)  
    for i in range(len(X1)):
        X[i] = math.ceil((int(X1[i]) + int(X2[i])) / 2)
    return X




# Load n value
n=3
k=n+1

# Load the cover image and display it
X = cv2.imread('/home/aydin/Vscode/Kuliah/SteganographyResearch-master/7.1.03.tiff', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Input: Cover image', X)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

w,h=X.shape
XFlatten=X.flatten() #convert to 1D array
print("XFlatten:",XFlatten)
d=getD(XFlatten,n)
# for i in d:
#     print(i,end=" ")
cpX1=np.copy(XFlatten)
cpX2=np.copy(XFlatten)

# Load the payload
with open('/home/aydin/Vscode/Kuliah/SteganographyResearch-master/random_numbers.txt', 'r') as file:
    payload_data = file.read().strip().split('\n')

# Convert the payload data to a numpy array
Payload = np.array([int(char) for char in payload_data])
print("Payload type:", type(Payload))
print("Payload shape:", Payload.shape)
print("Payload:", Payload)
newPayload=secretDataStep(Payload,k)

d1,d2=centerShiftOperation(newPayload,d,n)

x1,x2=getStego(cpX1,cpX2,d1,d2)
# print("x1:",x1)
# print("x2:",x2)
X1=x1.reshape(w,h)
X2=x2.reshape(w,h)

# cv2.imshow('Output: Stego image', X1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
psnr_value = cv2.PSNR(X, X1)
print("PSNR Value1:", psnr_value)


# cv2.imshow('Output: Stego image', X2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
psnr_value = cv2.PSNR(X, X2)
print("PSNR Value2:", psnr_value)



##EXTRACTION
dAks2=calculateDifferent(x1,x2)
print("diff:",dAks2)
X=getCoverImage(x1,x2)
print("X:",X)
Xtemp=X.reshape(w,h)
# print("Are equal:",np.array_equal(XFlatten,X))
# cv2.imshow('Output: cover image', Xtemp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


