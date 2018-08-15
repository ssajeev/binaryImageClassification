from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import random

mypath = "/Users/Sandra/Downloads/AppliedMachineLearningData"
manMadePics = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".jpg") and f.startswith("manmade")]
naturalPics = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".jpg") and f.startswith("natural")]

random.shuffle(manMadePics)
random.shuffle(naturalPics)


fileList = []

devDataFile = open('devData.arff','w')
fileList.append(devDataFile)
trainDataFile = open('trainData.arff','w')
fileList.append(trainDataFile)
testDataFile = open('testData.arff','w')
fileList.append(testDataFile)
finalTestDataFile = open('finalTestData.arff','w')
fileList.append(finalTestDataFile)


for file in fileList:

    file.write("@relation natural_vs_manmade\n")
    file.write("@attribute filename string\n")


    file.write("@attribute color1red numeric\n")
    file.write("@attribute color1green numeric\n")
    file.write("@attribute color1blue numeric\n")

    file.write("@attribute color2red numeric\n")
    file.write("@attribute color2green numeric\n")
    file.write("@attribute color2blue numeric\n")

    file.write("@attribute color3red numeric\n")
    file.write("@attribute color3green numeric\n")
    file.write("@attribute color3blue numeric\n")

    file.write("@attribute color4red numeric\n")
    file.write("@attribute color4green numeric\n")
    file.write("@attribute color4blue numeric\n")

    file.write("@attribute keypoints numeric\n")
    file.write("@attribute descriptors numeric\n")


    file.write("@attribute class {manmade, natural}\n")

    file.write("@data\n")




trainCutoffManmade4 = len(manMadePics)//16
trainCutoffNatural4 = len(naturalPics)//16

def siftFeatureDes(filename):
    img = cv2.imread(filename)
    gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    kp, des = sift.compute(gray, kp)
    #assert(len(kp) >= 4)
    siftString = str(len(kp)) + ", " + str(len(des))
    return siftString

def colorReduce(filename):
    img = cv2.imread(filename)
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    colorString = ""
    for i in range(0, len(center)):
        color = center[i]
        r = str(round(color[2]));
        g = str(round(color[1]));
        b = str(round(color[0]));

        colorString = colorString + " " + r + " " + g + " " + b;

    return colorString



def stringGen(i, listName):
    fileName = listName[i]
    fileClass = fileName[0: fileName.find("_")]
    siftString = siftFeatureDes(fileName)
    colorString = colorReduce(fileName)

    return fileName + " " + colorString + " " + siftString + " " + fileClass + "\n"




for i in range(0, trainCutoffManmade4):
    print(i)
    devDataFile.write(stringGen(i, manMadePics))


for i in range(0, trainCutoffNatural4):
    print(i)
    devDataFile.write(stringGen(i, naturalPics))

trainCutoffManmade = len(manMadePics)//2
trainCutoffNatural = len(naturalPics)//2

for i in range(trainCutoffManmade4, trainCutoffManmade):
    print(i)
    trainDataFile.write(stringGen(i, manMadePics))


for i in range(trainCutoffNatural4, trainCutoffNatural):
    print(i)
    trainDataFile.write(stringGen(i, naturalPics))



for i in range(0, trainCutoffManmade4):
    print(i)
    trainDataFile.write(stringGen(i, manMadePics))


for i in range(0, trainCutoffNatural4):
    print(i)
    trainDataFile.write(stringGen(i, naturalPics))


trainCutoffManmade2 = (len(manMadePics)*7)//8
trainCutoffNatural2 = (len(naturalPics)*7)//8

for i in range(trainCutoffManmade, trainCutoffManmade2):
    print(i)
    testDataFile.write(stringGen(i, manMadePics))


for i in range(trainCutoffNatural, trainCutoffNatural2):
    print(i)
    testDataFile.write(stringGen(i, naturalPics))


for i in range(trainCutoffManmade2, len(manMadePics)):
    print(i)
    finalTestDataFile.write(stringGen(i, manMadePics))


for i in range(trainCutoffNatural2, len(naturalPics)):
    print(i)
    finalTestDataFile.write(stringGen(i, naturalPics))