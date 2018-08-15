
import pandas as pd
import numpy as np
import cv2
import math
mypath = "/Users/Sandra/Downloads/AppliedMachineLearningData/rawdata"

fileList = []

devDataFile = pd.read_csv('devData.csv')
fileList.append(devDataFile)
trainDataFile = pd.read_csv('trainData.csv')
fileList.append(trainDataFile)
testDataFile = pd.read_csv('testData.csv')
fileList.append(testDataFile)
finalTestDataFile = pd.read_csv('finalTestData.csv')
fileList.append(finalTestDataFile)


def extractFeature(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    return [laplacian, sobelx, sobely, gaussian, bilateral]

def extractFeaturesAll(imgList):
    listPop = []
    path = "rawdata/"
    for filename in imgList:
        print(filename)
        siftFeatures = []
        img = cv2.imread(path + filename)
        sift = findSiftFeatures(img)
        features = extractFeature(img)
        print(len(sift))
        for feat in features:
            curFeat = []

            for i in range(0, 5):

                (x, y) = (sift[i].pt)
                x = math.floor(x)
                y = math.floor(y)
                # print((x,y))
                # print(np.shape(feat))
                if(x>= (np.shape(feat))[0]):
                    x = np.size(feat,0)-1
                if (y >= (np.shape(feat))[1]):
                    y = np.size(feat, 1) - 1

                curFeat= curFeat + ((feat[x,y]).tolist())

            siftFeatures = siftFeatures + (curFeat)
        listPop.append(siftFeatures)
    return listPop

def findSiftFeatures(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    kp, des = sift.compute(gray, kp)
    return kp

def createDictionary(imgList):

        Z = extractFeaturesAll(imgList)
        # Z = np.float32(Z)
        g = 0
        b = None
        for z in Z:
            z = (np.matrix(z, dtype=np.float32))
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 50
            ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            newCenters = center.flatten()

            if g == 0:
                b = [center.flatten()]
                g = 1

            else:
                b = np.append(b,  [center.flatten()], axis=0)

        np.savetxt('destTest.csv', b, delimiter=',')


        return b

def foregroundExtraction(filename):
    path = path = "rawdata/";
    img = cv2.imread(path + filename)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    height, width, channels = img.shape

    rect = (int(width * (.1)), int(height * .1), int(width * (.8)), int(height * .8))

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    word = (filename).split('.', 1)[0]
    return img



    # cv2.imwrite(path + 'foreg_' + word + '.png', dst)

def colorReduce(img):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    colorList = []
    for i in range(1, len(center)):
        color = center[i]
        r = (round(color[2]));
        g = (round(color[1]));
        b = (round(color[0]));

        colorList.append(r)
        colorList.append(g)
        colorList.append(b)

    return colorList

def fft(filename):
    path = "rawdata/"
    img = cv2.imread(path + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    M, N = magnitude_spectrum.shape
    K = 5
    L = 5

    MK = M // K
    NL = N // L
    pooled = magnitude_spectrum[:MK * K, :NL * L].reshape(MK, K, NL, L).max(axis=(1, 3));
    x, y = pooled.shape
    mid_x = x // 2
    mid_y = y // 2
    offset = 3
    reshaped = pooled[mid_x - offset:mid_x + offset, mid_y - offset:mid_y + offset].reshape(1, 36)

    return reshaped.tolist()





def update():
    fileList = devDataFile['filename'].tolist()
    # createDictionary(fileList)

    i = 0
    fftCSV = []
    for filename in fileList:
        # img = foregroundExtraction(filename);
        # colorString = colorReduce(img);
        reshaped = fft(filename)
        fftCSV.append(reshaped[0])

        i+=1
        print(i)
    np.savetxt('fftForeDev.csv', np.array(fftCSV), delimiter=',')
    print("one file down")

    colorCSV = []
    fileList = trainDataFile['filename'].tolist()
    for filename in fileList:
        reshaped = fft(filename)
        colorCSV.append(reshaped[0])
        i += 1
        print(i)
    np.savetxt('fftForeTrain.csv', np.array(colorCSV), delimiter=',')
    print("second file down")

    colorCSV = []
    fileList = testDataFile['filename'].tolist()
    for filename in fileList:
        reshaped = fft(filename)
        colorCSV.append(reshaped[0])
        i += 1
        print(i)
    np.savetxt('fftTest.csv', np.array(colorCSV), delimiter=',')
    print("third file down")


    colorCSV = []
    fileList = finalTestDataFile['filename'].tolist()
    for filename in fileList:
        reshaped = fft(filename)
        colorCSV.append(reshaped[0])
        i += 1
        print(i)
    np.savetxt('fftFinalTest.csv', np.array(colorCSV), delimiter=',')
    print("fourth file down")

    # colorCSV = []
    # fileList = trainDataFile['filename'].tolist()
    # for filename in fileList:
    #     img = foregroundExtraction(filename);
    #     colorString = colorReduce(img);
    #     colorCSV.append(colorString)
    #     i += 1
    #     print(i)
    # np.savetxt('colorForeTrain.csv', np.array(colorCSV), delimiter=',')
    # print("second file down")



    # centers = createDictionary(fileList)
    # print(centers)
    # return centers


update()

