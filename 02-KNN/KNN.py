from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0],[0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels

##KNN算法
def classify0(inX, dataSet, labels, k):
    ##the number of data
    dataSetSize = dataSet.shape[0]
    #  1.calculate the distance:此处欧氏距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis =  1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #sort by index [2,3,1,0]
    #  2.choose the smallest k (k = 3) distances
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #{'A':1,'B':2}
    #  3.sort
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)  #[('B', 2), ('A', 1)]
    return sortedClassCount[0][0]

# 2.2 使用k-近邻算法改进约会网站的匹配效果
# 2.2.1 准备数据:从文本文件中解析数据, 将文本文件转化为numpy
def file2matrix(filename):
    fr = open(filename)
    # 1. get the number of lines of a file
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 2. create Numpy matrix and return
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    # 3. 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # Numpy自动识别数据类型
        returnMat[index,:] = listFromLine[0:3] #first 3 (features)
        classLabelVector.append(int(listFromLine[-1])) #the last one (class)
        index += 1
    return returnMat,classLabelVector

# 2.2.2 分析数据:使用matplotlib创建散点图
'''
    >>> import matplotlib
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
    >>> plt.show()
'''

# 2.2.3 准备数据:归一化数值--将特征等权重
'''
    将取值范围处理为0到1或-1到1之间：newValue = (oldValue - min) / (max - min)
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 0:使函数可以从当前列中选取最小值，而不是选取当前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1)) #特征值矩阵1000*3， minVals和range的值为1*3，tile（）将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals
    
# 2.2.4 测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 0.10 #测试比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] #行数
    numTestVecs = int(m*hoRatio) #测试个数
    errorCount = 0.0
    k = 3
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    
# 2.2.5 使用算法：构建完整可用系统--约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    '''
       3个features：每年获得的飞行常客里程数，玩视频游戏所耗时间百分比，每种消费的冰淇淋公升数
    '''
    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))   
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = array([ffMiles, percentTats,iceCream])
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    k = 3
    classifierResult = classify0((inArr-minVals)/ranges, normMat,datingLabels,k)
    print("you will probably like this person: ", resultList[classifierResult - 1])

# 2.3 使用k-近邻算法的手写识别系统
# 2.3.1 准备数据：将图像转化为测试向量
'''
    该函数创建1*1024的Numpy数组，
    然后打开给定的文件，
    循环读出文件的前32行，
    并将每行的头32个字符值存储在Numpy数组中，
    最后返回数组。
'''
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range (32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 2.3.2 测试算法:使用k-近邻算法识别手写数字
def handwritingClassTest():
    # 1. load the training set
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    # 2. 从文件名解析分类数字 (train)
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    # 3. load the testing set
    testFileList =listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    # 4. test
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n the total error rate is: %f" %(errorCount/float(mTest)))
        
    
    

    
    

        
    
    
    
