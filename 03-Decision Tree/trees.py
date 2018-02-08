## ID3
from math import log
import operator
import matplotlib.pyplot as plt
  

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

## 3.1 决策树的构造
# 3.1.1 信息增益--计算给定数据集的香农熵:H = -∑ p(xi) log p(xi)
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 1. 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 2. 计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 3.1.2 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    # 1.创建新的list对象
    retDataSet = []
    # 2.chop out axis used for splitting
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选取最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 1.创建唯一的分类标签列表
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 2. 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 3. 计算最好的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 3.1.3 递归构建决策树
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classConunt.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

#创建树的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 1. 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 2. 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}} #输出de格式
    # 3. 得到的列表包含的所有属性值
    del(labels[bestFeat]) #用完的feature删除
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree




## 3.2 在python中使用Matplotlib注解绘制树形图
# 3.2.1 Matplotlib注解 使用文本注解绘制树节点
# 1. 定义文本框和箭头格式
decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
# 2. 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,xy = parentPt, xycoords = 'axes fraction',\
                            xytext = centerPt,textcoords = 'axes fraction',\
                            va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)

'''def createPlot():
    fig = plt.figure(1,facecolor = "white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('decisionNode', (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('leafNode',(0.8,0.1), (0.3,0.8), leafNode)
    plt.show()
    '''

# 3.2.2 构造注解树
# 获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstSides = list(myTree.keys()) 
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__ =='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstSides = list(myTree.keys()) 
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

# 1. 在父子节点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
# 2. 计算宽与高
def plotTree(myTree, parentPt,nodeTxt):
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys()) 
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 + float(getNumLeafs(myTree)))/2.0/plotTree.totalW, plotTree.yOff)
    # 3. 标记子节点属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 4. 减少y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            plotTree(secondDict[key],cntrPt,str(key))  #recursion
        else: #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt,leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key)) # 计算父节点子节点的中间位置
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
            


## 3.3 测试和存储分类器
# 3.3.1 测试算法：使用决策树执行分类
def classifiy(inputTree, featLabels, testVec):
    firstSides = list(myTree.keys()) 
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
        else:
            classLabel = secondDict[key]
    return classLabel


## 3.4 使用决策树预测隐形眼镜类型
'''
    >>> fr = open('lenses.txt')
    >>> lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    >>> lensesLabels = ['age','prescript','astigmatic','tearRate']
    >>> lensesTree = createTree(lenses, lensesLabels)
'''

            
    
