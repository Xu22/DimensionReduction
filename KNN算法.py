'''
KNN,通常在分类任务中可使用“投票法”，即选择这K个样本中出现最多的类别标记作为预测结果
    在回归任务中可使用“平均法”，即将这K个样本的实值输出标记的平均值作为预测结果
'''
import numpy as np
import operator
def createDataSet():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  #tile函数位于python模块 numpy.lib.shape_base中，他的功能是重复某个数组。比如tile(A,n)，功能是将数组A重复n次，构成一个新的数组
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #每一行各列求和    axis=0代表列,axis=1代表行
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort() #argsort其实是返回array排序后的下标（或索引）
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   #Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值。
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   #operator.itemgetter函数 operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号
    return sortedClassCount[0][0]
if __name__ == '__main__':
    group, labels = createDataSet()
    test = [101,20]
    test_class = classify(test, group, labels, 3)
    print(test_class)
# from numpy import *
# import operator
# def knn(k,testData,trainData,labels):
#     trainDataSize=trainData.shape[0]
#     '''
#     a=array([1,2,5,4,3])
#     tile(a,2)   #array([1,2,5,4,3,1,2,5,4,3])
#     tile(a,(2,1)) #array([[1,2,5,4,3],
#                          [1,2,5,4,3]])
#     '''
#     dif=tile(testData,(trainDataSize,1))-trainData
#     sqdif=dif**2
#     sumsqdif=sqdif.sum(axis=1)   #每一行各列求和
#     distance=sumsqdif**0.5
#     sortDistance=distance.argsort()  #排序
#     count={}
#     for i in range(0,k):
#         vote=labels[sortDistance[i]]
#         count[vote]=count.get(vote,0)+1
#     sortcount=sorted(count.items(),key=operator.itemgetter(1),reverse= True)    #operator.itemgetter函数 operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号
#     return sortcount[0][0]
