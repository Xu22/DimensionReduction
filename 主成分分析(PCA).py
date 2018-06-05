'''
主成分分析（PCA）：
    在用统计分析方法研究多变量的课题时，变量个数太多就会增加课题的复杂性。
    人们自然希望变量个数较少而得到的信息较多。在很多情形，变量之间是有一定的相关关系的，
    当两个变量之间有一定相关关系时，可以解释为这两个变量反映此课题的信息有一定的重叠。
    主成分分析是对于原先提出的所有变量，将重复的变量（关系紧密的变量）删去多余，建立尽可能少的新变量，
    使得这些新变量是两两不相关的，而且这些新变量在反映课题的信息方面尽可能保持原有的信息。
    设法将原来变量重新组合成一组新的互相无关的几个综合变量，
    同时根据实际需要从中可以取出几个较少的综合变量尽可能多地反映原来变量的信息的统计方法叫做主成分分析或称主分量分析，
    也是数学上用来降维的一种方法
通过对原始变量进行线性组合，得到优化的指标
把原先多个指标的计算降维为少量几个经过优化指标的计算（占去绝大部分份额）
基本思想：设法将原先众多具有一定相关性的指标，重新组合为一组新的互相独立的综合指标，并代替原先的指标
'''
'''
协方差;Cov(X,Y)=E[XY]-E[X]E[Y]
'''
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
X=np.array([[1,-1],[2,1],[-6,-3],[-3,-2],[1,1],[3,2],[4,3]])
# pca=PCA(copy=True,n_components=2,whiten=False)
pca=PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_ratio_,pca.transform(X))
#可视化
import matplotlib.pyplot as plt
plt.figure()
plt.plot(pca.explained_variance_,'k',linewidth=2)
plt.xlabel('n_components',fontsize=16)
plt.ylabel('explained_variance_',fontsize=16)
plt.show()
'''
先创建一个PCA对象，其中参数n_components表示保留的特征数，默认为1。如果设置成‘mle’,那么会自动确定保留的特征数
最后显示的 参数 explained_variance_ratio_：array, [n_components]返回 所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率，
'''
# class DimensionValueError(ValueError):
#     '''定义异常类'''
#     pass
# class PCA(object):
#     '''定义PCA类'''
#     def __init__(self,x,n_components=None):
#         self.x=x
#         self.dimension=x.shape[1]
#         if n_components and n_components>=self.dimension:
#             raise DimensionValueError("n_components error")
#         self.n_components=n_components
#     def cov(self):
#         '''求x的协方差矩阵'''
#         x_T=np.transpose(self.x)    #转置
#         x_cov=np.cov(x_T)
#         return x_cov
#     def get_feature(self):
#         '''求协方差矩阵C的特征值和特征向量'''
#         x_cov=self.cov()
#         a,b=np.linalg.eig(x_cov)    #计算矩阵的特征值和特征向量
#         m=a.shape[0]
#         c=np.hstack((a.reshape((m,1)),b))   #增加维度
#         c_df=pd.DataFrame(c)
#         c_df_sort=c_df.sort(columns=0,ascending=False)
#         return c_df_sort
#     def reduce_dimension(self):
#         '''指定维度降维和根据方差贡献率自动降维'''
#         c_df_sort=self.get_feature()
#         varience=self.explained_varience_()
#         if self.n_components:
#             p=c_df_sort.values[0:self.n_components,1:]
#             y=np.dot(p,np.transpose(self.x))
#             return np.transpose(y)
#         varience_sum=sum(varience)
#         varience_radio=varience/varience_sum
#         varience_contribution=0
#         for R in range(self.dimension):
#             varience_contribution+=varience_radio[R]
#             if varience_contribution>=0.99:
#                 break
#         p=c_df_sort.values[0:R+1,1:]
#         y=np.dot(p,np.transpose(self.x))
#         return np.transpose(y)