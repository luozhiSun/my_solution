"""使用自己的方法创建无人机自主路径规划环境
2020年12月2日"""
from pylab import *

import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 用来给出三维坐标系。
import xlwt

plt.ion()
Boundary=30
TargetNum=3

#类定义
class UAV:
    #local代表位置信息
    xloacl=0
    ylocal=0
    #本地目标概率矩阵,初始化为0.5
    LoaclTPM = np.zeros((Boundary, Boundary), dtype=np.float)
    for i in range(Boundary):
        for j in range(Boundary):
            LoaclTPM[i][j]=0.5

    #本地栅格探测次数矩阵
    LoalcSensorTime=np.zeros((Boundary,Boundary),dtype=np.int)
    #需要设定传感器的参数
    def __init__(self,pd,pf):
        self.pd=pd
        self.pf=pf
    #更新位置信息
    def SetLocal(self,x,y):
        self.xloacl=x
        self.ylocal=y
    #获取位置信息
    def GetLoacl(self):
        return self.xloacl,self.ylocal
    #更新本地传感器信息,传入目前的概率与目标是否存在的标志
    #返回下一时刻的目标存在概率,计算概率
    def UpdateLocalTPM(self,bt,x,y):
        p=self.LoaclTPM[x][y]
        #观测的栅格中目标存在
        if bt==True:
            pnext = self.pd * p / (p * self.pd + self.pf * (1 - p))
        #观测的栅格中目标不存在
        else:
            pnext = (1 - self.pd) * p / ((1 - self.pd) * p + (1 -self. pf) * (1 - p))
        self.LoaclTPM[x][y]=pnext
    #获取本地的目标概率矩阵
    def GetLoaclTPM(self):
        return self.LoaclTPM

#方法的定义


#传感器探测次数更新矩阵
def UpdateSensorTime(UAV,x,y):
    UAV.LoalcSensorTime[x][y]=UAV.LoalcSensorTime[x][y]+1
#判断UAV所在位置是否为目标位置
def ISTargetLocal(TargetArray,x,y):
    for i in range(TargetNum):
        if TargetArray[i][0]==x and TargetArray[i][1]==y:
            return True
    return False
#目标概率图更新,b代表对应位置目标是否存在



map_grid = np.full((Boundary, Boundary), float(10), dtype=np.float)

# 画出定义的栅格地图
plt.figure(figsize=(6,6))
my_x_ticks = np.arange(0, Boundary+10, 5)
my_y_ticks = np.arange(0, Boundary+10, 5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlim(0,Boundary)
plt.ylim(0,Boundary)
'''静态目标初始位置设定33'''
TargetArray=np.array([[12,8],[23,14],[28,9]])
for i in range(TargetNum):
    plt.scatter(TargetArray[i][0], TargetArray[i][1], marker='^', c='red', s=100)
plt.grid(True)
#创建UAV1
UAV1=UAV(0.8,0.2)


#测试代码，随机运动，显示TPM

'''二维情况下设置UAV的8个运动方向'''
#定义一个数组存储运动方向，x和y的位置变化
#设置7X2数组存储xy方向运动的距离
#第一行定义为方向，2，3行定义为对应的位置变化
MoveType=np.zeros((8,3),dtype=np.int)
j=1
for i in range(8):
    MoveType[i][0]=j
    j=j+1
'''用数学公式表示无人机之间的运动'''
'''手动赋值,无人机运动矩阵'''
'''设定初始运动位置为（0，0），运动为右上角'''
#初始位置开始运动
move=[1,1]
Dist=[0,0,0]
#对三个运动方向随机选择
tx=move[0]
ty=move[1]
angle=0
Set=[0,0,0]
#用来计算迭代次数
t=0

k=0
x=0
y=0
for x in range(Boundary):
    if(k%2==0):
        for i in range(Boundary-1):
            # 更新传感器探测次数矩阵
            UpdateSensorTime(UAV1, move[0], move[1])
            # 判断所在位置是否为目标位置
            b = ISTargetLocal(TargetArray, move[0], move[1])
            print(b)
            # 对概率进行更新
            UAV1.UpdateLocalTPM(b, move[0], move[1])

            nextx=x
            nexty=i+1
            plt.plot([move[0], nextx], [move[1], nexty], 'm.-', c='black')
            move[0] = nextx
            move[1] = nexty
            plt.pause(0.000001)
            print(move)
    else:
        nexty=Boundary
        for i in range(Boundary-1):
            # 更新传感器探测次数矩阵
            UpdateSensorTime(UAV1, move[0], move[1])
            # 判断所在位置是否为目标位置
            b = ISTargetLocal(TargetArray, move[0], move[1])
            print(b)
            # 对概率进行更新
            UAV1.UpdateLocalTPM(b, move[0], move[1])
            nextx=x
            nexty=nexty-1

            plt.plot([move[0], nextx], [move[1], nexty], 'm.-', c='black')
            move[0] = nextx
            move[1] = nexty
            plt.pause(0.000001)
            print(move)
    k=k+1


#对无人机的八个方向的随机运动，并且设定了边界限制
 #显示目标概率图
TPM=UAV1.GetLoaclTPM()
print(TPM)# 对TPM进行画图
def ShowTPM(TPM):
    # 画出三维坐标系：
    figure = plt.figure()
    axes = Axes3D(figure)

    X = np.arange(0, Boundary, 1.0)

    Y = np.arange(0, Boundary, 1.0)

    # 限定图形的样式是网格线的样式：

    X, Y = np.meshgrid(X, Y)

    # 绘制曲面，采用彩虹色着色：
    surf = axes.plot_surface(X, Y, TPM, cmap='rainbow')

    # 图形可视化：
    plt.colorbar(surf, shrink=0.5, aspect=7)
    plt.show()
    plt.pause(100)

print(TPM)
#ShowTPM(TPM)
#将数据写入excel表格
def save(data, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
    f.save(path)

#将目标概率矩阵存储到表格中
save(TPM,'test.xls')
plt.show()
