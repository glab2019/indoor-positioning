import numpy as np
import math
import sys
sys.dont_write_bytecode = True
#输入：文件列表，大区间索引列表，大区间大小，小区间大小，信号强度（列表）
#输出：小区间索引列表
#过程：
#     从索引*大区间的位置开始，每隔一个小区间取一次平均值，存入列表中
#     用NN算法求出信号强度属于哪个区间，并返回该区间的索引
import time
start=time.time()

def calDistance(data,rss): #计算距离
    distance =0.0
    for i in range(6):
        distance += abs(data[i] - rss[i])**2
    return distance
    
def nn_location(data_list,rss): #选择距离最近的点
    distance_list = [] 
    for data in data_list:
        distance = calDistance(data,rss)
        distance_list.append(distance)
    nn_index=distance_list.index(min(distance_list))
    return nn_index

def average_num(d_list,min_l,max_l,min_w,max_w):
    average=[]
    for i in range(6):
        sum=0.0
        for m in range(min_w,max_w):
            for n in range(min_l,max_l):
                sum+=d_list[n+(m-1)*199][i]
        average.append(sum/(max_l-min_l)/(max_w-min_w))
    return average
        

def max_division(data_list,Index_list,rss_list): #选择最合适的区间
    rss_division = [0,0,0,0] 
    for i in range(len(rss_list)):
        max_length = 1990
        max_width = 1490
        min_length = 0
        min_width = 0
        for j in Index_list[i]:
            if j == 5:continue
            if j==0 or j==2:  
                max_width = max_width-int((max_width-min_width)/2) 
            else:min_width = min_width+int((max_width-min_width)/2)
            if j-1>0: min_length = min_length+int((max_length-min_length)/2)
            else:
                max_length = max_length-int((max_length-min_length)/2)
        rss_division[0] = average_num(data_list,int(min_length/10),int(min_length/10)+int((max_length-min_length)/20),int(min_width/10),int(min_width/10)+int((max_width-min_width)/20))
        rss_division[1] = average_num(data_list,int(min_length/10),int(min_length/10)+int((max_length-min_length)/20),int(min_width/10)+int((max_width-min_width)/20),int(max_width/10))
        rss_division[2] =  average_num(data_list,int(min_length/10)+int((max_length-min_length)/20),int(max_length/10),int(min_width/10),int(min_width/10)+int((max_width-min_width)/20))
        rss_division[3] = average_num(data_list,int(min_length/10)+int((max_length-min_length)/20),int(max_length/10),int(min_width/10)+int((max_width-min_width)/20),int(max_width/10))
        index = nn_location(rss_division,rss_list[i])
        Index_list[i].append(index)
    return Index_list

def first_division(div,Index_list,rss_list): #选择最合适的区间
    for i in range(len(rss_list)):
        index = nn_location(div[0:4],rss_list[i])
        Index_list[i].append(index)
    return Index_list
    
def afirst_division(div,Index_list,rss_list,a): #选择最合适的区间
    index = nn_location(div[0:4],rss_list[a])
    Index_list[a].append(index)
    return Index_list
    
def sec_division(div,Index_list,rss_list): #选择最合适的区间
    for i in range(len(rss_list)):
        if Index_list[i][1]==0:Index_list=afirst_division(div[4:8],Index_list,rss_list,i)
        if Index_list[i][1]==1:Index_list=afirst_division(div[8:12],Index_list,rss_list,i)
        if Index_list[i][1]==2:Index_list=afirst_division(div[12:16],Index_list,rss_list,i)
        if Index_list[i][1]==3:Index_list=afirst_division(div[16:20],Index_list,rss_list,i)
    return Index_list

    
# 定位准确度
def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

#读入训练集数据
x_train = np.loadtxt('../data/train_rss.txt')
y_train = np.loadtxt('../data/train_location.txt')

#读入测试集数据
x_test = np.loadtxt('../data/trace_rss.txt')[0:1000, :]
y_test = np.loadtxt('../data/trace_location.txt')[0:1000 :]

div_list = np.loadtxt('../data/division.txt')

index_list=np.zeros(1000).reshape(1000,1).tolist()
for i in range(len(index_list)):
    index_list[i]=[5]
    
index_list=first_division(div_list,index_list,x_test)
index_list=sec_division(div_list,index_list,x_test)
for i in range(5):
    index_list=max_division(x_train,index_list,x_test)
pre_list = []
for i in range(len(index_list)):
    max_length = 1990
    max_width = 1490
    min_length = 0
    min_width = 0
    for j in index_list[i]:
            if j == 5:continue
            if j==0 or j==2: 
                max_width = max_width-int((max_width-min_width)/2) 
            else:min_width = min_width+int((max_width-min_width)/2)
            if j-1>0: min_length = min_length+int((max_length-min_length)/2) 
            else:
                max_length = max_length-int((max_length-min_length)/2) 
    pre_list.append(y_train[int(min_length/10)+int(min_width/10)*199])
acc = accuracy(pre_list, y_test)
print ("accuracy: ", acc/100, "m")


#中间写上代码块
end=time.time()
print('Running time: %s Seconds'%(end-start))




