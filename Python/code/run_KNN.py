# 导入数据
import numpy as np
import scipy.io as scio
from sklearn import neighbors
import warnings
import time
start=time.time()
warnings.filterwarnings('ignore')
offline_location, offline_rss = np.loadtxt('../data/train_location.txt'), np.loadtxt('../data/train_rss.txt')
trace, rss = np.loadtxt('../data/trace_location.txt')[0:1000, :], np.loadtxt('../data/trace_rss.txt')[0:1000, :]


# 定位准确度
def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

# knn回归
from sklearn import neighbors
knn_reg = neighbors.KNeighborsRegressor(40, weights='uniform', metric='euclidean')
predictions = knn_reg.fit(offline_rss, offline_location).predict(rss)
acc = accuracy(predictions, trace)
print ("accuracy: ", acc/100, "m")

#中间写上代码块
end=time.time()
print('Running time: %s Seconds'%(end-start))

# 预处理，标准化数据
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(offline_rss)
X_train = standard_scaler.transform(offline_rss)
Y_train = offline_location
X_test = standard_scaler.transform(rss)
Y_test = trace

# 交叉验证，在knn里用来选择最优的超参数k
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
parameters = {'n_neighbors':range(1, 50)}
knn_reg = neighbors.KNeighborsRegressor(weights='uniform', metric='euclidean')
clf = GridSearchCV(knn_reg, parameters)
clf.fit(offline_rss, offline_location)
scores = clf.cv_results_['mean_test_score']
k = np.argmax(scores) #选择score最大的k

# 绘制超参数k与score的关系曲线
import matplotlib.pyplot as plt
plt.plot(range(1, scores.shape[0] + 1), scores, '-o', linewidth=2.0)
plt.xlabel("k")
plt.ylabel("score")
plt.grid(True)
plt.savefig("../pictures/file1.png")
plt.show()


# 使用最优的k做knn回归
knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
predictions = knn_reg.fit(offline_rss, offline_location).predict(rss)
acc = accuracy(predictions, trace)
print ("accuracy: ", acc/100, "m")

# 训练数据量与accuracy
data_num = range(100, 30000, 300)
acc = []
for i in data_num:
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
    predictions = knn_reg.fit(offline_rss[:i, :], offline_location[:i, :]).predict(rss)
    acc.append(accuracy(predictions, trace) / 100)
    
# 绘制训练数据量与accuracy的曲线
import matplotlib.pyplot as plt
plt.plot(data_num, acc, '-o', linewidth=2.0)
plt.xlabel("data number")
plt.ylabel("accuracy (m)")
plt.grid(True)
plt.savefig("../pictures/file2.png")
plt.show()








