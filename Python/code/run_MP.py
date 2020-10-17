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

from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(hidden_layer_sizes=(100, 100))
clf.fit(offline_rss, offline_location)
predictions = clf.predict(rss)
acc = accuracy(predictions, trace)
print ("accuracy: ", acc/100, "m")

#中间写上代码块
end=time.time()
print('Running time: %s Seconds'%(end-start))