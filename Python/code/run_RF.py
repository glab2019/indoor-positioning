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

from sklearn.ensemble import RandomForestClassifier
labels = np.round(offline_location[:, 0]/100.0) * 100 + np.round(offline_location[:, 1]/100.0)
estimator = RandomForestClassifier(n_estimators=20, max_features=None, max_depth=20) # 内存受限，tree的数量有点少
estimator.fit(offline_rss, labels)
predict_labels = estimator.predict(rss)
x = np.floor(predict_labels/100.0)
y = predict_labels - x * 100
predictions = np.column_stack((x, y)) * 100
acc = accuracy(predictions, trace)
print ("accuracy: ", acc/100, 'm')

#中间写上代码块
end=time.time()
print('Running time: %s Seconds'%(end-start))