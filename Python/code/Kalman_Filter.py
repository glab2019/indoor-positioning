# 导入数据
import numpy as np
import scipy.io as scio
from sklearn import neighbors
import warnings
warnings.filterwarnings('ignore')
offline_location, offline_rss = np.loadtxt('../data/train_location.txt'), np.loadtxt('../data/train_rss.txt')
trace, rss = np.loadtxt('../data/trace_location.txt')[0:1000, :], np.loadtxt('../data/trace_rss.txt')[0:1000, :]


# 定位准确度
def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

# knn回归
from sklearn import neighbors
knn_reg = neighbors.KNeighborsRegressor(40, weights='uniform', metric='euclidean')
knn_predictions = knn_reg.fit(offline_rss, offline_location).predict(rss)
acc = accuracy(knn_predictions, trace)
print ("accuracy: ", acc/100, "m")

# 对knn定位结果进行卡尔曼滤波

from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
def kalman_tracker():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.
    # 状态转移矩阵
    tracker.F = np.array([[1, dt, 0,  0], 
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    # 用filterpy计算Q矩阵
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    # tracker.Q = block_diag(q, q)
    tracker.Q = np.eye(4) * 0.01
    # tracker.B = 0
    # 观测矩阵
    tracker.H = np.array([[1., 0, 0, 0],
                          [0, 0, 1., 0]])
    # R矩阵
    tracker.R = np.array([[4., 0],
                          [0, 4.]])
    # 初始状态和初始P
    tracker.x = np.array([[7.4, 0, 3.3, 0]]).T 
    tracker.P = np.zeros([4, 4])
    return tracker
tracker = kalman_tracker()
zs = np.array([np.array([i]).T / 100. for i in knn_predictions]) # 除以100，单位为m
mu, cov, _, _ = tracker.batch_filter(zs) # 这个函数对一串观测值滤波
knn_kf_predictions = mu[:, [0, 2], :].reshape(1000, 2)
acc = accuracy(knn_kf_predictions, trace / 100.)
print ("accuracy: ", acc, "m")

import matplotlib.pyplot as plt
x_i = range(220, 280)
tr, = plt.plot(trace[x_i, 0] / 100., trace[x_i, 1] / 100., 'k-', linewidth=3)
kf, = plt.plot(knn_kf_predictions[x_i, 0], knn_kf_predictions[x_i, 1], 'b-')
knn_ = plt.scatter(knn_predictions[x_i, 0] / 100., knn_predictions[x_i, 1] / 100.)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend([tr,  kf, knn_], ["real trace", "kf", "knn"])
plt.savefig("../pictures/file3.png")
plt.show()
