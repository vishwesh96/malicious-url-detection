import sys
sys.path.append('/home/navneet/Downloads/libsvm-3.21/python')
from svmutil import *

y,x = svm_read_problem('../url_svmlight/Day0.svm')
'''loading the training data '''
problem = svm_problem(y[:8000],x[:8000])
param = svm_parameter('-t 3 -c 10000  -h 0')
m = svm_train(problem, param)
p_labels, p_acc, p_vals = svm_predict(y[8000:],x[8000:],m)
print p_acc

