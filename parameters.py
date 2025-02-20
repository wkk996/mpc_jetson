import numpy as np
import matplotlib.pyplot as plt

from utils import euler_to_quaternion

DT = 0.1#SANMPC=CBF-MPC=0.1,AMPC=0.01
N =8  #SANMPC=8#CBF-MPC=30
USE_RBF = False
#FILE_NAME="results/data_{}.txt".format(USE_RBF)
FILE_NAME="test/data_test.txt"
#FILE_NAME_BASE = "results/data_test_1w_2.12V1"
FILE_NAME_BASE = "test/data_test_jetson"
def reference(total_time, type=0):
    num = int(total_time/DT)+1
    t = np.linspace(0, total_time, num)
    if type == 0:
        xr = -0.5*np.sin(2*np.pi*t/30)
        yr =  0.5*np.sin(2*np.pi*t/20)

        dxr = -0.5*2*np.pi/30*np.cos(2*np.pi*t/30)
        dyr =  0.5*2*np.pi/20*np.cos(2*np.pi*t/20)
        thetar = 2*np.arctan2(np.sqrt(dxr**2+dyr**2)+dxr, dyr)
    return t, xr, yr, thetar

def get_reference(time, x0, n, dt):
    x_ref = [x0]
    for i in range(n):
        t = time + dt*i
        
        xr = -5.*np.sin(2*np.pi*t/30)
        yr =  5.*np.sin(2*np.pi*t/20)

        dxr = -5.*2*np.pi/30*np.cos(2*np.pi*t/30)
        dyr =  5.*2*np.pi/20*np.cos(2*np.pi*t/20)
        thetar = 2*np.arctan2(np.sqrt(dxr**2+dyr**2)+dxr, dyr)
        
        p_ref = np.array([xr, yr, 1.0])
        q_ref = euler_to_quaternion(0, 0, thetar)
        v_ref = np.zeros(3)
        w_ref = np.zeros(3)
        x_ref.append(np.concatenate([p_ref, q_ref, v_ref, w_ref]))
    return np.array(x_ref).T


def get_reference_fixed(time, x0, n, dt, target=np.array([10.0, 10.0,10.0])):

    x_ref = [x0]
    q_ref = euler_to_quaternion(0, 0, 0)  
    v_ref = np.zeros(3)
    w_ref = np.zeros(3)
    for i in range(n):
        x_ref.append(np.concatenate([target, q_ref, v_ref, w_ref]))
    return np.array(x_ref).T
