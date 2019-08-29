import numpy as np
import rospy
import matplotlib.pyplot as plt

def LSPB(q0,qf,t,tf,v):
    tb = (q0-qf+v*tf)/v
    a = v/tb
    vel_limit = (qf-q0)/tf
    print vel_limit
    if vel_limit > v or v > 2*vel_limit:
        return []

    if 0 <= t and t < tb:
        q = q0 + a/2*t*t
    elif tb < t and t <= tf-tb:
        q = (qf+q0-v*tf)/2 + v*t
    elif tf-tb < t and t <= tf:
        q = qf-a*tf*tf/2+a*tf*t-a/2*t*t
    else:
        return []
    return q

if __name__ == '__main__':
    tf = 3.0
    tb = 0.8
    t = np.linspace(0,tf,1000)
    q = np.zeros(np.shape(t))
    q0 = 0.02
    qf = 0.1
    v = 0.04
    for i in range(np.size(t)):
        q[i] = LSPB(q0, qf, t[i], tf, v)
    plt.plot(t,q)
    plt.show()
