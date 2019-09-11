import numpy as np

def DHtoTF(alpha, a, d, theta, unit='rad'):
    if unit == 'deg':
        alpha = alpha*np.pi/180.
        theta = theta*np.pi/180.

    T = np.matrix([
                [np.cos(theta),              -np.sin(theta),                0.,             a],
                [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                [0.,                          0.,                           0.,             1.]
                 ])
    return T


if __name__ == "__main__":
    T = DHtoTF(0, 0, 0, 30, 'deg')
    p = [1,0,0,0]
    print np.matmul(T,p)
