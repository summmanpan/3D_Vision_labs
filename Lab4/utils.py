import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def line_draw(line, canv, size):
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]

    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]

    w, h = size

    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    canv.line([beg, end], width=8)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)

def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, w, h, fig, legend, scale=1):
    
    o = optical_center(P)
    
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return

# Compute the Image of the Absolute Conic
def compute_img_abs_conic(N,H, start_N = 1):
    A = np.zeros((2*N, 6))

    for n in range(start_N, N + start_N):
        i = n-start_N
        h = H[i]
        A[2*i-1, :] = np.array([h[0,0]*h[0,1], h[0,0]*h[1,1] + h[1,0]*h[0,1], h[0,0]*h[2,1] + h[2,0]*h[0,1], h[1,0]*h[1,1], h[1,0]*h[2,1] + h[2,0]*h[1,1], h[2,0]*h[2,1] ])
        A[2*i, :] = np.array([h[0,0]**2, 2*h[0,0]*h[1,0], 2*h[0,0]*h[2,0], h[1,0]**2, 2*h[1,0]*h[2,0], h[2,0]**2])
        A[2*i, :] -= np.array([h[0,1]**2, 2*h[0,1]*h[1,1] , 2*h[0,1]*h[2,1], h[1,1]**2, 2*h[1,1]*h[2,1], h[2,1]**2])
    
    u,s,vt = np.linalg.svd(A);
    # linalg gives the v matrix transposed
    # we need to transpose it again to obtian x in the last column
    v = np.transpose(vt)
    x = v[:,5]# last colum of the v

    w = np.zeros([3,3])
    w[0,:] = np.asarray([x[0],x[1],x[2]])
    w[1,:] = np.asarray([x[1],x[3],x[4]])
    w[2,:] = np.asarray([x[2],x[4],x[5]])

    # normalize by dividing all components by the last one
    w /= w[2,2]

    return w


def compute_K(w):
    w_inv = np.linalg.inv(w)
    L = np.linalg.cholesky(w_inv)
    K = L.T
    return K