import cv2
import numpy as np
import matplotlib.pyplot as plt

distortion = np.array([
    -2.71969672e-01,    # k1
    1.12979477e-01,     # k2
    6.30516626e-06,     # p1
    2.62608080e-04,     # p2
    -2.82170416e-02,    # k3
])
k1, k2, p1, p2, k3 = distortion

def brown_rmax(k1, k2, k3):
    roots = np.roots([7*k3, 5*k2, 3*k1, 1])
    real = np.sqrt(np.abs(roots[np.isreal(roots)]))
    if len(real)>0:
        rmax = min(real[real>0])
    else:
        rmax = np.inf
    return rmax

scale = 3
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(scale*3,scale*1))
r = np.linspace(0, 2.0, 100)
r_s = r*r
# First order
R_1 = (1 + k1*r_s) * r
dR_1 = 1 + 3*k1*r_s
rmax_1 = brown_rmax(k1=k1, k2=0, k3=0)
ax1.vlines(x=rmax_1, ymin=-4, ymax=4, colors='green', label=f"rmax={rmax_1:.3f}")
ax1.set_title('First Order')
ax1.plot(r, R_1, label="D(r)")
ax1.plot(r, dR_1, label="D'(r)")
ax1.set_xlim([0, 2])
ax1.set_ylim([-4, 4])
ax1.legend()
ax1.grid()
# Second order
R_2 = (1 + k1*r_s + k2*r_s*r_s) * r
dR_2 = 1 + 3*k1*r_s + 5*k2*r_s*r_s
rmax_2 = brown_rmax(k1=k1, k2=k2, k3=0)
ax2.vlines(x=rmax_2, ymin=-4, ymax=4, colors='green', label=f"rmax={rmax_2:.3f}")
ax2.set_title('Second Order')
ax2.plot(r, R_2, label="D(r)")
ax2.plot(r, dR_2, label="D'(r)")
ax2.set_xlim([0, 2])
ax2.set_ylim([-4, 4])
ax2.legend()
ax2.grid()
# Third order
R_3 = (1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s) * r
dR_3 = 1 + 3*k1*r_s + 5*k2*r_s*r_s + 7*k3*r_s*r_s*r_s
rmax_3 = brown_rmax(k1=k1, k2=k2, k3=k3)
ax3.vlines(x=rmax_3, ymin=-4, ymax=4, colors='green', label=f"rmax={rmax_3:.3f}")
ax3.set_title('Third Order')
ax3.plot(r, R_3, label="D(r)")
ax3.plot(r, dR_3, label="D'(r)")
ax3.set_xlim([0, 2])
ax3.set_ylim([-4, 4])
ax3.legend()
ax3.grid()

fig.tight_layout()
plt.show()
