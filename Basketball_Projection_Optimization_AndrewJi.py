'''
#  Multi-Variable Calculus Project
#  Basketball Free Throw Projection Opitimization
#  by Andrew Ji
#  11/26/2024
'''


import numpy as np
import matplotlib.pyplot as plt
import math as math

H = 3.05  # Basketball rim height 10 ft = 10*12*25.4/1000
h = 1.9     # People's height, does not consider if the person will bend knee or not, if they throw the ball over their head
d = 4.6   # The distance between person and rim center 15 ft = 15*12*25.4/1000
g = 9.81  # Gravity
r_rim = 0.23 # Rim radius
r_ball = 0.12 # Basketball radius


def gradient(estimate, g, H, h,d):
    grad = np.zeros_like(estimate)
    v0=estimate[0]
    theta0 = estimate[1]
    f=(v0* math.cos(theta0) * (v0*math.sin(theta0)+np.sqrt((v0*math.sin(theta0))**2-2*9.81*(3.05-2)))/9.81-d)
    grad[0] =( 2*v0*math.cos(theta0)*math.sin(theta0)/g + math.cos(theta0) * np.sqrt((v0*math.sin(theta0))**2-2*g*(H-h))/g + v0* math.cos(theta0) *math.sin(theta0)**2*v0/np.sqrt((v0*math.sin(theta0))**2-2*g*(H-h))/g)*f/abs(f)
    grad[1] = (v0**2*math.cos(2*theta0)/g - v0*math.sin(theta0)*np.sqrt((v0*math.sin(theta0))**2-2*g*(H-h))/g + v0*math.cos(theta0) * v0**2*math.sin(theta0) *math.cos(theta0)/np.sqrt((v0*math.sin(theta0))**2-2*g*(H-h))/g)*f/abs(f)
   
    return grad

def minimize_func(func, fist_estimate, g,H,h,d,k=0.001, num=10000, tol=1e-4):
    """
    Minimizes a function func using gradient descent.
    Parameters:
    func: The function to minimize.
    fist_estimate: The initial guess.
    k: The convergence rate.
    num: The number of iterations.
    tol: The tolerance for convergence.
    """
    estimate = np.array(first_estimate)
    temp = np.zeros((num,3))
    for i in range(num):
        grad = gradient(estimate, g, H, h,d)
        estimate = estimate - k * grad
        if func(estimate,g, H, h, d) < tol:
            break
        temp[i,0]=estimate[0]
        temp[i,1]=estimate[1]*180/3.1415926
        temp[i,2]=func(estimate,g, H, h, d)
    return estimate, func(estimate,g, H, h, d), i

def func(estimate,g, H, h, d):
    v0=estimate[0]
    theta0 = estimate[1]
    return abs(v0* math.cos(theta0) * (v0*math.sin(theta0)+np.sqrt((v0*math.sin(theta0))**2-2*g*(H-h)))/g-d)


first_estimate = [7,50/180*3.1415926]
# Need to modify the convergence rate and number of iteration for better convergence.
k = 0.001
num = 10000
result = minimize_func(func, first_estimate,  g, H, h,d, k, num)
optimal_v0 = result[0][0]
optimal_theta0 = result[0][1]*180/3.1415926
print(optimal_v0,optimal_theta0)


'''
Numerical simulation to find the possible initial angle and velocity 
that around optimal angles and velocities which can also make the goal

'''
output = np.zeros((40000,7))
theta0 = 40/180*3.1415926 
n=0
for i in range (0,200):   
    v0 = 7.2
    for j in range (0, 200):        
        y=H
        x = v0* math.cos(theta0) * (v0*math.sin(theta0)+np.sqrt((v0*math.sin(theta0))**2-2*g*(H-h)))/g       
        if ((x-(d-r_rim))  > r_ball) & (((d+r_rim)-x)  > r_ball) :
            goal = 1
        else:
            goal = 0
        output[n,0]=v0
        output[n,1]=theta0*180/3.1415926
        output[n,2]=goal
        output[n,3]=x     
        output[n,4]=(d-r_rim)+ r_ball 
        output[n,5]=(d+r_rim) - r_ball 
        output[n,6]=r_ball
        n=n+1
        v0 = v0 + 0.005
    theta0 = theta0 + 0.15/180*3.1415926  
a = output[0:200,0]  
b = output[:,1][output[:,0]==output[0,0]] 
c = np.transpose(output[:,2].reshape(200,200))  
plot = plt.figure()
plt.contour(b,a,c)  
plt.ylabel('velocity (m/s)', fontsize =30)
plt.xlabel('Theta (o)', fontsize =30)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.show()

 
'''
Plot the basketball Project with the optimial angle and velocity
'''
time=(2*optimal_v0*math.sin(optimal_theta0*(3.1415926/180)))/g
t =np.linspace(0,time)
vx = optimal_v0*math.cos(optimal_theta0*(3.1415926/180))
vy = optimal_v0*math.sin(optimal_theta0*(3.1415926/180))
sx = vx*t
sy=h+vy*t-0.5*g*t**2
plot2 = plt.figure()
plt.plot(sx[0:41],sy[0:41],'o',markersize = 10)
plt.xlabel('x', fontsize =30)
plt.ylabel('y', fontsize =30)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.show()

    



