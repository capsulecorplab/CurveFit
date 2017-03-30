#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# http://www.walkingrandomly.com/?p=5215
# http://www.walkingrandomly.com/images/python/least_squares/Python_nonlinear_least_squares.pdf

xdata = np.array([-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9])
ydata = np.array([0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001])

#plt.plot(xdata,ydata,'*')
#plt.xlabel('xdata')
#plt.ylabel('ydata')
#plt.show()

def func(x, p1,p2):
	return p1*np.cos(p2*x) + p2*np.sin(p1*x)

popt, pcov = curve_fit(func, xdata, ydata,p0=(1.0,0.2))

p1 = popt[0]
p2 = popt[1]
residuals = ydata - func(xdata,p1,p2)
fres = sum(residuals**2)

print 'popt', popt
print 'pcov', pcov
print 'p1', p1
print 'p2', p2
print 'residuals', residuals
print 'fres', fres

curvex=np.linspace(-2,3,100)
curvey=func(curvex,p1,p2)
plt.plot(xdata,ydata,'*')
plt.plot(curvex,curvey,'r')
plt.xlabel('xdata')
plt.ylabel('ydata')

plt.show()

