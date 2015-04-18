#!/usr/bin/env python 

#Generate list of boxcar smoothing functions and call ran_trials_2 with them.

#import numpy as np
from astropy.convolution import convolve, Box1DKernel
from ran_trials_2 import ran_trials
#import datetime

#create list of functions
metrics = []

#define boxcar width
w=.5e-6 #Hz
metrics.append(lambda x,df,w=.5e-6: convolve(x, Box1DKernel(w/df)))
metrics.append(lambda x,df,w=1e-6: convolve(x, Box1DKernel(w/df)))
metrics.append(lambda x,df,w=.5e-6: convolve(x**2., Box1DKernel(w/df)))
metrics.append(lambda x,df,w=1e-6: convolve(x**2., Box1DKernel(w/df)))


'''
for w in widths:
    metrics.append(lambda x,fper,w=w: convolve(x, Box1DKernel(w)))
    metrics.append(lambda x,fper,w=w: convolve(x, Box1DKernel(w)))
'''


#metrics.append(lambda x,y: smoothed(x,y,w=1.0e-6))
#metrics.append(lambda x,y: smoothedsquared(x,y,w=0.5e-6))
#metrics.append(lambda x,y: smoothedsquared(x,y,w=1.0e-6))


#run the trials
ran_trials('exampledata.dat',nperm = 10,lowfreq=0.0006,hifreq=0.00145,hifac=0.2,metrics=metrics)

#plot the results:

