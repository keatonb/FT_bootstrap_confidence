#!/usr/bin/env python 

#Generate list of boxcar smoothing functions and call ran_trials_2 with them.

import numpy as np
from astropy.convolution import convolve, Box1DKernel
from ran_trials_2 import ran_trials

#create list of functions
metrics = []

#define range of boxcar widths
widths=[510] #1 muHz
'''
for w in widths:
    metrics.append(lambda x,w=w: convolve(x, Box1DKernel(w)))
    metrics.append(lambda x,w=w: convolve(x, Box1DKernel(w)))
'''
metrics.append(lambda x,w=510: convolve(x, Box1DKernel(w)))
metrics.append(lambda x,w=510: convolve(x**2., Box1DKernel(w)))

#run the trials
ran_trials('exampledata.dat',nperm = 10,hifac=0.2,metrics=metrics)

#plot the results:

