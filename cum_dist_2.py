#!/usr/bin/env python 
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
from scipy import interpolate
import numpy as np
import datetime
import lomb
from astroML.density_estimation import histogram as hst

from matplotlib.ticker import MaxNLocator, AutoMinorLocator
my_locator = MaxNLocator(10)
minor2a = AutoMinorLocator(2)
minor2b = AutoMinorLocator(2)
minor4 = AutoMinorLocator(4)

# Set up axes and plot some awesome science

import utils

from matplotlib import rc_file
#rc_file('/Users/mikemon/.matplotlib/matplotlibrc_apj')
rc_file('matplotlibrc_apj')

# Read in file of maximum peaks from shuffled data sets
infile='Q11-Q17lc_nooutliers_tot.hist'

print '\nReading in peak values...'

vars=np.loadtxt(infile)
amp   = vars[:,0]
pow   = vars[:,1]

amp = 1.e+3 * amp  # put in mmas

print 'Largest peak is',max(amp)

# "Fancy" histogram from astroML
hvals, bins = hst(amp,bins='freedman')

# "Working Man's" histogram from numpy
# hvals, bins = np.histogram(amp,bins=75)

nbins = len(bins)
xvals = 0.5*(bins[0:-1]+bins[1:])
hcum = np.cumsum(hvals)
hcum2 = hcum/float(max(hcum))

hcum3 = hcum2[hvals > 0]
xvals3 = xvals[hvals > 0]
f = interpolate.interp1d(hcum3, xvals3)
cvals = np.array([0.683, 0.955, 0.997])
xcvals = f(cvals)

fig=figure(1,figsize=(5,4))
#fig=figure(1)

ax1 = fig.add_subplot(2,1,1)
plot(xvals,hvals,drawstyle='steps-mid')
utils.padlim(ax1)
ylabel(r'$N$')
#plot(yl,xl,'r--')

ax2 = fig.add_subplot(2,1,2,sharex=ax1)
plot(xvals,hcum2)
utils.padlim(ax2)

ax1.yaxis.set_minor_locator(minor2a)
ax1.xaxis.set_minor_locator(minor4)
ax2.yaxis.set_minor_locator(minor2b)

for i in np.arange(len(cvals)):
    xl = np.array([-1.e6,1.e6])
    yl = xcvals[i] + 0.*xl
    ax1.plot(yl,xl,'r--')
    ax2.plot(yl,xl,'r--')
    #lab = '{0:.3f}'.format(xcvals[i])
    lab = '{0:.3f}'.format(cvals[i])
    ax2.text(xcvals[i]+0.0005, 0.3,lab,fontsize=9,rotation=90,color='red')

xlabel(r'Amplitude (mma)')
ylabel(r'Cumulative Distribution')
fig.subplots_adjust(hspace=0.0001)

fileout = 'cum_dist.pdf'
savefig(fileout,bbox_inches='tight')

print '\nPlot saved in',fileout,'\n'
