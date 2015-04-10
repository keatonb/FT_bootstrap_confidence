#!/usr/bin/env python 

import numpy as np
from numpy.random import permutation
from scipy.signal import argrelmax
import datetime
import lomb

# This function just makes sure that the length of arrays returned 
# by the lomb.fasper() function have the same dimension.

def ft_fix(x0,y0):
    nftx = len(x0)
    nfty = len(y0)
    fx = x0
    fy = y0
    if (nftx != nfty):
        if nftx > nfty:
            fx = x0[0:nfty]
            nft = nfty
        else:
            fy = y0[0:nftx]
            nft = nftx
    else:
        nft = nftx
    return fx, fy, nft

# Input Kepler flux/light curve file
# file0='Q11-Q17lc_nooutliers.dat'
file0='tmp.dat'

# If FT_orig = 1, then it will first compute and store an FT
# of the original data set
FT_orig = 1

# Number of random shufflings of data to make, i.e., 1000 or 10 000
nperm = 4

ofile0 = file0[0:-4] 

# All the time() calls is just to obsess about how long different
# parts of the computations take

# Read in origianl file
t0 = datetime.datetime.now()
print '\nReading in',file0,'...'
vars=np.loadtxt(file0)
t1 = datetime.datetime.now()
print 'Elapsed time: ',t1-t0,'\n'

t  = vars[:,0]
lc = vars[:,1]

# Force mean of light curve to be zero, and shift the time such that
# the "mean" time is also zero
t = t - 0.5*(max(t)-min(t))
lc = lc - np.mean(lc)

# ofac is the oversampling factor, and hifac sets the upper limit of the
# frequency range. Thus hifac=0.5 means the periodogram will be computed
# from a frequency of zero up to 0.5*nyquist_frequency
ofac=10.
hifac=1.0

# Compute FT (periodogram, actually) of unmodified data
if FT_orig == 1:

    print 'Computing fast Lomb-Scargle periodogram of unmodified data.'
    print 'This could take a while...'
    fx0,fy0, amp0, nout, jmax, fap_vals, amp_probs = lomb.fasper(t,lc, ofac, hifac)
    t2 = datetime.datetime.now()
    print 'Elapsed time: ',t2-t1,'\n'

    #print fap_vals
    #print amp_probs
    fx0, fy0, nft  = ft_fix(fx0,fy0)
    fx0, amp0, nft = ft_fix(fx0,amp0)

    outarr = np.zeros((nft,3))
    outarr[:,0] = fx0
    outarr[:,1] = fy0
    outarr[:,2] = amp0
    ofile1 = ofile0 + '.ft'
    nvals = len(fap_vals)
    stsig = ''
    for i in np.arange(nvals):
        stsig = stsig + '  {0:f}    {1:f}\n'.format(fap_vals[i],amp_probs[i])
    
        head = 'Significance levels ( {0} ) for amplitude from formal periodogram criteria: '.format(nvals) + '\n    FAP       amplitude\n' + stsig + 'freq (hz)  power (normed) amplitude'
    print 'Writing out FT to {0}...'.format(ofile1)
    np.savetxt(ofile1,outarr,header=head,fmt='%e')
    t3 = datetime.datetime.now()
    print 'Elapsed time: ',t3-t2,'\n'

head0 = 'Generated from {0} using lomb.fasper() with ofac= {1}, hifac= {2}, npts= {3}'.format(file0,ofac,hifac,len(t))
pmaxvals = []
amaxvals = []

print '\nRandomly shuffling data',nperm,'times...\n'

for i in np.arange(nperm):
    t3 = datetime.datetime.now()
    print 'Permutation {0}...'.format(i)
    lcper = permutation(lc)
    fx0,fy0, amp0, nout, jmax, fap_vals, amp_probs = lomb.fasper(t,lcper, ofac, hifac)
    t4 = datetime.datetime.now()
    pmaxvals.append(fy0[jmax])
    amaxvals.append(amp0[jmax])
    print 'Elapsed time: ',t4-t3,'\n'


print 'Finished shuffling data',nperm,'times\n'

# print amaxvals
n=len(amaxvals)
amaxvals = np.array(amaxvals)
powvals = amaxvals**2
outarr = np.zeros((n,2))
outarr[:,0] = amaxvals
outarr[:,1] = powvals
ofile2 = ofile0 + '.hist'
print 'Writing maximum values to',ofile2

# Write out the peak amplitude and power for each shuffled data set 
head = 'Maximum power (and amplitude) values from {0} randomly shuffled trials of {1}\n  Amp         Power '.format(nperm,file0)
np.savetxt(ofile2,outarr,header=head,fmt='%e')

print 'Total elapsed time: ',t4-t0,'\n'
