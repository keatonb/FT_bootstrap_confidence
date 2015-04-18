#!/usr/bin/env python 

import numpy as np
from numpy.random import permutation
from scipy.signal import argrelmax
import datetime
import inspect
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


def ran_trials(file0,nperm = 4,ofac=10.,lowfreq=0,hifreq=1e9,hifac=1.0,metrics = [lambda x: x, lambda x: x**2.]):
    '''
    Determine bootstrap significance threshold for FT of given data.
        
    Keyword arguments:
    file0 -- filename of data to be shuffled and FTed (2 columns e.g. time,flux)
    nperm -- number of random shufflings of data to make, i.e., 1000 or 1e4 (default 4, for speed)
    ofac -- the FT oversampling factor (default 10.)
    lowfreq -- lower limit of frequency range (in Hz; doesn't save time but helps with interpretation).
    lowfreq -- upper limit of frequency range (in Hz; doesn't save time but helps with interpretation).
    hifac -- upper limit of the *calculated* frequency range (as fraction of the nyquist frequncy, [0-1]; default 1)
    metrics -- list of passed (lambda) functions that are evaluated along randomized FTs, the max values being recorded. 
    (default power and amplitude)
    
    Outputs:
    filename.ft -- FT of original data, and ft processed by all metrics
    filename.hist -- record of highest values of computed metric for each run.
    '''
    
    # If FT_orig = 1, then it will first compute and store an FT
    # of the original data set
    FT_orig = 1
    
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
    
    
    # Compute FT (periodogram, actually) of unmodified data
    if FT_orig == 1:
    
        print 'Computing fast Lomb-Scargle periodogram of unmodified data.'
        print 'This could take a while...'
        fx0,fy0, amp0, nout, jmax, fap_vals, amp_probs,df= lomb.fasper(t,lc, ofac, hifac)
        t2 = datetime.datetime.now()
        print 'Elapsed time: ',t2-t1,'\n'
    
        #print fap_vals
        #print amp_probs
        #fx0, fy0, nft  = ft_fix(fx0,fy0)
        fx0, amp0, nft = ft_fix(fx0,amp0)
        
        outarr = np.zeros((nft,2+len(metrics)))
        outarr[:,0] = fx0        
        outarr[:,1] = amp0
        
        #run all metrics on ft
        for i,m in enumerate(metrics): outarr[:,2+i] = m(amp0,df)

        ofile1 = ofile0 + '.ft'
        nvals = len(fap_vals)
        stsig = ''
        for i in np.arange(nvals):
            stsig = stsig + '  {0:f}    {1:f}\n'.format(fap_vals[i],amp_probs[i])
        
        head = 'Significance levels ( {0} ) for amplitude from formal periodogram criteria: '.format(nvals) + '\n    FAP       amplitude\n' + stsig + 'freq(hz)   amplitude'
        for i in range(len(metrics)): head+= '     metric'+str(i)
        print 'Writing out FT to {0}...'.format(ofile1)
        np.savetxt(ofile1,outarr,header=head,fmt='%e')
        t3 = datetime.datetime.now()
        print 'Elapsed time: ',t3-t2,'\n'
    
    head0 = 'Generated from {0} using lomb.fasper() with ofac= {1}, hifac= {2}, npts= {3}'.format(file0,ofac,hifac,len(t))
  #  pmaxvals = []
  #  amaxvals = []
    maxvals = []    
    medvals = []
    
    print '\nRandomly shuffling data',nperm,'times...\n'
    
    for i in np.arange(nperm):
        t3 = datetime.datetime.now()
        print 'Permutation {0}...'.format(i)
        lcper = permutation(lc)
        fx0,fy0, amp0, nout, jmax, fap_vals, amp_probs, df = lomb.fasper(t,lcper, ofac, hifac)
        t4 = datetime.datetime.now()
        thesemaxvals=[]
        thesemedvals=[]
        for m in metrics: 
            processed=m(amp0,df)
            inrange=processed[np.where((fx0 > lowfreq) & (fx0 < hifreq))]
            thesemaxvals.append(np.max(inrange))
            thesemedvals.append(np.median(inrange))
#        pmaxvals.append(np.max(metrics[0](amp0)))
#        amaxvals.append(np.max(metrics[1](amp0)))
        maxvals.append(thesemaxvals)
        medvals.append(thesemedvals)
        print 'Elapsed time: ',t4-t3,'\n'
    
    
    print 'Finished shuffling data',nperm,'times\n'
    
    # print amaxvals
    #n=len(maxvals)
    maxvals = np.array(maxvals)
    medvals = np.array(medvals)
    ofile2 = ofile0 + '.hist'
    print 'Writing maximum,median values to',ofile2
    
    # Write out the peak amplitude and power for each shuffled data set 
    head = 'Maximum, then median values from {0} randomly shuffled trials of {1}\n'.format(nperm,file0)
    head += 'Values from the following function definitions:\n'
    for i,m in enumerate(metrics): head += str(i+1)+ inspect.getsource(m) + '\n'
    np.savetxt(ofile2,np.concatenate((maxvals,medvals),1),header=head,fmt='%e')
    
    print 'Total elapsed time: ',t4-t0,'\n'
    
