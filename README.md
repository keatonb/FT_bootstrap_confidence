# FT_bootstrap_confidence
Code for calculating false alarm probabilities for a generic local measurement in a Fourier transform.

Most of this code (initial commit) was written by Mikemon (https://github.com/mmmikemon).  I put the code here to modify to my liking.  Specifically, I am generalizing the measurement that I calculate false alarm probablilities for to any local measurement in the Fourier transform rather than just the amplitude at every single frequency.  This will be useful in cases where we are searching for signals with power extending over a range of frequencies, e.g., when there is inherent phase/amp/freq modulation in the observed system.
