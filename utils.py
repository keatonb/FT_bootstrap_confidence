import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from scipy import signal
import mesa as ms
import string as str
#import gyre


def padlim(ax,aspect=1.618,pad=0.05):
    """
    Set limits on current axis object in a SM-like manner
    """
    lims = ax.dataLim
    xint = lims.intervalx
    yint = lims.intervaly
    deltax = xint[1]-xint[0]
    xpad = pad/aspect
    ypad = pad
    dx = xpad * deltax
    x0 = xint[0] - dx
    x1 = xint[1] + dx
    xlim(x0,x1)
    deltay = yint[1]-yint[0]
    dy = ypad * deltay
    y0 = yint[0] - dy
    y1 = yint[1] + dy
    ylim(y0,y1)

def padylim(ax,aspect=1.618,pad=0.05):
    """
    Set limits on current axis object in a SM-like manner
    """
    lims = ax.dataLim
    yint = lims.intervaly
    ypad = pad
    deltay = yint[1]-yint[0]
    dy = ypad * deltay
    y0 = yint[0] - dy
    y1 = yint[1] + dy
    ylim(y0,y1)

def padxlim(ax,aspect=1.618,pad=0.05):
    """
    Set limits on current axis object in a SM-like manner
    """
    lims = ax.dataLim
    xint = lims.intervalx
    yint = lims.intervaly
    deltax = xint[1]-xint[0]
    xpad = pad/aspect
    dx = xpad * deltax
    x0 = xint[0] - dx
    x1 = xint[1] + dx
    xlim(x0,x1)

def reversexy():
    """
    Reverse x and y axes

    """
    xlim(xlim()[::-1])
    ylim(ylim()[::-1])

def reversex():
    """
    Reverse x axis

    """
    xlim(xlim()[::-1])

def reversey():
    """
    Reverse y axis

    """
    ylim(ylim()[::-1])

def reverse(x):
    """
    Reverse the order of a vector
    """
    y=x[::-1]
    return y

def readtxt(file,ncol,spos) :
    """
    This is a special version of the loadtxt command modified to read a
    file containing columns of floats and strings.

    Parameters
    ----------
    file : name of input file
    ncol : the total number of columns of the file
    spos : a vector containing the column numbers whose entries are strings
           Example: spos = np.array([3,4,7]) means that the third, fourth, and
           seventh columns are to be treated as strings, and that all other
           columns are to be considered floats

    output : This routine outputs the array of floats followed by the array
    of string-valued columns. For example the command

             var2,names=utils.readtxt('filename',11,[3])

    reads 11 columns from 'filename' and assumes that all columns are floats
    except for the fourth column (the first column has an index of 0). It then
    returns the floats in 'var2' and the strings in 'names'.
    """
    nl=np.size(spos)
    dtype = [('data',np.float)]*ncol
    for i in range(ncol) :
        name = 'data' + str(i)
        dtype[i]= (name,np.float)
    for i in range(nl) :
        name = 'name' + str(i)
        ix=spos[i]
        dtype[ix]= (name,'S50')
    vals=np.loadtxt(file,comments='#',dtype=dtype)
    vals = vals.view(np.dtype(dtype))
    ifirst=0
    for i in range(ncol) :
        ibreak=0
        for j in range(nl) :
            ix=spos[j]
            if i==ix :
                ibreak=1
        if ibreak==1 :
            continue
        name = 'data' + str(i)
        vout = vals[name]
        if ifirst==0 :
            vout2 = vout
            ifirst=1
        else :
            vout2 = np.column_stack((vout2,vout))
    for i in range(nl) :
        name = 'name' + str(i)
        vout = vals[name]
        if i==0 :
            vout3 = vout
        else :
            vout3 = np.column_stack((vout3,vout))
    return vout2,vout3

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)

def param_get(path0,hnum):
    import re
    cfile='/controls{0}.data'.format(hnum)
    cfile= path0 + cfile
    parvals = ['MIXING_LENGTH_ALPHA','USE_HENYEY_MLT','MLT_OPTION']
    f = open(cfile)
    con=f.readlines()
    for par in con:
        if par != con[0]:
            par=par.replace("="," = ")
            par=par.replace(",","")
            par=par.replace("\"","")
            par=par.replace("\n","")
            for pvals in parvals:
                if re.search(pvals,par):
                    p1=par.split()
                    p0=p1[0]
                    alpha=p1[2]
                    #print p0, ' = ',alpha
                    if pvals == parvals[0]:
                        s1=alpha
                    if pvals == parvals[2]:
                        s0=alpha
                    if pvals == parvals[1]:
                        s2=alpha
    return s0,s1,s2

def filter(str):
    str=str.replace("="," = ")
    str=str.replace(",","")
    str=str.replace("\"","")

def pigin_write(tmpfile, mass, teff, logg, MLT='ML2', alpha=1.0, X=1.00, Z=0.00):
    mtheories=['ML1','ML2','Mihalas','Henyey']
    nm=np.size(mtheories)
    for i in range(nm):
       if MLT == mtheories[i]:
           imlt=i+1
           break
       if i == nm-1:
           print MLT, " not an allowed version of MLT"
           print "stop"
           exit(1)
    f=open(tmpfile,'w')
    str1="{0} {1} {2}\n".format(mass,teff,logg)
    f.write(str1)
    str2="{0} {1} \n".format(X,Z)
    f.write(str2)
    f.write("1 1\n")
    str3="{0} \n".format(imlt)
    f.write(str3)
    str4="{0}\n".format(alpha)
    f.write(str4)
    f.write("0\n")
    f.close()

def mesa_histplot(s,vars):
    xvar=vars[0]
    yvar=vars[1]

    x  = s.get(xvar)
    n=np.size(vars)
    for i in np.arange(1,n):
        yvar=vars[i]
        y1 = s.get(yvar)
        plot(x,y1)
    xlabel(xvar)
    if n==2:
        ylabel(yvar)
    else:
        leg=vars[1:]
        lhand=legend(leg,'best',fancybox=True,shadow=False)
        lhand.draw_frame(False)
    #xlim(0.05,-18)

def mesa_logplot(vars,logfile=''):
    xvar=vars[0]
    yvar=vars[1]
    if logfile=='':
        f = open('profiles.index', 'r')
        pind=f.readlines()
        line= pind[-1]
        aa=line.split()
        logfile='log{0}.data'.format(aa[2])

    s=ms.star_log('.',slname=logfile)
    x  = s.get(xvar)
    n=np.size(vars)
    for i in np.arange(1,n):
        yvar=vars[i]
        y1 = s.get(yvar)
        plot(x,y1)
    xlabel(xvar)
    if n==2:
        ylabel(yvar)
    else:
        leg=vars[1:]
        legend(leg,'best',fancybox=True,shadow=False)
    xlim(0.05,-18)

def mesa_profileplot(vars,lognum,type='',prefix='profile'):
    #from matplotlib.transforms import offset_copy
    """
    python routine to read and plot profile files
    vars:   a vector of names 
            var[0] is name of x variable
            var[1] is name of first y variable
            var[2] is name of second y variable, etc.
    """
    clf()
    Msun=1.988e+33
    Rsun=6.955e+10
    gconst=6.67e-8
    xvar=vars[0]
    a1=ms.mesa_profile('.',lognum,num_type='log_num',log_prefix=prefix)
    Teff0= a1.header_attr.get('Teff')
    Mass= a1.header_attr.get('star_mass')
    Mh= a1.header_attr.get('star_mass_h1')
    Mhe= a1.header_attr.get('star_mass_he4')
    Mh=Mh/Mass
    Mhe=Mhe/Mass
    rad=a1.get('radius')
    radius=rad[0]
    logg = np.log10(gconst*Mass*Msun/(Rsun*radius)**2)
    lMh=round(np.log10(Mh),3)
    lMhe=round(np.log10(Mhe),3)
    Mass=round(Mass,4)
    Teff=int(round(Teff0,0))
    flab=r'$T_{\rm eff}$' + r'$={0} \, K,  \, M_\star/M_\odot={1},\,$'.format(Teff,Mass)
    #flab2=r'$\log \,M_{\rm H}/M_\star=' + '{0}$, '.format(lMh) + '$\, M_{\\rm He}=' + '{0}$'.format(lMhe)
    flab2=r'$\log M_{\rm  H}/M_\star\,=' + '{0},\,$ '.format(lMh) 
    flab3=r'$\log M_{\rm He}/M_\star=' + '{0}$ '.format(lMhe) 
    flab=flab+flab2+flab3
    fig=figure(1,frameon=False)
    ax = fig.add_subplot(111)
    plt.text(0.50, 1.06, flab, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,size='large')
    if xvar == 'phi':
        r  = a1.get('radius')
        N2  = a1.get('brunt_N2')
        aN = np.sqrt(np.abs(N2))
        ndim=np.size(r)
        dr = 0*r
        phi = 0*r
        dr[1:ndim-1]=r[1:ndim-1]-r[2:ndim]
        dr[0]=0.
        phi[0]=0.
        for i in np.arange(1,ndim):
            phi[i]=phi[i-1] + dr[i]*aN[i]/r[i]
        phi=phi/phi[ndim-1]
        x=1.-phi
    else:
        x  = a1.get(xvar)
    n=np.size(vars)
    for i in np.arange(1,n):
        yvar=vars[i]
        y1 = a1.get(yvar)
        if type == 'sl':
            semilogy(x,y1)
            #semilogy(x,y1,'o')
        else:
            plot(x,y1)
    if n==2:
        ylabel(yvar)
    else:
        leg=vars[1:]
        legend(leg,'best',fancybox=True,shadow=False)
    x1,x2=xlim()
    if xvar == 'logxq':
        x1=-18
        xlim(0.5,-18)
        xlabel(r'$\log (1-M_r/M_\star)$',size='x-large')
    elif xvar == 'phi':
        xlim(-0.025,1.025)
        xlabel(r'$\Phi$',size='xx-large')
    else:
        dx=x2-x1
        x1s=x1-0.05*dx
        x2s=x2+0.05*dx
        xlim(x1s,x2s)
        xlabel(xvar)
    y1,y2=ylim()
    dy=y2-y1
    y1s=y1-0.05*dy
    y2s=y2+0.05*dy
    ylim(y1s,y2s)

def plotlabel(a1,ax,xpos,ypos,pos='box',vers='no'):
    """
    python routine to read header info from file handle
    a1 and put this data on figure ax, at the position 
    (xpos,ypos)
    """
    Teff= a1.header_attr.get('Teff')
    Mass= a1.header_attr.get('star_mass')
    Mh= a1.header_attr.get('star_mass_h1')
    Mhe= a1.header_attr.get('star_mass_he4')
    #print a1,ax
    lMh=round(np.log10(Mh),3)
    lMhe=round(np.log10(Mhe),3)
    Mass=round(Mass,4)
    Teff=int(round(Teff,0))

    flab0=r'{0}\,$'.format(Teff) + r'${\rm K}$'
    flab1=r'$M_\star={0}\,M_\odot$'.format(Mass)
    flab= r'$T_{\rm eff}=' + flab0
    flab2=r'$\log \,M_{\rm  H}/M_\star\,=' + '{0}$ '.format(lMh) 
    flab3=r'$\log \,M_{\rm He}/M_\star=' + '{0}$ '.format(lMhe) 
    if pos == 'box':
        dy=-0.05
        y1=ypos-1.5*dy
        y2=y1+dy
        y3=y2+dy
        y4=y3+dy
        text(xpos, y1, flab1, horizontalalignment='left', verticalalignment='center',transform = ax.transAxes,size='small')
        text(xpos, y2, flab,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes,size='small')
        text(xpos, y3,flab2,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes,size='small')
        text(xpos, y4,flab3,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes,size='small')
    elif pos == 'top':
        toplabel = flab1 + ', ' + flab + ', ' + flab2 + ', ' + flab3 
        ax.set_title(toplabel,size=12)
    if vers == 'yes' :
        versionlabel(a1,ax)

def plotlabel2(a1,ax,xpos,ypos,pos='box',vers='no'):
    """
    python routine to read header info from file handle
    a1 and put this data on figure ax, at the position 
    (xpos,ypos)
    """
    Teff= a1.header_attr.get('Teff')
    Mass= a1.header_attr.get('star_mass')
    Mass=round(Mass,4)
    Teff=int(round(Teff,0))

    flab0=r'{0}\,$'.format(Teff) + r'${\rm K}$'
    flab1=r'$M_\star={0}\,M_\odot$'.format(Mass)
    flab= r'$T_{\rm eff}=' + flab0
    if pos == 'box':
        dy=-0.05
        y1=ypos-1.5*dy
        y2=y1+dy
        y3=y2+dy
        y4=y3+dy
        text(xpos, y1, flab1, horizontalalignment='left', verticalalignment='center',transform = ax.transAxes,size='small')
        text(xpos, y2, flab,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes,size='small')
    elif pos == 'top':
        toplabel = flab1 + ', ' + flab 
        ax.set_title(toplabel,size=12)
    if vers == 'yes' :
        versionlabel(a1,ax)

def version():
    """
    return version number of MESA installation
    """
    vfile='/Users/mikemon/mesa/data/version_number'
    f = open(vfile, 'r')
    line=f.readline()
    aa=line.split()
    vers=aa[0]
    print '\nComputed with MESA version',vers,'\n'
    return vers

def versionlabel(a1,ax):
    """
    get local time and version number of MESA installation and 
    label the current plot with it (axis instance 'ax')
    """
    import time
    localtime = time.asctime( time.localtime(time.time()) )
    #vnumb=version()
    #vnumb= a1.header_attr.get('version_number')
    #vnumb=int(vnumb)
    #print '\nComputed with MESA version',vnumb,'\n'
    #flab4=r'$\rm MESA \, version\, {0}$'.format(vnumb) 
    #flab4='MESA version {0}'.format(vnumb) 
    text(1.00, -0.10,localtime,  size='xx-small',horizontalalignment='right', verticalalignment='bottom',transform = ax.transAxes)
    #text(1.00, -0.13,flab4,  size='xx-small',horizontalalignment='right', verticalalignment='bottom',transform = ax.transAxes)

def histplotlabel(s,ax,xpos,ypos,pos='box',vers='no'):
    """
    python routine to read header info from file handle
    s and put this data on figure ax, at the position 
    (xpos,ypos)
    """
    Massv = s.get('star_mass')
    ndim=Massv.size - 1
    Mass=Massv[ndim]
    Mhv = s.get('total_mass_h1')
    Mh = Mhv[ndim]/Mass
    Mhev = s.get('total_mass_he4')
    Mhe = Mhev[ndim]/Mass
    lMh=round(np.log10(Mh),3)
    lMhe=round(np.log10(Mhe),3)
    Mass=round(Mass,4)

    flab1=r'$M_\star\,=\,{0}\,M_\odot$'.format(Mass)
    flab2=r'$\log \,M_{\rm  H}/M_\star\,=' + '{0}$ '.format(lMh) 
    flab3=r'$\log \,M_{\rm He}/M_\star=' + '{0}$ '.format(lMhe) 
    if pos == 'box':
        dy=-0.05
        y1=ypos-1.0*dy
        y2=y1+dy
        y3=y2+dy
        text(xpos, y1, flab1, horizontalalignment='left', verticalalignment='center',transform = ax.transAxes)
        text(xpos, y2, flab2,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes)
        text(xpos, y3, flab3,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes)
    elif pos == 'top':
        toplabel = flab1 + ', ' + flab2 + ', ' + flab3 
        ax.set_title(toplabel,size=12)
    if vers == 'yes' :
        versionlabel(s,ax)

def histplotlabel2(s,ax,xpos,ypos,pos='box',vers='no'):
    """
    python routine to read header info from file handle
    s and put this data on figure ax, at the position 
    (xpos,ypos)
    """
    Massv = s.get('star_mass')
    Minit  = Massv[0]
    Mfinal = Massv[-1]
    Minit =round(Minit,4)
    Mfinal=round(Mfinal,4)

    flab1=r'$M_\star\,=\,{0}\,M_\odot$'.format(Minit)
    flab1 = r'${\rm Initial}$ ' + flab1
    flab2=r'$M_\star\,=\,{0}\,M_\odot$'.format(Mfinal)
    flab2 = r'${\rm Final}$ ' + flab2
    if pos == 'box':
        dy=-0.05
        y1=ypos-1.0*dy
        y2=y1+dy
        y3=y2+dy
        text(xpos, y1, flab1, horizontalalignment='left', verticalalignment='center',transform = ax.transAxes)
        text(xpos, y2, flab2,  horizontalalignment='left', verticalalignment='center',transform = ax.transAxes)
    elif pos == 'top':
        toplabel = flab1 + ', ' + flab2 
        ax.set_title(toplabel,size=12)
    if vers == 'yes' :
        versionlabel(s,ax)

def eevalprint(file='summary.h5'):
    """
    read and print out the frequencies and associated information
    from an h5 file generated by gyre
    """
    import h5py
    import numpy as np

    with h5py.File(file, 'r') as f:
        freq = f['freq'].value
        fr_re = freq['re']
        fr_im = freq['im']
        n_p = f['n_p'].value
        n_g = f['n_g'].value
        E = f['E'].value
        nm = np.size(n_g)
        n = n_p - n_g

    print 'mode no.    n   n_p     n_g         E          freq       period'
    for i in range(nm):
        nn=int(n_p[i]) - int(n_g[i])
        fre=fr_re[i]
        per=1.e+06/fre
        print '{0:4d} {1:8d} {2:4d} {3:7d} {4:15.5e} {5:10.3f} {6:10.3f}'.format(i,nn,int(n_p[i]),int(n_g[i]),E[i],fre,per)
     
def evalprint(file='summary.h5',nad='no'):
    """
    read and print out the frequencies and associated information
    from an h5 file generated by gyre
    """
    import h5py
    import numpy as np
    from astropy import constants as const

    sbconst = const.sigma_sb.cgs.value
    Msun = const.M_sun.cgs.value
    Lsun = const.L_sun.cgs.value
    grav = const.G.cgs.value

    freq_units='HZ'

    f = gyre.read_output(file)
    data,local = gyre.read_output(file)
    
    Lstar=data['L_star']
    Mstar=data['M_star']
    Rstar=data['R_star']

    g = grav*Mstar/Rstar**2
    logg = np.log10(g)

    Teff=(Lstar/(4.*np.pi*Rstar**2 * sbconst))**(0.25)
    logLoLsun = np.log10(Lstar/Lsun)
    outstr = '         log Lstar/Lsun= {0:.3f}, Mstar/Msun= {1:0.3f}, Teff= {2:.0f} K, log g= {3:.3f} (cgs)'.format(logLoLsun,Mstar/Msun,Teff,logg)
    #print Lstar/Lsun,Mstar/Msun,Rstar,Teff,logg
    print outstr

    omega=local['omega']
    freq=local['freq']
    fr_re = np.real(freq)
    fr_im = np.imag(freq)
    n_p = local['n_p']
    n_g = local['n_g']
    ell = local['l']
    E = local['E']

    nm = len(n_g)
    n = n_p - n_g

    fac=1.0
    if freq_units=='UHZ':
        fac=1.e+6
    if nad == 'yes':
	print '         Periods are in seconds, growth times in years, and frquency units are',freq_units
	print 'mode no.   ell    n   n_p     n_g         E          fr_re       fr_im        period   tgrow (yrs)'
	for i in range(nm):
            nn=int(n_p[i]) - int(n_g[i])
            fre=fr_re[i]
            fri=fr_im[i]
            l=int(ell[i])
            per=fac/fre
            if abs(fri)>1.e-17:
                tgrow=fac/fri
                tgrow=tgrow/(3600.*24*365.25)
            else:
                tgrow=1.e+99
            if freq_units=='UHZ':
                print '{0:4d} {1:8d} {2:5d} {3:5d} {4:7d} {5:14.5e} {6:10.3f} {7:12.5e} {8:10.3f} {9:12.5e}'.format(i,l,nn,int(n_p[i]),int(n_g[i]),E[i],fre,fri,per,tgrow)
            else:
                print '{0:4d} {1:8d} {2:5d} {3:5d} {4:7d} {5:14.5e} {6:12.5e} {7:12.5e} {8:10.3f} {9:12.5e}'.format(i,l,nn,int(n_p[i]),int(n_g[i]),E[i],fre,fri,per,tgrow)
    else:
        print '         Periods are in seconds and frquency units are',freq_units
        print 'mode no.    n   n_p     n_g         E          freq       period'
        for i in range(nm):
            nn=int(n_p[i]) - int(n_g[i])
            fre=fr_re[i]
            per=fac/fre
            print '{0:4d} {1:8d} {2:4d} {3:7d} {4:15.5e} {5:10.3f} {6:10.3f}'.format(i,nn,int(n_p[i]),int(n_g[i]),E[i],fre,per)

    return logLoLsun,Mstar/Msun,Teff,logg,Rstar
     

def xvals_calc(r,rho,mr,N2,ldqfit=-8):

    ndim=len(rho)
    ivals=range(ndim-1,-1,-1)
    sum=0.0
    dq=np.zeros(ndim)
    for i in ivals:
        if i < ndim-1:
            dr1 = 0.5*(r[i+1]-r[i])
        else:
            dr1 = 0.0
        if i > 0:
            dr2 = 0.5*(r[i]-r[i-1])
        else:
            dr2 = r[i]
        dr=dr1+dr2

        dm=4.*np.pi*r[i]**2 * rho[i] * dr
        sum=sum+dm
        dq[i]=sum

    ldq= np.log10(dq/dq[0])
    ldqfit=-8
    ltest = min(abs(ldq - ldqfit))+ldqfit
    itest = np.where(ldq==ltest)

    dq0=1.-mr/mr[-1]
    dq0[dq0 < 10**(ldqfit-1)] = 10**(ldqfit-1)
    ldq0=np.log10(dq0)

    dldq = (ldq0[itest]-ldq[itest])[0]
    #print ldq0[itest],ldq[itest],dldq
    ldqnew=ldq + dldq
    #print len(ldqnew),len(ldq0)
    mask = ldqnew > ldqfit
    np.copyto(ldqnew,ldq0,where=mask)
    #ldqnew[ldqnew > ldqfit] = ldq0 

    phi=np.zeros(ndim)
    for i in range(0,ndim):
        nval = np.sqrt(abs(N2[i]))
        if i == 0:
            dr = r[i]
            nval = np.sqrt(abs(N2[i]))
            #phi[i]= dr*nval/r[i]
            phi[i]= 0.
        else:
            dr = r[i]-r[i-1]
            nval = 0.5*(np.sqrt(abs(N2[i]))+np.sqrt(abs(N2[i])))
            ravg = 0.5*(r[i]+r[i-1])
            phi[i]=phi[i-1]+ dr*nval/ravg

        #phi[i]=phi[i-1]+ dr*nval/(0.5*(r[i+1]+r[i]))
        #print i

        #print r[0]
        #print phi[0]
        #print r[ndim-1]
        #print phi[ndim-1]
    phi = phi/phi[-1]


    return ldqnew,phi

def inlist_set(keys,filein='inlist_1.0',fileout='inlist'):
    """
    Read the inlist file "filein" and reset the values of select inlist parameters, 
    saving the result in "fileout". The variable "keys" is a list of the form 
    [ [keyname1,keyval1], [keyname2,keyval2], ... ]. For example, if you first element
    of this list is ["initial_mass",3.0], then the initial_mass parameter in the 
    inlist file will be set to 3.0.
    """
    print '\nFiltering inlist file \'{0}\'...'.format(filein)
    f=open(filein,"r")
    fo=open(fileout,"w")
    count= [ [item[0],0] for item in keys ]
    dcount = dict(count)
    for line in f:
        rline = line
        for key in keys:
            larr = line.split()
            if len(larr) > 0:
                if key[0] == larr[0]:
                    skey = '{0}'.format(key[1])
                    if not is_number(skey):
                        if (skey == '.true.' or skey == '.false.'):
                            skey = '{0}'.format(key[1])
                        else:
                            skey = '\'{0}\''.format(skey)
                    rline = line.replace(larr[2],skey)
                    dcount[key[0]]+=1
        fo.write(rline)
    for key in keys:
        k0=key[0]
        if dcount[k0] == 0:
            print '\nError: {0} instances of \'{1}\' found'.format(dcount[k0],k0)
            print 'Exiting...\n'
            exit()
        if dcount[k0] > 1:
            print 'Warning: {0} instances of \'{1}\' found'.format(dcount[k0],k0)
    print '\nNew inlist file stored in \'{0}.\'\n'.format(fileout)
    f.close()
    fo.close()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def fig_dims(pts=245.26653):
    # pts=245.26653 is the column width in ApJ format
    fig_width_pt  = pts
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio * 1.20   # figure height in inches
    dims      = [fig_width_in, fig_height_in] # fig dims as a list
    return dims

