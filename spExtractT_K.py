#Extract a mean spectrum (in K vs. GHz written as a 2-column ascii file), from a CASA image cube, using a mask that contains a fraction ptile of the total flux in the supplied image channel and rectangular region. 
#Execute this script inside CASA using execfile("spExtractT.py")
import numpy as np
from casa import *
from scipy.optimize import brentq
from matplotlib import pyplot as plt 

c=2.99792458e10
Jy=1.0e-23
k=1.380658e-16

imgname="HC3N.X156f.clean1.image"
outfile="HC3N_Spectrum.txt"
CHAN=100 #Channel to use for generating the threshold mask
XMIN=108;YMIN=108;XMAX=148;YMAX=148 #Coordinates of rectangular subimage containing the object of interest
ptile=0.9 #Fraction of flux contained in the mask


# Telescope beam area in sr    
def beamOmega(x,y):
    # x and y are the minor and major ellipse axes in arcsec
    # 4*pi*(180/pi)**2 *60**4 = 534638377800.0 is the number of square arcsec in a sphere
    # Uses formula for geometric area under a 2D Gaussian: pi*a*b/(4 ln2) 
    
    return (4.*np.pi) * (2.*np.pi*x*y)/(534638377800.0*8.*np.log(2.0))

# Conversion factor for flux in Jy per beam to TMB (Rayleigh Jeans) in K
def flux2Tb(imgname):
    cubeheader = imhead(imagename=imgname, mode="list")
    x = cubeheader['beammajor']['value']
    y = cubeheader['beamminor']['value']
    freq = cubeheader['restfreq'][0]
    
    #Get beam angular size
    omega = beamOmega(x,y)
    #Get wavelength
    l=c/freq
    
    return Jy * (l**2/(2.*k*omega))
    

# Helper function for spExT to find threshold corresponding to given flux percentile 
def fluxThreshEq(thresh,image,ptile):
    masked = np.ma.array(image,mask=image<thresh,fill_value=0.0)
    return masked.filled().sum() / image.sum() - ptile

# Write two-column spectrum
def write2col(data1,data2,outfile):
    f = open(outfile,'w')
    for i in range(len(data1)):
        f.write("%12.8f  %12.5e\n" %(data1[i], data2[i]))
    f.close()

    
# Get image info
cubeheader = imhead(imagename=imgname, mode="list")
# Get the number of frequency channels
nspec = cubeheader['shape'][3]
# reference freq in Hz
f0 = float(cubeheader['crval4']) 
# channel width in Hz
df = float(cubeheader['cdelt4']) 
# reference pixel
i0 = float(cubeheader['crpix4'])

# Generate frequency grid
freqspec = ((np.arange(nspec) - i0) * df + f0)

# Extract data in (default) box
subcube = imval(imagename=imgname,box=str(XMIN)+','+str(YMIN)+','+str(XMAX)+','+str(YMAX))['data']

# Take spatial slice at channel CHAN
subcubeplane=subcube[:,:,CHAN]

thresh = brentq(fluxThreshEq,np.min(subcubeplane),np.max(subcubeplane),args=(subcubeplane,ptile))
masked = np.ma.array(subcubeplane,mask=subcubeplane<thresh)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(1,1,1)
ax.set_title('Extraction subregion mask')
ax.imshow(masked.transpose(),origin='lower',interpolation='nearest')
fig.show()

Kfactor=flux2Tb(imgname)

spectrum=[]
for i in range(len(subcube[0][0])):
  masked = np.ma.array(subcube[:,:,i],mask=subcubeplane<thresh)
  spectrum.append(masked.mean()*Kfactor)

#always return spectrum from low wn to high wn
if freqspec[0] > freqspec[-1]:
  freqspec = freqspec[::-1]
  spectrum = spectrum[::-1]

write2col(freqspec/1e9,np.asarray(spectrum),outfile)

 
