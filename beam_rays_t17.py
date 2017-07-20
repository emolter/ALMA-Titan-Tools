#!/usr/local/bin/python
'''In order to carry out spatially resolved NEMESIS retrievals with ALMA data it is necessary to correctly model the flux from Titan that lies within a single ALMA beam pointed toward an arbitrary part of Titan's disk. This program calculates the rays and weightings required to accomplish this given an elliptical beam at an arbitrary position and angle from North. The output is a .spx header to execute the averaging in NEMESIS. This program must be run inside CASA using execfile('beam_rays.py') and was written based on CASA 4.5'''

### Ned Molter 07/05/16
### This version creates new plots as in Thelen et al., 2017; XET 07/20/17.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####################################################################################################
############# input parameters ########################
####################################################################################################

##### Observation Parameters #####
img = 'Titan_17.image'
outfile = 'test_t17.spx'
Titan_dist = 9.13708675379994 #au
d_shift = 14.0790459 #km/s
subobslat = 24.16 #degrees
ccw = 2.1285 #degrees


##### Model Options #####
showplot = False #print the mean latitude and longitude of the observation and show beam placement and weightings
diskavg = False #compute disk averaged spectrum?  Note disk average takes ~1 minute to run because of runtime of imval
wtsCutoff = 0.0 #How small of a weighting is too small to put into the .spx header? Value between zero and 1. Value <= 0 means no cutoff


##### Model Parameters #####
top_atm = 1200 + 2575 #km
if not diskavg:
    #cx,cy tell the model where to extract data. Together they form a grid of x,y distances (in km from the center of Titan) at which to extract spectra and compute weights. Note that 0.05 arcsec ~= 350 km at around Titan's distance. A disk averaged model forces cx = cy = [0.0]
    cx = [0.0]
    cy = [-3000.0,0.0,3000.0]
radii = [0.0,500.0,1000.0,1500.0,2000.0,2500.0,2575.0,2600.0,2625.0,2650.0,2675.0,2700.0,2725.0,2750.0,2775.0,2800.0,2825.0,2850.0,2875.0,2900.0,2925.0,2950.0,2975.0,3000.0,3025.0,3050.0,3075.0,3125.0,3175.0,3225.0,3275.0,3325.0,3375.0,3425.0,3525.0,3625.0,3750.0] #radii of annuli in km
dx = 5 #resolution of model Titan in km. Runtime increases exponentially as this number decreases

#####################################################################################################
############## Function definitions ######################
#####################################################################################################

def casa_extract(img):
    '''Brings image header info into script using imhead'''
    cubeheader = imhead(imagename=img,mode='list')
    (beamx,beamy) = (float(cubeheader['beammajor']['value']),float(cubeheader['beamminor']['value']))
    theta = float(cubeheader['beampa']['value']) #here theta defined as angle from north to east in degrees
    (refx,refy) = (float(cubeheader['crpix1']),float(cubeheader['crpix1'])) #center pixel
    (pixszx,pixszy) = (np.degrees(np.fabs(3600*float(cubeheader['cdelt1']))),np.degrees(np.fabs(3600*float(cubeheader['cdelt2']))))
    nspec = cubeheader['shape'][3] #number of channels
    f0 = float(cubeheader['crval4']) #reference freq in Hz
    df = float(cubeheader['cdelt4']) #channel width in Hz
    i0 = float(cubeheader['crpix4']) #reference channel usually 0
    return [beamx,beamy,theta,refx,refy,pixszx,pixszy,nspec,f0,df,i0]

def jy2rad(val):
    # Find conversion from Jy/bm to radiance units
    abeam      = 2.0*pi*beamx*beamy/(8.0*np.log(2.0)) #beam area
    apix       = abs(pixszx*pixszy) #pixel area
    apixdeg    = apix*(1.0/3600.0)**2
    beam_per_pix = apix/abeam #number of beams per pixel
    sr_per_pix = apixdeg*(np.pi/180.0)**2 #convert to steradians
    fconv      = beam_per_pix/sr_per_pix * 1e-26 * 1e-4 * cm2hz # 1 Jy = 1e-26 W/m2/Hz, so units are eventually W/cm2/sr/cm-1
    return fconv*val

def gauss2d(x,y,fwhm_x,fwhm_y,x0,y0,theta):
    """ Takes in np arrays for x,y. Returns the value of the 2d elliptical gaussian. Theta defined as angle in radians from x axis to semimajor axis """
    sig_x = fwhm_x / (2*np.sqrt(2*np.log(2)))
    sig_y = fwhm_y / (2*np.sqrt(2*np.log(2)))
    A = 1.0
    a1 = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2)
    b1 = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2)
    c1 = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2)
    g = A*np.exp(-(a1*(x-x0)**2 - 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    return g

def spEx(img,xpix,ypix):
    '''Extracts a spectrum at a single point on Titan then converts to radiance units and gets correctly doppler shifted.  Takes a bunch of the casa_extract paraemters'''

    # Generate frequency grid
    freqspec = ((np.arange(nspec) - i0) * df + f0) #in Hz
    
    # Convert and extract spectrum -- in CASA 4.0.1 imval can only extract regions, not individual pixels, but lower left pixel is in ['data'][0][0]
    freqspec = freqspec*(d_shift*1000+lightspeed)/lightspeed #doppler shift
    freqspec = freqspec/(lightspeed*100) #convert to cm-1 from Hz. 
    imgvalues = imval(imagename=img,box=str(xpix)+','+str(ypix)+','+str(xpix+1)+','+str(ypix+1))

    #Calculate how far away from cx,cy the pixel actually is located
    real_xpix,real_ypix = imgvalues['blc'][0],imgvalues['blc'][1]
    xcorr_km,ycorr_km = (real_xpix - xpix)*pixszx*(a/beamx),(real_ypix - ypix)*pixszy*(b/beamy) #beamx/a and beamy/b are easy conversion between km and arcsec
    
    valspec = imgvalues['data'][0][0]
    valspec = jy2rad(valspec) #convert spectrum to radiance

    #always return spectrum from low wn to high wn
    if freqspec[0] > freqspec[-1]:
        freqspec = freqspec[::-1]
        valspec = valspec[::-1]
    
    return freqspec,valspec,xcorr_km,ycorr_km #freq in GHz, flux
        
def computeWeights(cx,cy,xcorr_km,ycorr_km):
    """ Takes in definition of elliptical beam at arbitrary location on Titan's disk.  Returns weightings to be used in .spx header """

    #create grid z where each point represents radius from center of Titan
    lim = top_atm+a
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = np.sqrt(x**2+y**2)
    
    #Change values of z to the angle corresponding to those radii
    angles = [np.degrees(np.arcsin(float(r)/float(top_atm))) for r in radii] #top of atmosphere emission angle
    for i in range(len(radii)):
        z[np.logical_and(z >= radii[i-1], z < radii[i])] = angles[i]
    z[z >= radii[-1]] = float('NaN')

    #Make Gaussian beam
    g = gauss2d(x,y,a,b,cx+xcorr_km,cy+ycorr_km,theta)

    print('Actual location of pixel extracted: ('+str(cx+xcorr_km)+', '+str(cy+ycorr_km)+')')
    
    #compute normalized weights
    wts = {}
    for val in angles:
        garr = g[np.where(z == val)]
        wts[val] = sum(garr)
    gnanarr =  g[np.where(np.isnan(z))] #treating NaN values such that weighting will not add up to one
    s = sum(wts.values())+sum(gnanarr)
    for key,val in wts.items():
        val = float(val)/float(s)
        wts[key] = val

    meanangle = sum([val*key for key,val in wts.items()])
    print('Mean emission angle: '+str(meanangle))
        
    ########################################################################

    #Now compute mean latitude and longitude of observation

    #Finding vector of true north of Titan
    northx = -np.sin(ccw)*np.cos(subobslat)
    northy = np.cos(ccw)*np.cos(subobslat)
    northz = np.sin(subobslat) 

    z_lat = y*x*0.0    
    with np.errstate(divide='ignore',invalid='ignore'): #We actually want all y > Titan_radius + top_atm to be nans, so the invalid inputs to arcsin are helping here
        zcoord = np.sqrt((top_atm)**2 - x**2 - y**2) #these are the actual z-coordinates (distance from Titan center to observer) at each x,y point
        dprod = (northx*x + northy*y + northz*zcoord)/(top_atm) #dot product of north pole vector and each vector in model planet
        z_lat = 90 - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid

    conv = np.multiply(g,z_lat)
    meanlat = np.nansum(conv)/np.nansum(g)
    print('Mean top-of-atmosphere latitude: '+str(meanlat))

    z_flat = np.copy(z)*0 + 1
    conv_flat = np.multiply(g,z_flat)
                
    return wts,meanlat,conv_flat,z_lat,x,y,z,g

def wtsReduce(wts,wtsCutoff):
    '''Remove any annulus with less than a given weighting. Renormalize remaining annuli by uniformly multiplying them such that they add up to the value they used to. Note this isn't the most nuanced treatment: it'd be better to "give" the weighting of the deleted annuli to a nearby annulus, not distribute them evenly.'''
    norm = np.sum(wts.values()) #the weightings may not add up to 1 at this stage because of treatment of NaN values
    wts_new = {key:wts[key] for key in wts if wts[key] >= wtsCutoff}
    renorm = norm/np.sum(wts_new.values())
    wts_new = {key:wts_new[key]*renorm for key in wts_new}
    return wts_new



### Disk averaged stuff ###
    
def jy2rad_diskavg(val):
    # Find conversion from Jy/bm to radiance units
    abeam      = 2.0*pi*beamx*beamy/(8.0*np.log(2.0)) #beam area
    apix       = abs(pixszx*pixszy) #pixel area
    apixdeg    = apix*(1.0/3600.0)**2
    beam_per_pix = apix/abeam #number of beams per pixel
    sr_per_pix = apixdeg*(np.pi/180.0)**2 #convert to steradians
    atitan = np.pi*((beamx/a)*top_atm)**2 #in arcsec. beamx/a is just an easy way to convert km to arcsec
    titan_pix = atitan/apix
    fconv      = beam_per_pix/(titan_pix*sr_per_pix) * 1e-26 * 1e-4 * cm2hz # 1 Jy = 1e-26 W/m2/Hz, so units are eventually W/cm2/sr/cm-1
    return fconv*val
    
def spEx_diskavg(imagename,pixx,pixy):
    '''Extracts a spectrum summed over Titan's full disk then converts to radiance units and gets correctly doppler shifted.  Takes a bunch of the casa_extract paraemters'''
    xcorr_km,ycorr_km = 0.0,0.0 #a disk average has cx = cy = [0.0] so this correction should always be zero
    
    # Generate frequency grid
    freqspec = ((np.arange(nspec) - i0) * df + f0) #in Hz
    
    # Convert and extract spectrum -- in CASA 4.0.1 imval can only extract regions, not individual pixels, but lower left pixel is in ['data'][0][0]
    freqspec = freqspec*(d_shift*1000+lightspeed)/lightspeed #doppler shift
    freqspec = freqspec/(lightspeed*100) #convert to cm-1 from Hz.

    r_diskavg = top_atm + a/(np.sqrt(2*np.log(2))) #this defines Titan plus a buffer equal to 2*sigma_psf. This defines how big our average will be
    print('Averaging over a circular region with radius r = '+str(r_diskavg)+' km')
    print('This is equal to the radius of Titan including atmosphere + 2*sigma_psf')
    
    averaging_r_pix = np.round((r_diskavg)*(beamx/a)*(1/pixszx)) #the radius of the disk over which we want to average in pixels: Titan up to top of atmosphere plus half a beam
    imgvalues = imval(imagename=img,box=str(pixx-averaging_r_pix)+','+str(pixy-averaging_r_pix)+','+str(pixx+averaging_r_pix)+','+str(pixy+averaging_r_pix)) #This step takes a while
    imgdata = imgvalues['data']

    #turn box of image values into circle
    counter = 0
    testgrid = []
    for i in range(len(imgdata)):
        for j in range(len(imgdata[0])):
            if np.sqrt(np.abs(i-averaging_r_pix)**2 + np.abs(j-averaging_r_pix)**2) > averaging_r_pix:
                counter += 1
                imgdata[i][j] = np.zeros(len(imgdata[i][j]))
                testgrid.append([i,j,0])
            else:
                testgrid.append([i,j,1])

    #compute disk averaged spectrum
    valspec = np.sum(np.sum(imgdata,axis=0),axis=0) #sum over all pixels in image plane
    valspec = jy2rad_diskavg(valspec)

    #always return spectrum from low wn to high wn
    if freqspec[0] > freqspec[-1]:
        freqspec = freqspec[::-1]
        valspec = valspec[::-1]

    return freqspec,valspec,xcorr_km,ycorr_km

def computeWeights_diskavg():
    """ Returns weightings of a disk average to be used in .spx header """

    #create grid z where each point represents radius from center of Titan
    lim = top_atm+a
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = np.sqrt(x**2+y**2)
    
    #Change values of z to the angle corresponding to those radii
    angles = [np.degrees(np.arcsin(float(r)/float(top_atm))) for r in radii] #top of atmosphere emission angle
    for i in range(len(radii)):
        z[np.logical_and(z >= radii[i-1], z < radii[i])] = angles[i]
    z[z >= radii[-1]] = float('NaN')

    z_flat = np.copy(z)*0 + 1
    
    #compute normalized weights
    wts = {}
    for val in angles:
        arr = z_flat[np.where(z == val)]
        wts[val] = sum(arr)
    s = sum(wts.values())
    for key,val in wts.items():
        val = float(val)/float(s)
        wts[key] = val    

    return wts
        
#####################################################################################################
############## Code #################
#####################################################################################################
#Constants
lightspeed = 299792458.0 #m/s
cm2hz = 29.9792458 * 1e9

subobslat = np.deg2rad(subobslat)
ccw = np.deg2rad(ccw)

if diskavg:
    cx = [0.0]
    cy = [0.0]

[beamx,beamy,theta,refx,refy,pixszx,pixszy,nspec,f0,df,i0] = casa_extract(img) #extracting image parameters from CASA

#calculated parameters
Titan_dist = Titan_dist*149597870 #to km
theta = np.radians(theta+90) #change to what's required by computeWeights
a = Titan_dist*np.radians(np.arcsin(beamx/3600)) #major axis at half-max in km
b = Titan_dist*np.radians(np.arcsin(beamy/3600)) #minor axis at half-max in km

#for each grid point cx,cy extract spectrum and weights
spx = {}
beam_ind = len(cy)
#these arrays need to be in the same shape as conv_flat from computeWeights, which is printed.
conv_flat2 = np.zeros((2504,2504))
conv_ell = np.zeros((beam_ind,2504,2504))
g2 = np.zeros((2504,2504))
conv_ell = np.zeros((beam_ind,2504,2504))
g2_ell = np.zeros((beam_ind,2504,2504))
ind = 0
for i in cy:
    for j in cx:
        print('-----------------------------------------')
        print('Input location of pixel extraction: ('+str(j)+', '+str(i)+')')
        pixx = refx+j*(beamx/a)*(1/pixszx) #beamx/a is just an easy way to convert km to arcsec
        pixy = refy+i*(beamx/a)*(1/pixszy)
        if not diskavg:
            (freq,flux,xcorr_km,ycorr_km) = spEx(img,pixx,pixy)
            wts,meanlat,conv_flat,z_lat,x,y,z,g = computeWeights(j,i,xcorr_km,ycorr_km)
            conv_flat2 += conv_flat
            g2 += g
            conv_ell[ind,:,:] = conv_flat
            g2_ell[ind,:,:] = g
            ind += 1
        else:
            (freq,flux,xcorr_km,ycorr_km) = spEx_diskavg(img,pixx,pixy)
            wts = computeWeights_diskavg()
            meanlat = 0.0
        if wtsCutoff > 0:
            wts = wtsReduce(wts,wtsCutoff)
        spx[(j,i)] = (wts,freq,flux,meanlat)
        
print('-----------------------------------------')
print('Shape of arrays (for showplot): '+str(conv_flat.shape)) #use for the above arrays
        
if not diskavg:        
    if showplot:
        # Plot extraction aperture overlaid on Titan with lines of latitude
        fig,ax = plt.subplots(figsize = (10,10))
        g2[np.sqrt(x**2 + y**2) >= 3775] = float('NaN')
        g2_ell[:,np.sqrt(x**2 + y**2) >= 3775] = float('NaN')
        img=ax.imshow(g2,extent=[x.min(),x.max(),y.min(),y.max()], origin='lower', interpolation='nearest', cmap='coolwarm',vmax=1.)
        titanlimb = plt.Circle((0, 0), 2575, color='k',fill=false, linestyle='dashed', linewidth=2)
        titanatm = plt.Circle((0, 0), top_atm, color='k',linestyle='dashed',fill=false,linewidth=3)
        ax.add_artist(titanlimb)
        #ax.add_artist(titanatm)
    
        #Overlay latitudes as contours
        ctr=ax.contour(z_lat,colors='black',extent=[x.min(),x.max(),y.min(),y.max()], lindwidths=3)
        #man_loc = [(931,570),(661,827),(540,1160),(585,1492),(796,1740),(1069,1912)]
        ax.clabel(ctr, inline=1, fontsize=12, fmt='%.0f')  #,manual=man_loc), add for manual or interactive latitude labels
        for line in ctr.collections: #Making negative contours solid instead of dashed
            if line.get_linestyle() != [(None, None)]:
                line.set_linestyle([(None, None)])

        #Overlay the original extraction aperture (interpolated to new grid so it looks a bit pixelated)
        for i in range(beam_ind):
                ax.contour(g2_ell[i], levels=[1./e],colors='black',linestyles='dotted',linewidths=3,extent=[x.min(), x.max(), y.min(),y.max()])
                
        ax.set_xlim([-4000,4000])
        ax.set_ylim([-4000,4000])
        ax.set_xlabel('Radius (km)',fontsize=16)
        ax.set_ylabel('Radius (km)',fontsize=16)
    
        #Optional: Set plot title and print mean top-of-atmosphere latitudes:
        #ax.set_title('2015', fontsize=16)
        #plt.text(524,2753,"48.10 N",fontsize=12)
        #plt.text(524,-127,"23.06 N",fontsize=12)
        #plt.text(524,-3103,"16.78 S",fontsize=12)
    
        #Colorbar
        cbar = fig.colorbar(img,orientation="horizontal",fraction=0.04375,pad=0.08,ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
        cbar.ax.set_xlabel('Weighting',fontsize=14)
        
        fig.show()
    
#output to file in .spx format
with open(outfile,'w') as f:
    for location,spec in sorted(spx.items()):
        f.write(str(np.float32(2*(spx[(cx[0],cy[0])][1][1]-spx[(cx[0],cy[0])][1][0])))+'     '+str(spec[3])+'     0.0     1\n')
        f.write(str(len(spec[1]))+'\n')
        f.write(str(len(spec[0]))+'\n')
        for key,val in sorted(spec[0].items()):
            f.write('      0.00000      0.00000      '+str(key)+'      '+str(key)+'      180.000      '+str(val)+'\n')
        for i in range(len(spec[1])):
            f.write(str(spec[1][i])+'     '+str(spec[2][i])+'     1.00000E-11'+'\n')
f.close()
