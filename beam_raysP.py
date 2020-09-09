#!/usr/local/bin/python
'''In order to carry out spatially resolved NEMESIS retrievals with ALMA data it is necessary to correctly model the flux from Titan that lies within a single ALMA beam pointed toward an arbitrary part of Titan's disk. This program calculates the rays and weightings required to accomplish this given an elliptical beam at an arbitrary position and angle from North. The output is a .spx header to execute the averaging in NEMESIS.  Functionality is included to extract a single point in an ALMA spectrum or a full disk average. This program must be run inside CASA using execfile('beam_rays.py') and was written based on CASA 4.5. Updated by MAC to use absolute image pixel values instead of values in km with respect to Titan.  Accounts for possible ephemeris error using cenpix.'''
### Ned Molter 07/14/16 ###

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker

####################################################################################################
############# input parameters ########################
####################################################################################################

##### Observation Parameters #####
img = 'Titan_170508_H13CCCN.cleanNat_b0.25x0.2.image'
sigma = 6.0 # Spectral noise (mJy) 
outfile = '/home/mcordine/projects/titan/NEMESIS/Continuum/Cordiner2017/testW/Titan_170508_H13CCCN.cleanNat_b0.25x0.2-west.spx'
outimg = 'beam_rays.ps' #only used if showplot = True
Titan_dist = 9.258 #au. Get from delta column in JPL Horizons
d_shift = 0.0 #km/s. Get from deldot column in JPL Horizons
subobslat = 26.41 #degrees. Get from Ob-lat column in JPL Horizons
ccw = 5.26 #degrees. Get from  NP.ang column in JPL Horizons - counterclockwise from north
cenpix = [127.6,132.5]
resfact = 2 # channels per spectral resolution element
skip = 2 # Channel decimation factor for output spectrum

##### Model Options #####
showplot = False #Display image of beam overlaid on Titan with latitudes as contours?
diskavg = False  #compute disk averaged spectrum?  Note disk average takes ~1 minute to run because of runtime of imval
wtsCutoff = 0.0 #How small of a weighting is too small to put into the .spx header? Value between zero and 1. Value <= 0 means no cutoff. Be careful when using this - may give unexpected results - see function definition for wtsReduce routine below

##### Model Parameters #####
pixels = [[144,132]] # x,y pixel pair(s) for spectral extraction and modeling
top_atm = 1000. + 2575. #km. Must be same as top of atmosphere in eventual NEMESIS model.
nsig_psf = 2.0 #Used for diskavg = True only. distance past the top of the atmosphere included in the disk average.  units of sigma_psf i.e. 2.0 -> 2sigma_psf

#radii = np.asarray([0.,500.,1000.,1500.,2000.,2500.,2600.,2700.,2800.,2900.,3000.,3100.,3200.,3300.,3400.,3500.,top_atm]) #radii of annuli in km - basic accuracy
#radii = np.asarray([0.,500.,1000.,1500.,2000.,2500.,2550.,2600.,2650.,2700.,2750.,2800.,2850.,2900.,2950.,3000.,3050.,3100.,3150.,3200.,3250.,3300.,3350.,3400.,3450.,3500.,top_atm]) #radii of annuli in km - better; accurate for most lines
radii = np.asarray([0.,500.,1000.,1500.,1750.,2000.,2250.,2500.,2575.,2600.,2625.,2650.,2675.,2700, top_atm]) #radii of annuli in km - disk; ONLY FOR CONTINUUM up to 125 km (optimised for limb beam to within 0.05% accuracy)
#radii = np.asarray([0.0,500.,1000.,1500.,1750.,2000.,2250.,2500.,2575.,2600.,2625.,2650.,2675.,2700.,2725.,2750.,2775.,2800.0,2825.,2850.,2875.,2900.,2925.,2950.,2975.,3000.,3025.,3050.,3075.,3125.,3175.,3225.,3275.,3325.,3375.,3425.,3525.,3625.,top_atm]) #radii of annuli in km
dx = 5 #resolution of model Titan in km. Runtime increases exponentially as this number decreases


#####################################################################################################
############## Function definitions ######################
#####################################################################################################

######## Functions interfacing with CASA #######

def casa_extract(img):
    #Brings image header info into script using imhead
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

def spEx(img,xpix,ypix):
    #Extracts a spectrum at a single point on Titan then converts to radiance units and gets correctly doppler shifted.  Takes in a bunch of the casa_extract paraemters

    # Generate frequency grid
    freqspec = ((np.arange(nspec) - i0) * df + f0) #in Hz
    
    # Convert and extract spectrum -- in CASA 4.0.1 imval can only extract regions, not individual pixels, but lower left pixel is in ['data'][0][0]
    freqspec = freqspec*(d_shift*1000+lightspeed)/lightspeed #doppler shift
    freqspec = freqspec/(lightspeed*100) #convert to cm-1 from Hz. 
    imgvalues = imval(imagename=img,box=str(xpix)+','+str(ypix)+','+str(xpix+1)+','+str(ypix+1))
    
    valspec = imgvalues['data'][0][0]
    valspec = jy2rad(valspec) #convert spectrum to radiance

    #always return spectrum from low wn to high wn
    if freqspec[0] > freqspec[-1]:
        freqspec = freqspec[::-1]
        valspec = valspec[::-1]
    
    return freqspec,valspec #freq in GHz, flux


######## Helper Functions #######
    
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
    #Takes in np arrays for x,y. Returns the value of the 2d elliptical gaussian. Theta defined as angle in radians from x axis to semimajor axis
    sig_x = fwhm_x / (2*np.sqrt(2*np.log(2))) #sig is the standard deviation
    sig_y = fwhm_y / (2*np.sqrt(2*np.log(2)))
    A = 1.0
    a1 = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2)
    b1 = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2)
    c1 = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2)
    g = A*np.exp(-(a1*(x-x0)**2 - 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    return g



############## The "Guts" ###############
            
def computeWeights(i,j):
    """ This is where the magic happens. Builds grid representing a model Titan. Computes top-of-atmosphere emission angle at each point on the grid. Convolves this with a Gaussian beam at an arbitrary point on Titan. Computes and returns weightings to be used in .spx header. Also calculates mean lat/lon and emission angle of observation. """

    # Calculate offsets of chosen pixel from Titan center (in km)
    cx = (a/beamx) * (i-cenpix[0]) *pixszx
    cy = (a/beamx) * (j-cenpix[1]) *pixszy
    
    #create grid z where each point represents radius from center of Titan
    lim = top_atm+a
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = np.sqrt(x**2+y**2) #z now contains radii from the center
    
    midpoints = 0.5 * (radii[1:] + radii[:-1])
    
    #Change values of z to the angle corresponding to those radii
    angles = [np.degrees(np.arcsin(float(r)/float(top_atm))) for r in midpoints] #top of atmosphere emission angle
    for i in range(len(angles)):
        z[np.logical_and(z >= radii[i], z < radii[i+1])] = angles[i]
    z[z >= radii[-1]] = float('NaN')

    #Make Gaussian beam
    g = gauss2d(x,y,a,b,cx,cy,theta)

    print('Location of extracted spectrum relative to center: ('+str(cx)+', '+str(cy)+')')
    
    #compute normalized weights
    wts = {}
    for val in angles:
        garr = g[np.where(z == val)]
        wts[val] = sum(garr)
    gnanarr =  g[np.where(np.isnan(z))] 
    s = sum(wts.values())+sum(gnanarr) #treating NaN values such that weighting will not add up to one, i.e. not renormalizing. This improves model fit to data significantly
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

    z_lat = y*x*0.0  #making an empty array the size of z  
    with np.errstate(divide='ignore',invalid='ignore'): #We actually want all y > Titan_radius + top_atm to be nans, so the invalid inputs to arcsin are helping here
        zcoord = np.sqrt((top_atm)**2 - x**2 - y**2) #these are the actual z-coordinates (distance from Titan center to observer) at each x,y point
        dprod = (northx*x + northy*y + northz*zcoord)/(top_atm) #dot product of north pole vector and each vector in model planet
        z_lat = 90 - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid

    conv = np.multiply(g,z_lat)
    meanlat = np.nansum(conv)/np.nansum(g)
    print('Mean top-of-atmosphere latitude: '+str(meanlat))

    ########################################################################

    #Plots
    
    if showplot:
        # Plot beam overlaid on image of Titan with lines of latitude
        z_flat = np.copy(z)*0 + 1
        conv_flat = np.multiply(g,z_flat)
        fig2,ax = plt.subplots(figsize = (8.5,8.5))

        #Tricking the plot to turn axis labels into values of km, even though data is in terms of indices of z
        major_ticks_corrected = [-4000,-3000,-2000,-1000,0,1000,2000,3000,4000] #this is what we want them to eventually be
        major_ticks_raw = [(val + top_atm + a)/dx for val in major_ticks_corrected] #converting into the native units of the arrays z_lat and conv_flat
        ax.set_xticks(major_ticks_raw) #setting locations to those converted values
        ax.set_yticks(major_ticks_raw)

        #Plot the thing
        im = ax.imshow(conv_flat, cmap='RdBu',origin='lower') #Plot beam

        #Overlay latitudes as contours
        ctr = ax.contour(z_lat,colors='gold',linewidths=2)
        ax.clabel(ctr, inline=1, fontsize=18, fmt='%1.1f')
        for line in ctr.collections: #Making negative contours solid instead of dashed
            if line.get_linestyle() != [(None, None)]:
                line.set_linestyle([(None, None)])
        ax.legend([ctr.collections[-1]],['Top-of-Atmosphere Latitude'],loc='upper left')

        #Finish tricking the plot to turn axis labels into values of km
        ticklabels_old = ax.get_xticks().tolist()
        ticklabels = [val*dx - top_atm - a for val in ticklabels_old] #converting current tick label values back into km
        ticklabels = [str(int(val)) for val in ticklabels] #Removing decimal at end to make look nice
        ax.set_xticklabels(ticklabels,fontsize=14)
        ax.set_yticklabels(ticklabels,fontsize=14)
        ax.set_xlim([a/dx - 500/dx,(a+2*top_atm)/dx + 500/dx]) #Constrain to have only a bit of white space around Titan
        ax.set_ylim([a/dx - 500/dx,(a+2*top_atm)/dx + 500/dx])
        ax.set_xlabel('Distance from Center of Titan (km)',fontsize=16)
        ax.set_ylabel('Distance from Center of Titan (km)',fontsize=16)
        
        #Colorbar
        cbar = fig2.colorbar(im,orientation='horizontal',shrink=0.78,pad=0.1)
        cbar.ax.set_xlabel('Weighting',fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        
        fig2.savefig(outimg,bbox='None')

    #Other diagnostic plots
    
    ## #Plot Gaussian
    ## fig0 = plt.figure(figsize = (15,15))
    ## ax = fig0.add_subplot(111)       
    ## ax.imshow(g,cmap='RdBu',origin='lower')
    ## plt.show()
    ## #Plot z
    ## fig1 = plt.figure(figsize = (15,15))
    ## ax = fig1.add_subplot(111)
    ## ax.imshow(z, cmap='Blues',origin='lower')
    ## plt.show()

    ## #Plot convolution
    ## z_flat = np.copy(z)*0 + 1
    ## conv_flat = np.multiply(g,z_flat)
    ## conv = np.multiply(g,z)
    ## fig2,ax = plt.subplots(figsize = (15,15))
    ## im = ax.imshow(conv_flat, cmap='RdBu',origin='lower')
    ## ctr = ax.contour(z,colors='yellow')
    ## ax.clabel(ctr, inline=1, fontsize=14, fmt='%1.1f')
    ## cbar = fig2.colorbar(im,orientation="horizontal")
    ## cbar.ax.set_xlabel('Weighting',fontsize=18)
                
    return wts,meanlat

def wtsReduce(wts,wtsCutoff):
    '''Remove any annulus with less than a given weighting. Renormalize remaining annuli by uniformly multiplying them such that they add up to the value they used to. Note this isn't the most nuanced treatment: it'd be better to "donate" the weighting of the deleted annuli to a nearby annulus, not redistribute weightings evenly.'''
    norm = np.sum(wts.values()) #the weightings may not add up to 1 at this stage because of treatment of NaN values
    wts_new = {key:wts[key] for key in wts if wts[key] >= wtsCutoff}
    renorm = norm/np.sum(wts_new.values())
    wts_new = {key:wts_new[key]*renorm for key in wts_new}
    return wts_new





######### Functions redefined to work for disk average ##########
    
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
    '''Extracts a spectrum summed over Titan's full disk (2 sigma PSF past the top of the atmosphere), converts it to radiance units and performs Doppler shift. '''
        
    # Generate frequency grid
    freqspec = ((np.arange(nspec) - i0) * df + f0) #in Hz
    
    # Convert and extract spectrum -- in CASA 4.0.1 imval can only extract regions, not individual pixels, but lower left pixel is in ['data'][0][0]
    freqspec = freqspec*(d_shift*1000+lightspeed)/lightspeed #doppler shift
    freqspec = freqspec/(lightspeed*100) #convert to cm-1 from Hz.

    r_diskavg = top_atm + (nsig_psf/2)*(a/(np.sqrt(2*np.log(2)))) #this defines Titan plus a buffer equal to 2*sigma_psf. This defines how big our average will be
    print('Averaging over a circular region with radius r = '+str(r_diskavg)+' km')
    print('This is equal to the radius of Titan including atmosphere + '+str(nsig_psf)+'*sigma_psf')
    
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

    return freqspec,valspec

def computeWeights_diskavg():
    """ Returns weightings of a disk average to be used in .spx header """

    #create grid z where each point represents radius from center of Titan
    lim = top_atm+a
    xx = slice(-lim,lim+dx,dx)
    y,x = np.mgrid[xx,xx]
    z = np.sqrt(x**2+y**2) #z now contains radii from the center
    
    midpoints = 0.5 * (radii[1:] + radii[:-1])
    
    #Change values of z to the angle corresponding to those radii
    angles = [np.degrees(np.arcsin(float(r)/float(top_atm))) for r in midpoints] #top of atmosphere emission angle
    for i in range(len(angles)):
        z[np.logical_and(z >= radii[i], z < radii[i+1])] = angles[i]
    z[z >= radii[-1]] = float('NaN')

    z_flat = np.copy(z)*0. + 1.
    
    #compute normalized weights. here much simpler than in the spatially resolved one because don't need to convolve with Gaussian. Just assume all flux is captured and Titan is a nice well-behaved sphere.
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

#Define, extract, and calculate constants
lightspeed = 299792458.0 #m/s
cm2hz = 29.9792458 * 1e9

subobslat = np.deg2rad(subobslat)
ccw = np.deg2rad(ccw)

if diskavg:
    pixels = [cenpix]

[beamx,beamy,theta,refx,refy,pixszx,pixszy,nspec,f0,df,i0] = casa_extract(img) #extracting image parameters from CASA

Titan_dist = Titan_dist*149597870. #to km
theta = np.radians(theta+90) #change to what's required by computeWeights
a = Titan_dist*np.radians(np.arcsin(beamx/3600)) #major axis at half-max in km
b = Titan_dist*np.radians(np.arcsin(beamy/3600)) #minor axis at half-max in km

#for each grid point cx,cy extract spectrum and weights
spx = {}
for (i,j) in pixels:
    print('-----------------------------------------')
    print('Pixel to extract: ('+str(i)+', '+str(j)+')')
    pixx=np.round(i)
    pixy=np.round(j)
    if not diskavg:
        (freq,flux) = spEx(img,pixx,pixy) #extract spectrum
        wts,meanlat = computeWeights(i,j) #compute weights
    else:
        (freq,flux) = spEx_diskavg(img,pixx,pixy) #extract spectrum
        wts = computeWeights_diskavg() #compute weights
        meanlat = 0.0
    if wtsCutoff > 0:
        wts = wtsReduce(wts,wtsCutoff)
    spx[(i,j)] = (wts,freq,flux,meanlat)
    plt.plot(freq,flux)
    plt.show()
    
#output to file in .spx format
with open(outfile,'w') as f:
    for location,spec in sorted(spx.items()):
        # Decimate the output if desired
        nuout  =  spec[1][0::skip]
        fout   =  spec[2][0::skip]
        f.write(str(np.float32(resfact*(spx[tuple(pixels[0])][1][1]-spx[tuple(pixels[0])][1][0])))+'     '+str(spec[3])+'     0.0     1\n')
        f.write(str(len(nuout))+'\n')
        f.write(str(len(spec[0]))+'\n')
        for key,val in sorted(spec[0].items()):
            f.write('      0.00000      0.00000      %7.4f      %7.4f      180.000     %8.6f\n' %(key,key,val))
        for i in range(len(nuout)):
            f.write('%11.9f  %12.5e  %12.5e\n' % (nuout[i],fout[i],jy2rad(sigma/1000.)))
f.close()
