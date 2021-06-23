import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection

from astropy import units as u
from astropy.table import Table, unique
from astropy.time import Time

import lightkurve as lk

from tqdm import tqdm

from scripts import *

import warnings
warnings.filterwarnings("ignore")

import glob, os

plt.rcParams['font.size'] = 20
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['savefig.dpi']= 300             #72 
# mpl.rcParams["font.family"] = "Computer Modern Roman"
colours = mpl.rcParams['axes.prop_cycle'].by_key()['color']

### load data

data = Table.read('joined_lofar.csv',format='ascii')

xray = np.array(data['Soft X-ray Lum (x1E28 ergs/s)'])
radio = np.array(data['LOFAR Lum (x1E14 ergs/s/Hz)'])
dradio = np.array(data['Uncert. Lofar Lum'])
flarerate = np.array(data['Flare Rate'])
names = np.array(data['Name'])

quiescent = ['GJ 625', 'GJ 450', 'GJ 1151', 'LP 169-22', 'G 240-45']
binaries = ['DG CVn', 'CR Dra']

savedir = '../results/final/'

# iterate over stars

# all_lcs = []
# all_preds = []

for name in names:
    # if name != 'WX Uma':
    #     continue

    print('Doing %s' % name)

    flares = Table.read('%sflares_%s.csv' % (savedir,name.replace(' ','_').lower()),format='ascii')
    flares = unique(flares)
    flares.write('%sflares_%s.csv' % (savedir,name.replace(' ','_').lower()),format='ascii') # catch and fix this

    tic = data['TIC'][np.where(data['Name']==name)]
    tics, time, flux, errs, sects, data_all = load_lightcurve(name)
    lcs = lk.collections.LightCurveCollection(data_all).stitch().normalize().remove_nans()

    avg_preds = []
    for j in range(len(sects)):
        avg_preds.append(Table.read('%savg_preds_%s_%d.csv' % (savedir,name.replace(' ','_').lower(), j))['avg_preds'].data)
    avg_preds = np.hstack(np.array(avg_preds))
    # all_preds.append(avg_preds[0])
    
    for k, flare in enumerate(tqdm(flares)):
        tpeak, ed_s = flare['tpeak'], flare['ed_s']
        lctime = lcs.time.value

        mm = (lctime>tpeak-0.25) * (lctime <tpeak+0.25)
        yrange = np.max(lcs[mm].flux)-np.min(lcs[mm].flux)
        ymin, ymax = np.min(lcs[mm].flux)-0.1*yrange, np.max(lcs[mm].flux)+0.1*yrange

        plt.clf()
        
        fig, ax = plt.subplots(1,1,figsize=(12.0,8.0))    
        ax.set_rasterized(True)
        ax.scatter(lctime[mm],lcs.flux[mm],c=avg_preds[mm],label=name+' flare '+str(k),vmin=0, vmax=1, s=10)
        ax.plot(lctime[mm],lcs[mm].flux,'-',alpha=0.25,c=colours[1])
        ax.set_xlim(tpeak-0.25,tpeak+0.25)
        ax.set_ylim(ymin,ymax)
        ax.axvline(tpeak,alpha=0.25)
        ax.axvspan(tpeak-ed_s/3600./24.,tpeak+ed_s/3600./24.,alpha=0.05)
        ax.set_title('%s: flare %d' % (name,k))
        plt.savefig('%s/vetting/%s_flare_%d.png' % (savedir,name.replace(' ','_').lower(),k),
                   bbox_inches='tight',rasterized=True)

print('Finished all stars!')