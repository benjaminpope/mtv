import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection

from astropy import units as u
from astropy.table import Table
from astropy.time import Time

import lightkurve as lk

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import glob, os

from astropy.table import Table

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

savedir = 'results/simultaneous/'

# iterate over stars

# all_lcs = []
# all_preds = []

for name in names:
    if name != 'WX Uma':
        continue

    print('Doing %s' % name)

    flares = Table.read('%sflares_%s.csv' % (savedir,name.replace(' ','_').lower()),format='ascii')

    tic = data['TIC'][np.where(data['Name']==name)]
    search = lk.search_lightcurvefile('TIC %d' % tic) # why is the TIC wrong?

    if name == 'WX Uma':
        print('Doing a special reduction for WX UMa')
        tpf = lk.search_targetpixelfile('TIC 252803603').download()
        corrector = lk.TessPLDCorrector(tpf)
        lcs = corrector.correct().remove_nans().normalize()
    else:
        lcs = search.download_all().stitch().remove_nans().normalize()
        # all_lcs.append(lcs)
    avg_preds = []
    for j in range(len(search)):
        avg_preds.append(Table.read('avg_preds_%s_%d.csv' % (name.replace(' ','_').lower(), j))['avg_preds'].data)
    avg_preds = np.hstack(np.array(avg_preds))
    # all_preds.append(avg_preds[0])
    
    for k, flare in enumerate(tqdm(flares)):
        tpeak, dur = flare['tpeak'], flare['dur_min']/60./24.

        mm = (lcs.time>tpeak-0.25) * (lcs.time <tpeak+0.25)
        yrange = np.max(lcs[mm].flux)-np.min(lcs[mm].flux)
        ymin, ymax = np.min(lcs[mm].flux)-0.1*yrange, np.max(lcs[mm].flux)+0.1*yrange

        plt.clf()
        
        fig, ax = plt.subplots(1,1,figsize=(12.0,8.0))    
        ax.set_rasterized(True)
        ax.scatter(lcs.time[mm],lcs.flux[mm],c=avg_preds[mm],label=name+' flare '+str(k),vmin=0, vmax=1, s=10)
        ax.plot(lcs[mm].time,lcs[mm].flux,'-',alpha=0.25,c=colours[1])
        ax.set_xlim(tpeak-0.25,tpeak+0.25)
        ax.set_ylim(ymin,ymax)
        ax.axvline(tpeak,alpha=0.25)
        ax.axvspan(tpeak-dur,tpeak+dur,alpha=0.05)
        ax.set_title('%s: flare %d' % (name,k))
        plt.savefig('%s/vetting/%s_flare_%d.png' % (savedir,name.replace(' ','_').lower(),k),
                   bbox_inches='tight',rasterized=True)