import numpy as np
import matplotlib.pyplot as plt

from scripts import *

from astropy import units as u
import lightkurve as lk
from matplotlib.collections import LineCollection
from tqdm import tqdm as tqdm

import glob, os

import warnings
warnings.filterwarnings("ignore")

from astropy.table import Table, Column, unique

from astroquery.simbad import Simbad
Simbad.reset_votable_fields()
Simbad.add_votable_fields('sptype','id(tic)')

plt.rcParams['font.size'] = 20

'''------------------------------------------------
Script to re-reduce all the Callingham detections
for the paper.
------------------------------------------------'''


targets = Table.read('../data/driessen2.csv')

names = targets['Name']
print(targets.keys())
print(names)

savedir = '../results/driessen2/'

saved_files = glob.glob(savedir+'*')

failures = []


for j in range(len(names)):
    try:
        target = targets[j]
        name = names[j].strip()
        # if target['TESS?'] == 'N': # check if it is in fact in the TESS field
        #     print('No data for %s, continuing' % str(name))
        #     continue

        if '%s%s_output.txt' % (savedir,name.replace(' ','_').lower()) in saved_files:
            print('\n\nAlready done',name)
            continue

        print('\n\nDoing target %d/%d: %s' % (j, len(names),name))


        # if  np.ma.is_masked(target['TIC ID']) is False: # some targets do not resolve normally but I have manually identified their TIC ID
        #     print('Loading Light Curve for TIC %d' %  target['TIC ID'])
        #     tics, time, flux, errs, sects, data_all = load_lightcurve('TIC %d' % target['TIC ID'])
        #     tic = target['TIC ID']
        # else:
        print('Loading Light Curve for name %s' %  name)

        tics, time, flux, errs, sects, data_all = load_lightcurve(name,from_saved=False)
        tic = tics[0]

        print('Loaded light curve!\n')

        nsectors = len(time)

        print('Getting rotation period')
        period = get_rotation_period(tics,time,flux,errs)
        print('Period:',period,'d\n')

        # try:
        #     avg_preds = []
        #     for j in range(len(time)):
        #         avg_preds.append(Table.read('%savg_preds_%s_%d.csv' % (savedir,name.replace(' ','_').lower(), j))['avg_preds'].data)
        #     print('Loaded previously saved avg_preds')
        # except:
        print('Running CNN')
        avg_preds = run_cnn(tics,time,flux,errs)
        for j, pred in enumerate(avg_preds):
          col1, col2 = Column(time[j],name='time'), Column(pred,name='avg_preds')
          Table([col1,col2]).write('%savg_preds_%s_%d.csv' % (savedir,name.replace(' ','_').lower(), j))
        print('Saved avg_preds')

        flare_table = unique(get_flares(tics,time,flux,avg_preds,errs))
        flare_table.write('%sflares_%s.csv' % (savedir,name.replace(' ','_').lower()),format='ascii')
        print('Saved flare table to %sflares_%s.csv' % (savedir,name.replace(' ','_').lower()))

        n_tot = len(flare_table)
        print('Filtering flares')
        flare_table = filter_flares(data_all,flare_table)
        # flare_table = remove_false_positives(time,flare_table,name) # comment out on first run
        nflares = len(flare_table)
        print('Filtered flares: %d out of %d are legit' % (nflares,n_tot))

        flare_rate = get_flare_rate(time,flare_table)
        print('Flare rate:',flare_rate)

        do_plots(tics,time,flux,avg_preds,errs,data_all)
        figname = '%sflare_lc_%s.png' % (savedir,name.replace(' ','_').lower())
        plt.savefig(figname,bbox_inches='tight')
        print('Saved zoomed plots to %s' % figname)

        do_plots(tics,time,flux,avg_preds,errs,data_all,zoom=False)
        figname = '%sflare_lc_nozoom_%s.png' % (savedir,name.replace(' ','_').lower())
        plt.savefig(figname,bbox_inches='tight')
        print('Saved no-zoom plots to %s' % figname)

        f = open('%s%s_output.txt' % (savedir,name.replace(' ','_').lower()),'w')
        f.write('%s\n%f\n%d\n%f\n%d\n%d\n' % (name,period,nflares,flare_rate.value,nsectors,int(tic)))
        f.close()
        print('Saved fit to %s%s_output.txt' % (savedir,name.replace(' ','_').lower()))
    except:
        print('\nFailed on %s' % name)
        failures.append(name)

print('Finished all light curves!')
if len(failures) > 0:

    print('Failed on',*failures)