import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
import lightkurve as lk
from matplotlib.collections import LineCollection
from tqdm import tqdm_notebook

import glob, os

import warnings
warnings.filterwarnings("ignore")


from scripts import *

from astropy.table import Table
plt.rcParams['font.size'] = 20

targets = Table.read('names_best_cands_4sig_stokesv_clean_leakage_gaia_propermotion_applied stars_added_2019_10_24.fits') # file 1
targets = Table.read('m_dwarf_dections_27_03_20_correct_fluxes.fits')
names = targets['common_name']

savedir = 'results/'

saved_files = glob.glob(savedir+'*')

failures = []

for j in range(len(names)):
    try:
        target = targets[j]
        name = names[j].strip()
        if '%s%s_output.txt' % (savedir,name.replace(' ','_').lower()) in saved_files:
            print('\n\nAlready done',name)
            continue

        print('\n\nDoing target %d/%d: %s' % (j, len(names),name))

        print('Loading Light Curve')
        coords = '%f +%f' % (target['ra'],target['dec'])
        tics, time, flux, errs, sects, data_all = load_lightcurve(coords)
        print('Loaded light curve!\n')

        print('Getting rotation period')
        period = get_rotation_period(tics,time,flux,errs)
        print('Period:',period,'d\n')

        print('Running CNN')
        avg_preds = run_cnn(tics,time,flux,errs)

        flare_table = get_flares(tics,time,flux,avg_preds,errs)
        flare_table.write('%sflares_%s.csv' % (savedir,name.replace(' ','_').lower()),format='ascii')
        print('Saved flare table to %sflares_%s.csv' % (savedir,name.replace(' ','_').lower()))

        flare_rate = get_flare_rate(time,flare_table)
        print('Flare rate:',flare_rate)

        do_plots(tics,time,flux,avg_preds,errs,data_all)
        figname = '%sflare_lc_%s.png' % (savedir,name.replace(' ','_').lower())
        plt.savefig(figname,bbox_inches='tight')
        print('Saved plots to %s' % figname)

        f = open('%s%s_output.txt' % (savedir,name.replace(' ','_').lower()),'w')
        f.write('%s\n%f\n%f\n' % (name,period,flare_rate.value))
        f.close()
        print('Saved fit to %s%s_output.txt' % (savedir,name.replace(' ','_').lower()))
    except:
        print('\nFailed on %s' % name)
        failures.append(name)

print('Finished all light curves!')
print('Failed on',*failures)