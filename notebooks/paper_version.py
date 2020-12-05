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

'''------------------------------------------------
Script to re-reduce all the Callingham detections
for the paper.
------------------------------------------------'''

targets = Table.read('../data/lofartesscallingham.csv',format='ascii') # https://docs.google.com/spreadsheets/d/1k2le1xcRh-fbvqsZdP5cu_4tuGaiuwlf1NxZHdaXN1A/edit#gid=0
names = targets['Name']
print(targets.keys())

savedir = 'results/reanalysis/'

saved_files = glob.glob(savedir+'*')

failures = []


stars_simultaneous = {'EW Dra': '2020-04-02T22:55:34',
                      'HD 233153': '2019-12-12T19:41:00',
                      'HD 37394':'2019-12-12T19:41:00',
                      'Tau Boo': '2019-07-06T14:26:09',
                      'Wolf 1069': '2019-08-05T19:05:52',
                      'Ross 567': '2019-11-28T20:11:00',
                      'G262-15': '2019-08-17T18:06:56',
                      'UCAC4 642-113039':'2019-09-26T17:09:42',
                      'G258-33':'2019-11-10T10:48:00',
                      'G 227-22':'2019-11-10T10:48:00',
                      'IRAS21500+5903':'2019-09-18T18:11:00',
                      '54 Psc':'2019-10-19T18:11:00',
                      'HD 10780':'2019-11-03T18:31:10',
                      'UCAC4 655-108663': '2019-10-05T17:11:00',
                      'HD 223778 B': '2019-11-20T17:11:00',
                      'WX UMa': '2019-11-29T05:11:00',
                      'GJ 450':'2020-03-16T20:11:00',
                      'GJ 3861':'2020-02-19T23:57:00',
                      '2MASS J09481615+5114518':'2020-01-31T20:45:40',
                      'LP 212-62':'2020-02-04T20:48:20'
}


for j in range(len(names)):
    # try:
    target = targets[j]
    name = names[j].strip()
    if target['TESS?'] == 'N': # check if it is in fact in the TESS field
        print('No data for %s, continuing' % str(name))
        continue

    if '%s%s_output.txt' % (savedir,name.replace(' ','_').lower()) in saved_files:
        print('\n\nAlready done',name)
        continue

    print('\n\nDoing target %d/%d: %s' % (j, len(names),name))


    if  np.ma.is_masked(target['TIC ID']) is False: # some targets do not resolve normally but I have manually identified their TIC ID
        print('Loading Light Curve for TIC %d' %  target['TIC ID'])
        tics, time, flux, errs, sects, data_all = load_lightcurve('TIC %d' % target['TIC ID'])
        tic = target['TIC ID']
    else:
        print('Loading Light Curve for name %s' %  name)

        tics, time, flux, errs, sects, data_all = load_lightcurve(name)
        tic = tics[0]

    print('Loaded light curve!\n')

    nsectors = len(time)

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
    print('Saved zoomed plots to %s' % figname)

    do_plots(tics,time,flux,avg_preds,errs,data_all,zoom=False)
    figname = '%sflare_lc_nozoom_%s_.png' % (savedir,name.replace(' ','_').lower())
    plt.savefig(figname,bbox_inches='tight')
    print('Saved no-zoom plots to %s' % figname)

    if name in stars_simultaneous.keys():
        tstart = stars_simultaneous[name]
        simultaneous_plots(tics,time,flux,avg_preds,errs,data_all,tstart)
        figname = '%ssimultaneous_%s_.png' % (savedir,name.replace(' ','_').lower())
        plt.savefig(figname,bbox_inches='tight')
        print('Saved simultaneous plots to %s' % figname)

    f = open('%s%s_output.txt' % (savedir,name.replace(' ','_').lower()),'w')
    f.write('%s\n%f\n%f\n%d\n%d\n' % (name,period,flare_rate.value,nsectors,tic))
    f.close()
    print('Saved fit to %s%s_output.txt' % (savedir,name.replace(' ','_').lower()))
    # except:
    #     print('\nFailed on %s' % name)
    #     failures.append(name)

print('Finished all light curves!')
if len(failures) > 0:

    print('Failed on',*failures)