import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lightkurve as lk
from matplotlib.collections import LineCollection
from tqdm import tqdm_notebook

import glob, os

import warnings
warnings.filterwarnings("ignore")

from scripts import *

from astropy.table import Table, Column, unique
plt.rcParams['font.size'] = 20

'''------------------------------------------------
Script to re-reduce all the Callingham detections
for the paper.
------------------------------------------------'''

targets = Table.read('../data/lofartesscallingham.csv',format='ascii') # https://docs.google.com/spreadsheets/d/1k2le1xcRh-fbvqsZdP5cu_4tuGaiuwlf1NxZHdaXN1A/edit#gid=0
targets = targets[targets['Type']=='M Dwarf'] # restrict it to m dwarfs
targets = targets[targets['TESS?'] != 'N'] # restrict it to have tess data
names = targets['Name']
print(targets.keys())
print(names)

savedir = '.../results/final/'

saved_files = glob.glob(savedir+'*')

# failures = []

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
periods, flare_tots, flare_rates, sectors, tics = [], [], [], [], []

for j in range(len(names)):
  # try:
  target = targets[j]
  name = names[j].strip()
  if target['TESS?'] == 'N': # check if it is in fact in the TESS field
      print('No data for %s, continuing' % str(name))
      continue

  fname_in = '%s%s_output.txt' % (savedir,name.replace(' ','_').lower())

  f = open(fname_in)
  (name,period,nflares,flare_rate,nsectors,tic) = f.read().splitlines()
  f.close()
  if name == 'WX UMa':
    nflares = 9   
    totaltime = 0
    for i in range(len(time)):
        totaltime += (len(time[i])*2)
    totaltime = (totaltime*u.minute).to(u.day)

    flare_rate = 9./totaltime
    print('Special 9 flares by eye for WX UMa')

  periods.append(period)
  flare_rates.append(flare_rate)
  flare_tots.append(nflares)
  sectors.append(nsectors)
  tics.append(tic)

periods = Column(np.array(periods),name='Rotation Period')
flare_tots = Column(np.array(flare_tots),name='N Flares')
flare_rates = Column(np.array(flare_rates),name='Flare Rate')
sectors = Column(np.array(sectors),name='N Sectors')
tics = Column(np.array(tics),name='TIC')

joined_lofar = targets['Name','LOFAR Lum (x1E14 ergs/s/Hz)','Soft X-ray Lum (x1E28 ergs/s)','Uncert. Lofar Lum','Literature Rotation Period']
joined_lofar.add_columns([periods,flare_rates,flare_tots,sectors,tics])

joined_lofar.write('joined_lofar.csv',format='ascii')

