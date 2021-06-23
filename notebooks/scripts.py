import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from tqdm import tqdm_notebook

import lightkurve as lk
import stella

from astropy import units as u
from astropy.time import Time, TimeDelta

import glob, os, sys

from astropy.table import Table, unique
plt.rcParams['font.size'] = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['savefig.dpi']= 300             #72 


import more_itertools as mit


'''-----------------------------------------------
Port of Adina's script for CR Dra analysis with
stella
-----------------------------------------------'''

model_dir = '../data/run01/'
# model_dir = '/Users/benjaminpope/.stella/models/'

models = [os.path.join(model_dir,i) for i in
          os.listdir(model_dir) if i.endswith('.h5')]


'''-----------------------------------------------
First define the colourmap
-----------------------------------------------'''

from matplotlib.colors import LinearSegmentedColormap
from pylab import cm

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

false_pos = {'2MASS J09481615+5114518': [], # 
            '2MASS J14333139+3417472': [7],
            'CR Dra': [28,81,172], # one before 189 that's kinda big, one between 76 and 77, 81 is a duplicate
            'CW UMa': [4,24,], # one missing between 8 and 9
            'DG CVn': [10,12,2,21,22,26,27,28,29,34,79], # regularly identifying rotational modulation as flares, missing 2 between 32 and 33, one after 62 before 63
            'DO Cep': [1,20,28,30,36,38],
            'G 240-45': [0,1,11,26,31,33,41,43,59,6,60,61,7,70,71,], # missing a big between 31 and 32, most of them are close to the noise
            'GJ 1151': [1,5,7,9],
            'GJ 3861': [39,4,52,],
            'GJ 450': [],
            'GJ 625': [], 
            'LP 169-22': [1,12,2,7],
            'LP 212-62': [11,12,14,18,19,2,23,25,26,30,4],
            'LP 259-39': [],
            'WX Uma': []}

parula = LinearSegmentedColormap.from_list('parula', cm_data)

parula_colors = []
cmap = cm.get_cmap(parula, 2)
for i in range(cmap.N):
    rgb = cmap(i)[:3]
    parula_colors.append(mpl.colors.rgb2hex(rgb))
parula_colors = np.array(parula_colors)

def download_lightcurve(starname,radius=10.):
    if starname == 'WX Uma' or starname == 'TIC 252803603':
          print('Doing a special reduction for WX UMa')
          tpf = lk.search_targetpixelfile('TIC 252803603',exptime=120).download()
          corrector = lk.TessPLDCorrector(tpf)
          data_all = [corrector.correct()]
    else:
      search = lk.search_lightcurvefile(starname,radius=radius,exptime=120)
      search = search[np.where(search.target_name==search.target_name[0])]
      data_all = search.download_all()
    return data_all


def load_lightcurve(starname,radius=10.,from_saved=True,save=True):
    # first look for files
    if from_saved:
        try:
            fnames = glob.glob('../data/lcs/%s*.fits' % (starname.replace(' ','_').lower()))
            fnames.sort()
            sects = [fname[-12:-8] for fname in fnames]
            if len(fnames)>0:
                data_all = []
                for fname in fnames:
                    d = lk.open(fname).remove_nans().normalize() 
                    d.targetid = int(fname[-12:-8])
                    data_all.append(d)
                    d.targetid = int(d.meta['LABEL'])# ach gotta fix this I fucked it up

                print('Loaded from saved files',fnames)
            else:
                print('No saved files!')
                return None
        except:
            data_all = download_lightcurve(starname,radius=radius)
            print('Downloaded lightcurve!')
    else: # if no files, download 
        data_all = download_lightcurve(starname,radius=radius)
        print('Downloaded lightcurve!')
        
    # read these out into the format that stella likes 
    tics, time, flux, errs, sects = [] ,[] ,[], [], []

    for data in data_all:
        try:
          d = data.PDCSAP_FLUX.remove_nans().normalize()
        except:
          d = data.remove_nans().normalize()
        qq = (d.quality==0)
        d = d[qq]
        time.append(d.time.value)
        flux.append(d.flux)
        errs.append(d.flux_err)
        tics.append(d.targetid)
        sects.append(d.sector)
        if save:
            savename = '../data/lcs/%s_s%04d_lc.fits' % (starname.replace(' ','_').lower(),d.sector)
            d.to_fits(savename,overwrite=True)

    return tics, time, flux, errs, sects, data_all

def get_rotation_period(tics,time,flux,errs):
    mProt = stella.MeasureProt(IDs=tics,
                           time=time,
                           flux=flux,
                           flux_err=errs)

    mProt.run_LS()
    period = mProt.LS_results['avg_period_days'].data[0] + 0.0
    return period

def run_cnn(tics,time,flux,errs):
    cnn = stella.ConvNN(output_dir='.')
    preds = []

    for model in models:
        cnn.predict(modelname=model,
                    times=time,
                    fluxes=flux,
                    errs=errs)
        preds.append(cnn.predictions)
    preds = np.array(preds)

    avg_preds = np.copy(time)

    for i in range(len(time)):
        temp = np.zeros((len(models), len(time[i])))
        for j in range(len(models)):
            temp[j] = preds[j][i]
        ap = np.nanmedian(temp, axis=0)
        avg_preds[i] = ap

    return avg_preds 

def get_flares(tics,time,flux,avg_preds,errs):
    ff = stella.FitFlares(id=tics,
                      time=time,
                      flux=flux,
                      predictions=avg_preds,
                      flux_err=errs)
    ff.identify_flare_peaks(threshold=0.6)
    return ff.flare_table

def get_flare_rate(time,flare_table,name=None):
    totaltime = 0
    for i in range(len(time)):
        totaltime += (len(time[i])*2)
    totaltime = (totaltime*u.minute).to(u.day)
    
    return np.nansum(flare_table['prob'])/totaltime

def remove_false_positives(time,flare_table,name):
      flare_table.remove_rows(false_pos[name])
      print('Removing %d flares' % len(false_pos[name]))
      return flare_table


def group_sectors(data_all):
    sectors = [d.sector for d in data_all]
    groups = [list([sectors.index(k) for k in j]) for j in mit.consecutive_groups(sorted(list(set(sectors))))]
    return groups,sectors

def do_plots(tics,time,flux,avg_preds,errs,data_all,zoom=True):
    groups,sectors = group_sectors(data_all)
    ngroups = len(groups)
    width_ratios = [len(group) for group in groups] 
    fig, axes = plt.subplots(ncols=ngroups, figsize=(ngroups*7,4),
                               sharey=True, gridspec_kw={'width_ratios':width_ratios})
    for j, g in enumerate(groups):
        if len(groups)>1:
            ax = axes[j]
        else:
            ax = axes
        if j == 0:
            ax.set_ylabel('Normalized Flux')
        for i in g:
            ax.scatter(time[i], flux[i], c=avg_preds[i],
                        vmin=0, vmax=1, s=6)
        if len(g)==1:
            ax.set_title('Sector '+str(sectors[g[0]]),y=1.01)
        else:
            ss = [sectors[s] for s in g]
            ax.set_title('Sectors ' + ", ".join([str(s) for s in ss]),y=1.01)
        ax.set_xlabel('TJD')
    yrange = np.percentile(np.hstack(flux),(2,50,98))
    
    if zoom:
      lims = (yrange[1]-1.0*(yrange[2]-yrange[0]), yrange[1]+1.0*(yrange[2]-yrange[0]))
      plt.ylim(*lims)
    plt.subplots_adjust(wspace=0.1)

def simultaneous_plots(tics,time,flux,avg_preds,errs,data_all,tstart,limit=True):
    dates = lk.btjd_to_astropy_time(np.hstack(time))
    t = Time(tstart, format='isot', scale='utc')
    dt = TimeDelta(3600.*8., format='sec')
    tfinish = t+dt

    fig = plt.figure(figsize=(8.0,6.0))
    plt.scatter(dates.decimalyear-2020,np.hstack(flux),c=np.hstack(avg_preds),
                        vmin=0, vmax=1, s=6)
    if limit:
      plt.xlim(t.decimalyear-2020,tfinish.decimalyear-2020)
    plt.axvline(t.decimalyear-2020)
    plt.axvline(tfinish.decimalyear-2020)
    plt.xlabel('Decimal Year - 2020')
    plt.ylabel('Flux')
    # plt.ylim(0.98,1.01)
    # plt.xlim(t.decimalyear-2020-0.05,tfinish.decimalyear-2020+0.05)
    plt.colorbar()
    # plt.title(name+' Simultaneous TESS')

