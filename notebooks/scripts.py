import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm_notebook

import lightkurve as lk
import stella
from astropy import units as u

import glob, os, sys

from astropy.table import Table
plt.rcParams['font.size'] = 20

import more_itertools as mit


'''-----------------------------------------------
Port of Adina's script for CR Dra analysis with
stella
-----------------------------------------------'''
model_dir = '/Users/benjaminpope/Downloads/run01/'
models = [os.path.join(model_dir,i) for i in
          os.listdir(model_dir) if i.endswith('.h5')]



'''-----------------------------------------------
First define the colourmap
-----------------------------------------------'''

from matplotlib.colors import LinearSegmentedColormap
from pylab import *

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

parula = LinearSegmentedColormap.from_list('parula', cm_data)

parula_colors = []
cmap = cm.get_cmap(parula, 2)
for i in range(cmap.N):
    rgb = cmap(i)[:3]
    parula_colors.append(matplotlib.colors.rgb2hex(rgb))
parula_colors = np.array(parula_colors)


def load_lightcurve(starname,radius=1.):
    search = lk.search_lightcurvefile(starname,radius=radius)
    search = search[np.where(search.target_name==search.target_name[0])]
    data_all = search.download_all()
    tics, time, flux, errs, sects = [] ,[] ,[], [], []

    for data in data_all:
        d = data.PDCSAP_FLUX.remove_nans().normalize()
        time.append(d.time)
        flux.append(d.flux)
        errs.append(d.flux_err)
        tics.append(d.targetid)
        sects.append(d.sector)

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

def get_flare_rate(time,flare_table):
    totaltime = 0
    for i in range(len(time)):
        totaltime += (len(time[i])*2)
    totaltime = (totaltime*u.minute).to(u.day)
    return np.nansum(flare_table['prob'])/totaltime

def group_sectors(data_all):
    sectors = [d.sector for d in data_all]
    groups = [list([sectors.index(k) for k in j]) for j in mit.consecutive_groups(sorted(list(set(sectors))))]
    return groups,sectors

def do_plots(tics,time,flux,avg_preds,errs,data_all):
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
    lims = (yrange[1]-1.0*(yrange[2]-yrange[0]), yrange[1]+1.0*(yrange[2]-yrange[0]))
    plt.ylim(*lims)
    plt.subplots_adjust(wspace=0.1)
