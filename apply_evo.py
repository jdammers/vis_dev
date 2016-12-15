import pylab as pl
import numpy as np
def plot_evo(evokeds, evt_list, fn_fig_out):
    
    pl.figure('Compare', figsize=(16, 10))
    sub_fig = [221, 222, 223, 224]
    ax = pl.subplot(sub_fig[0])
    evokeds[0].pick_types(meg=True, ref_meg=False)
    evokeds[0].plot(axes=ax)
    ylim = dict(mag=ax.get_ylim()) 
    #evokeds[0].plot(axes=ax, ylim=ylim)       
    ymin, ymax = ylim['mag'][0], ylim['mag'][1]
    ax.set_ylim((ymin, ymax))
    ax.vlines(0, ymin, ymax)
    pl.title("%s" %(evt_list[0]))
    i = 1
    for evoked in evokeds[1:]:
        ax = pl.subplot(sub_fig[i])
        evoked.pick_types(meg=True, ref_meg=False)
        evoked.plot(axes=ax, ylim=ylim)
        ax.set_ylim((ymin, ymax))
        ax.vlines(0, ymin, ymax)
        pl.title("%s" %(evt_list[i]))
        #textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f\nRaw file name: %s' %(len(ecg_eve), tmin, tmax, raw)
        #ax.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=12, 
         #       verticalalignment='top', bbox=props)
        i = i + 1
    pl.tight_layout()
    pl.savefig(fn_fig_out, format='tif')
    pl.close('Compare')        

def _square_ave(x):
    return np.mean(x ** 2, axis=0)
    
def plot_evo_allchs(evokeds, evt_list, fn_fig_out):
    pl.figure('Compare', figsize=(16, 10))
    sub_fig = [221, 222, 223, 224]
    ax = pl.subplot(sub_fig[0])
    evokeds[0].pick_types(meg=True, ref_meg=False)
    pl.plot(evokeds[0].times*1e3, _square_ave(evokeds[0].data), 'r', axes=ax)
    ylim = dict(mag=ax.get_ylim()) 
    ymin, ymax = ylim['mag'][0], ylim['mag'][1]
    ax.set_ylim((ymin, ymax))
    ax.vlines(0, ymin, ymax)
    pl.title("%s" %(evt_list[0]))
    i = 1
    for evoked in evokeds[1:]:
        ax = pl.subplot(sub_fig[i])
        evoked.pick_types(meg=True, ref_meg=False)
        ax.vlines(0, ymin, ymax)
        ax.set_ylim((ymin, ymax))
        pl.plot(evoked.times*1e3, _square_ave(evoked.data), 'r', axes=ax)
        pl.title("%s" %(evt_list[i]))
        i = i + 1
    pl.tight_layout()
    pl.savefig(fn_fig_out, format='tif')
    pl.close('Compare')  