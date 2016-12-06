import mne, glob
import matplotlib.pyplot as plt
import numpy as np
from mne.event import define_target_events
# view inspection and interpolation
#raw.interpolate_bads(reset_bads=True)

#Define correct epoch with trigger and response event_ids
conds_id = [(1, 8), (2, 64), (3, 64), (4, 8)]
conditions = ['LLst', 'LRst', 'RRst', 'RLst']

# Define the raw_data path for epoching
fn_list = glob.glob('./201195*,nr,ocarta-raw.fif')
baseline = (None, 0)
reject={'mag': 4e-12}

ltmin, ltmax = 0, 0.8 # The time range for fitting visual events
tmin, tmax = -0.2, 0.8 # The time range for epochs

for fn_raw in fn_list:
#fn_raw = '203867_Chrono01_110615_1516_1_c,rfDC,nr,ocarta-raw.fif'
#'201195_Chrono01_110516_1413_1_c,rfDC,nr,ocarta,evtW_RLst_bc-epo.fif'
    raw = mne.io.read_raw_fif(fn_raw, preload=True)
    #raw.plot(start=0, duration=120)
    tri_events = mne.find_events(raw, stim_channel='STI 014')
    res_events = mne.find_events(raw, stim_channel='STI 013')
    
    #Get the events list
    sfreq = raw.info['sfreq']
    levents = np.array(list(tri_events) + list(res_events))
    cons_epochs = []
    i = 0
    # Based on the correct event couples and fixed time range, to find the appropriate epochs
    for cond_id in conds_id:
        tri_id = cond_id[0]
        res_id = cond_id[1]
        events_, lag = define_target_events(levents, tri_id, res_id,
                                            sfreq, ltmin, ltmax)
        picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False,
                                exclude='bads')
        epochs = mne.Epochs(raw, events_, tri_id, tmin,
                            tmax, picks=picks, baseline=(None, 0),
                            reject=dict(mag=4e-12))
        fn_epo = fn_raw[:fn_raw.rfind('-raw.fif')] + ',evtW_%s_bc-epo.fif' %conditions[i]
        epochs.save(fn_epo)
        i = i + 1
    
###############################################################################
# Reject bad epochs based on viusal inspection
# ----------------------
fn_epos = glob.glob('./203867*ocarta,evtW_*_bc-epo.fif')
fn_epo = fn_epos[0]
#fn_ave = fn_epo[:fn_epo.rfind('-epo.fif')] + '-ave.fif' 
epo = mne.read_epochs(fn_epo, preload=True)
epo.plot(block=True, n_epochs=40)
epo.interpolate_bads(reset_bads=True)
epo.save(fn_epo)
