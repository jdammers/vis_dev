'''1. preprocessing pipeline:
      remove electrical noise(50Hz) and noise below 5Hz
      
'''
from jumeg.jumeg_preprocessing import apply_filter
from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising
from jumeg.decompose import ocarta
from jumeg import jumeg_plot
import glob, mne
from mne.event import define_target_events
import numpy as np
from mne.epochs import equalize_epoch_counts
###########################
# Set your data path here
#--------------------------
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
MIN_path = subjects_dir + 'fsaverage'

###########################
# Set the parameters
#--------------------------
do_pre = False # data preprocessing
do_inter = False
do_fil = False # data filtering
do_epo = False # Crop data into epochs
avg_fil_epos = False # average filtered epos


res_name, tri_name = 'STI 013', 'STI 014'
ev_id = 1
###################################
# Raw data preprocessing
#----------------------------------


if do_pre:
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*rfDC-raw.fif')
    for fn_raw in fn_list:
        #import noise_reducer and plot_power_spectrum function
        #apply noise reducer for 50 Hz (and harmonics)
        fn_raw_nr = fn_raw[:fn_raw.rfind('-raw.fif')] + ',nr-raw.fif'
        noise_reducer(fn_raw, refnotch=50, detrending=False, fnout=fn_raw_nr) 
        noise_reducer(fn_raw_nr, refnotch=60, detrending=False, fnout=fn_raw_nr)
        #apply noise reducer for frequencies below 5 Hz
        noise_reducer(fn_raw_nr, reflp=5, fnout=fn_raw_nr)# plot power spectrum
        fn_power_spect = fn_raw_nr[:fn_raw_nr.rfind('-raw.fif')] + ',denoising'
        plot_denoising([fn_raw, fn_raw_nr], stim_name=tri_name,
                    event_id=ev_id, show=False, fnout=fn_power_spect,
                    tmin_stim=-0.4, tmax_stim=0.4)
    
    
    
''' Before data filtering, please make sure ECG and EOG cleared, and refer the script
    '2_bad_icacheck.py'
'''

###################################
# Interpolate bad channels
#----------------------------------
if do_inter:
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*,nr,ocarta-raw.fif')
    for fn_raw_cl in fn_list:
        raw = mne.io.Raw(fn_raw_cl, preload=True)
        if raw.info['bads'] != []:
            print('bad channels are:', raw.info['bads'])
            raw.interpolate_bads(reset_bads=True)
            print('bad channels are:', raw.info['bads'])
            raw.save(fn_raw_cl, overwrite=True)
        del raw
###################################
# Filter the data
#----------------------------------

if do_fil:
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*,nr,ocarta-raw.fif')
    for fn_raw_cl in fn_list:
        fn_raw_fil = fn_raw_cl[:fn_raw_cl.rfind('-raw.fif')] + ',fibp1-45-raw.fif'
        apply_filter(fn_raw_cl, flow=1, fhigh=45, order=4, njobs=4)
        jumeg_plot.plot_compare_brain_responses(fn_raw_cl, fn_raw_fil, event_id=ev_id, stim_name='trigger')
        
        
###################################
# Crop the data
#----------------------------------
if do_epo:
    conds_id = [(1, 8), (2, 64), (3, 64), (4, 8)]
    conditions = ['LLst', 'LRst', 'RRst', 'RLst']
    # Define the raw_data path for epoching
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*,nr,ocarta,fibp1-45-raw.fif')   
    tstmin, tstmax = 0.1, 0.8 # The time range for searching response events
    rstmin, rstmax = -0.8, -0.1 # The time range for searching trigger events
    tmin, tmax = -0.3, 0.6 # The time range for trigger epochs
    rtmin, rtmax = -0.8, 0.1 #
    
    for fn_raw in fn_list:
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
            picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False,
                                    exclude='bads')
                                    
            # Take the triger events as the reference events                         
            events_, lag = define_target_events(levents, tri_id, res_id,
                                                sfreq, tstmin, tstmax)                        
            epochs = mne.Epochs(raw, events_, tri_id, tmin,
                                tmax, picks=picks, baseline=None,
                                reject=dict(mag=4e-12))
            #Crope the baseline time range
            epochs.load_data()
            epo_base = epochs.copy().crop(None, 0)
            line_base = epo_base.get_data().mean(axis=-1, keepdims=True)
            # Baseline corrected for trigger events
            epochs._data = epochs.get_data() - line_base
            fn_epo = fn_raw[:fn_raw.rfind('-raw.fif')] + ',evt_%s_bc-epo.fif' %conditions[i]
            epochs.save(fn_epo)
            del epochs
            # Take the response events as the reference events   
            events_, lag = define_target_events(levents, res_id, tri_id, 
                                                sfreq, rstmin, rstmax)
            epochs = mne.Epochs(raw, events_, res_id, rtmin,
                                rtmax, picks=picks, baseline=None,
                                reject=dict(mag=4e-12))
            # Baseline corrected for response events
            #epochs.load_data()
            equalize_epoch_counts([epo_base, epochs])
            line_base = epo_base.get_data().mean(axis=-1, keepdims=True)
            epochs._data = epochs.get_data() - line_base
            res_con = conditions[i][:2] + 'rt'
            fn_epo = fn_raw[:fn_raw.rfind('-raw.fif')] + ',evt_%s_bc-epo.fif' %res_con
            epochs.save(fn_epo)
            del epochs
            i = i + 1

###################################
# Bad epochs visual inspection
#----------------------------------
''' Check the bad epochs of each subject
'''
#fn_epos = glob.glob(subjects_dir + '/203867/MEG/*,evt_*_bc-epo.fif')
#fn_epo = fn_epos[0]
##fn_ave = fn_epo[:fn_epo.rfind('-epo.fif')] + '-ave.fif' 
#epo = mne.read_epochs(fn_epo, preload=True)
#epo.plot(block=True, n_epochs=40)
#epo.save(fn_epo)

###################################
# Average filtered epos
#----------------------------------

if avg_fil_epos:
    fn_epos = glob.glob(subjects_dir + '/*[0-9]/MEG/*,evt_*_bc-epo.fif')
    for fn_epo in fn_epos:
        fn_ave = fn_epo[:fn_epo.rfind('-epo.fif')] + '-ave.fif'
        epo = mne.read_epochs(fn_epo)
        evo = epo.average()
        evo.save(fn_ave)


