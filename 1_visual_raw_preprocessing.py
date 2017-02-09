'''1. preprocessing pipeline:
      remove electrical noise(50Hz) and noise below 5Hz
      
'''
import os.path as op
#from jumeg.jumeg_preprocessing import apply_filter
from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising
#from jumeg.decompose import ocarta
#from jumeg import jumeg_plot
import glob, mne, os
import matplotlib.pyplot as plt
import shutil
from mne.preprocessing import ICA
#from mne.event import define_target_events
import numpy as np
#from mne.epochs import equalize_epoch_counts
from mne.event import define_target_events
from dirs_manage import reset_directory, set_directory
from jumeg.jumeg_preprocessing import (get_ics_cardiac,
                                       get_ics_ocular)
from jumeg.jumeg_plot import plot_artefact_overview
###########################
# Set your data path here
#--------------------------
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
MIN_path = subjects_dir + 'fsaverage'

###########################
# Set the parameters
#--------------------------
do_clip = False #This step relys on visual inspection
do_pre = False # data preprocessing
do_rej_HE = False # Remove noise related with heart beats and eye movements.
do_epo = False
do_unfil_epo = False
combine_epochs = False
check_epochs = True
        
ave_epo = False
conditions = ['LLst', 'LRst', 'RRst', 'RLst']
#conditions = ['LLst', 'LRst', 'RRst', 'RLst','LLrt', 'LRrt', 'RRrt', 'RLrt']
res_name, tri_name = 'STI 013', 'STI 014'
ev_id = 1
ecg_ch = 'ECG 001'
eog1_ch = 'EOG 001'
eog2_ch = 'E0G 002'
##################################################################
# Bad channels and noisy EOG events clipping by visual inspection
#----------------------------------------------------------------
if do_clip:
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*rfDC-raw.fif')
    fn_raw = fn_list[1]#Here indexing each subject for visual inspection
    shutil.copy2(fn_raw,fn_raw+'.orig' )
    raw = mne.io.Raw(fn_raw, preload=True)
    #Check bad channels
    raw.plot(start=0,duration=120)
    #Interpolate bad channels
    inter_bad = True
    if inter_bad == True:
        print('bad channels are:', raw.info['bads'])
        raw.interpolate_bads(reset_bads=True)
    #cut_badsegments = False
    #if cut_badsegments:
    #    raw.crop(tmin=15, tmax=raw.times.max()-5)
    
    #Save corrected raw data
    raw.save(fn_raw, overwrite=True)
    
    #Check noisy EOG channels.
    eog2, _ = raw[276, :]
    plt.plot(eog2[0], 'b', label='original signal')#Confirm the clipped threshold
    # Clip the EOG data using the clipped threshold
    clip_thre = -0.00005
    if clip_thre == None:#There are no apparent peaky EOG_2 events.
        eog2_clipped = eog2
        
    #Here the logic has some problems!!!!!!
    
    elif clip_thre < 0:
        eog2_clipped = np.clip(eog2, clip_thre, np.max(eog2))
    elif clip_thre > 0:
        eog2_clipped = np.clip(eog2, np.min(eog2), clip_thre)
    #eog2_clipped = np.clip(eog2, -0.00036, np.max(eog2))
    #eog2_clipped = np.clip(eog2_clipped, np.min(eog2_clipped), 0.00001)
    plt.plot(eog2_clipped[0], 'r', label='Clipped signal')
    plt.legend()
    
    
    eog2_correct_events = mne.preprocessing.eog._find_eog_events(eog2_clipped, 998, l_freq=1, h_freq=10,
                                                                sampling_rate=raw.info['sfreq'],
                                                                first_samp=raw.first_samp,
                                                                filter_length='10s', tstart=0.0)
    #write_events = True
    #if write_events:
    fn_eve = fn_raw[:fn_raw.rfind('-raw.fif')] + '_eog2-eve.fif'
    mne.write_events(fn_eve, eog2_correct_events)

##########################################################
# Noise_reducer for noise around 50Hz, 60Hz and below 5Hz
#---------------------------------------------------------
if do_pre:
    fn_list = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc-raw.fif')
    perf_path = subjects_dir + 'rn_perf/'
    set_directory(perf_path)
    for fn_raw in fn_list:
        #import noise_reducer and plot_power_spectrum function
        #apply noise reducer for 50 Hz (and harmonics)
        fn_raw_nr = fn_raw[:fn_raw.rfind('-raw.fif')] + ',nr-raw.fif'
        subject = os.path.basename(fn_raw_nr).split('_')[0]
        fn_per = perf_path + subject
        noise_reducer(fn_raw, refnotch=50, detrending=False, fnout=fn_raw_nr) 
        noise_reducer(fn_raw_nr, refnotch=60, detrending=False, fnout=fn_raw_nr)
        #apply noise reducer for frequencies below 5 Hz
        noise_reducer(fn_raw_nr, reflp=5, fnout=fn_raw_nr)# plot power spectrum
        fn_power_spect = fn_raw_nr[:fn_raw_nr.rfind('-raw.fif')] + ',denoising'
        plot_denoising([fn_raw, fn_raw_nr], stim_name=tri_name,
                    event_id=ev_id, show=False, fnout=fn_per,
                    tmin_stim=-0.4, tmax_stim=0.4)
    
##########################################################
# Remove ECG and EOG artifacts
#---------------------------------------------------------    

if do_rej_HE:   
    #from apply_chopICA import chop_and_apply_ica
    ecg_ch = 'ECG 001'
    eog1_ch = 'EOG 001'
    eog2_ch = 'EOG 002'    
    # use 8Hz to account for slower T waves (ecg)
    flow_ecg, fhigh_ecg = 8., 20.
    flow_eog, fhigh_eog = 1., 20.
    ecg_thresh, eog_thresh = 0.25, 0.2
    apply_on_unfiltered = True
    flow,fhigh=1., 45.
    chop_length=60.
    automatic_artefact_check=False
    manual_check=False
    save=True
    fn_list = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc,nr-raw.fif')
    perf_path = subjects_dir + 'HE_perf/'
    set_directory(perf_path)
    for fn_nr in fn_list:
        subject = os.path.basename(fn_nr).split('_')[0]
        overview_fname = perf_path + subject
        raw = mne.io.Raw(fn_nr, preload=True, verbose=True)
        picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    
        # filter the data (better to filter before chopping)
        raw_fil = raw.copy().filter(flow, fhigh, method='iir', n_jobs=2,
                                    iir_params={'ftype': 'butter', 'order': 4})
    
        # chop the data and apply filtering
        duration = (raw.n_times - 1) / raw.info['sfreq']
        to_end = False
        for tstep in np.arange(0, duration, chop_length):
            tmin = tstep
            tmax = tstep + chop_length
            if tstep + 2 * chop_length > duration:
                tmax = None
            print tmin, tmax
            # make sure to copy because the original is lost
            # run the ICA on the chops
    
            # building the file names here
            info_filt = "fibp%d-%d" % (flow, fhigh)
            if tmax is not None:
                ica_fname = fn_nr.rsplit('-raw.fif')[0] + ',{}-{}-ica.fif'.format(int(tmin), int(tmax))
                clean_fname = fn_nr.rsplit('-raw.fif')[0] +\
                    ',ar,{}-{}-raw.fif'.format(int(tmin), int(tmax))
                filt_clean_fname = fn_nr.rsplit(',bcc,nr-raw.fif')[0] +\
                    ',bcc,nr,{},ar,{}-{}-raw.fif'.format(info_filt, int(tmin), int(tmax))
            else:
                ica_fname = fn_nr.rsplit('-raw.fif')[0] + ',{}-{}-ica.fif'.format(int(tmin), tmax)
                clean_fname = fn_nr.rsplit('-raw.fif')[0] +\
                    ',ar,{}-{}-raw.fif'.format(int(tmin), tmax)
                filt_clean_fname = fn_nr.rsplit(',bcc,nr-raw.fif')[0] +\
                    ',bcc,nr,{},ar,{}-{}-raw.fif'.format(info_filt, int(tmin), tmax)
                    
            print 'Starting ICA...'
            if op.isfile(ica_fname):
                ica = mne.preprocessing.read_ica(ica_fname)
            else:
                raw_filt_chop = raw_fil.copy().crop(tmin=tmin, tmax=tmax)
                ica = ICA(method='fastica', n_components=60, random_state=None,
                        max_pca_components=None, max_iter=1500, verbose=False)
                ica.fit(raw_filt_chop, picks=picks, decim=None, reject={'mag': 5e-12},
                        verbose=True)
                # get ECG and EOG related components using MNE
                print 'Computing scores and identifying components..'
                ecg_scores = ica.score_sources(raw_filt_chop, target=ecg_ch, score_func='pearsonr',
                                            l_freq=flow_ecg, h_freq=fhigh_ecg, verbose=False)
                # horizontal channel
                eog1_scores = ica.score_sources(raw_filt_chop, target=eog1_ch, score_func='pearsonr',
                                                l_freq=flow_eog, h_freq=fhigh_eog, verbose=False)
                # vertical channel
                eog2_scores = ica.score_sources(raw_filt_chop, target=eog2_ch, score_func='pearsonr',
                                                l_freq=flow_eog, h_freq=fhigh_eog, verbose=False)
    
                # print the top ecg, eog correlation scores
                ecg_inds = np.where(np.abs(ecg_scores) > ecg_thresh)[0]
                eog1_inds = np.where(np.abs(eog1_scores) > eog_thresh)[0]
                eog2_inds = np.where(np.abs(eog2_scores) > eog_thresh)[0]
                highly_corr = list(set(np.concatenate((ecg_inds, eog1_inds, eog2_inds))))
                print 'Highly correlated artefact components are', highly_corr
    
                # get ECG/EOG related components using JUMEG
                ic_ecg = get_ics_cardiac(raw_filt_chop, ica,
                                        flow=flow_ecg, fhigh=fhigh_ecg,
                                        thresh=ecg_thresh,
                                        tmin=-0.5, tmax=0.5, name_ecg=ecg_ch,
                                        score_func='pearsonr',
                                        use_CTPS=True)
                ic_eog = get_ics_ocular(raw_filt_chop, ica,
                                        flow=flow_eog, fhigh=fhigh_eog, thresh=eog_thresh,
                                        name_eog_hor=eog1_ch, name_eog_ver=eog2_ch,
                                        score_func='pearsonr')
                print 'Identified ECG components are: ', ic_ecg
                print 'Identified EOG components are: ', ic_eog
    
                # if necessary include components identified by correlation as well
                # ica.exclude += list(ic_ecg) + list(ic_eog) + highly_corr
    
                # add them all to exclude list
                ica.exclude += list(ic_ecg) + list(ic_eog)
                ica.exclude = list(set(ica.exclude))  # to sort and remove repeats
    
                # do the most important manual check for removing bad ICs
                if manual_check:
                    ica.plot_sources(raw_filt_chop, block=True)
                    #ica.exclude = list(np.unique(list(ica.exclude) + list([1,6])))
                print 'ICA components excluded: ', ica.exclude
        
                # save ica object (fitted to filtered data)
                ica.save(ica_fname)
    
            # apply the ICA on data and save them
            new_overview_fname = overview_fname + ',%d' %(tmin)
            if apply_on_unfiltered:
                unfilt_raw_chop = raw.copy().crop(tmin=tmin, tmax=tmax)
                print 'Running cleaning on unfiltered data...'
                raw_chop_clean = ica.apply(unfilt_raw_chop.copy(), exclude=ica.exclude,
                                        n_pca_components=None)
                if save:
                    raw_chop_clean.save(clean_fname, overwrite=False)
                #overview_fname = clean_fname.rsplit('-raw.fif')[0] + ',overview-plot.png'
                plot_artefact_overview(unfilt_raw_chop, raw_chop_clean,
                                    overview_fname=new_overview_fname, ecg_ch=ecg_ch,
                                    eog1_ch=eog1_ch, eog2_ch=eog2_ch)
                print 'Saved ', overview_fname
            else:
                print 'Running cleaning on filtered data...'
                raw_chop_clean = ica.apply(raw_filt_chop.copy(), exclude=ica.exclude,
                                        n_pca_components=None)
                if save:
                    raw_chop_clean.save(filt_clean_fname, overwrite=True)
                #filt_overview_fname = filt_clean_fname.rsplit('-raw.fif')[0] + ',overview-plot.png'
                plot_artefact_overview(raw_filt_chop, raw_chop_clean,
                                    overview_fname=new_overview_fname,  ecg_ch=ecg_ch,
                                    eog1_ch=eog1_ch, eog2_ch=eog2_ch)
                print 'Saved ', overview_fname
            
            if tmax == None:
                break 

###################################
# Crop the filtered data
#----------------------------------

from mne.epochs import equalize_epoch_counts
if do_epo:
    conds_id = [(1, 8), (2, 64), (3, 64), (4, 8)]
    # Define the raw_data path for epoching
    fn_list = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc,nr,fibp1-45,ar,*-raw.fif') 
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
            if len(events_) != 1:
                equalize_epoch_counts([epo_base, epochs])
            line_base = epo_base.get_data().mean(axis=-1, keepdims=True)
            epochs._data = epochs.get_data() - line_base
            res_con = conditions[i][:2] + 'rt'
            fn_epo = fn_raw[:fn_raw.rfind('-raw.fif')] + ',evt_%s_bc-epo.fif' %res_con
            epochs.save(fn_epo)
            del epochs
            i = i + 1
            
###################################
# Crop the unfiltered data
#----------------------------------
if do_unfil_epo:
    #compute_epoch_per_chop = True  # compute and save epoch per chop per subj
    #combine_chopped_epochs = False  # combine the epoch chops
    #manual_check_epochs = False  # plot epochs and manually makr bad segments
    conds_id = [(1, 8), (2, 64), (3, 64), (4, 8)]
    conditions = ['LLst', 'LRst', 'RRst', 'RLst']
    baseline = (None, 0)
    reject = {'mag': 4e-12}
    ltmin, ltmax = 0, 0.8  # the time range for fitting visual events
    tmin, tmax = -0.2, 0.8  # the time range for epochs
    
    #if compute_epoch_per_chop:
        # provide the correct regex to find chops here !! (crucial)
    fn_list = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc,nr,ar,*-raw.fif')  # unfiltered
    # loop across chops per subject (done for better ICA)
    for fn_raw in fn_list:
        print 'processing file ', fn_raw
        raw = mne.io.Raw(fn_raw, preload=True)
        tri_events = mne.find_events(raw, stim_channel='STI 014')
        res_events = mne.find_events(raw, stim_channel='STI 013')
        levents = np.array(list(tri_events) + list(res_events))
        # based on the correct event pairs and a fixed time range
        # identify the correct epochs
        # loop across conditions
        for i, cond_id in enumerate(conds_id):
            fn_epo = fn_raw.rsplit('-raw.fif')[0] + ',evt_%s_bc-epo.fif' % conditions[i]
            if not op.isfile(fn_epo):
                print 'running for condition ', conditions[i]
                tri_id, res_id = cond_id[0], cond_id[1]
                events, lag = define_target_events(levents, tri_id, res_id,
                                                    raw.info['sfreq'], ltmin, ltmax)
                print 'Stim events before dropping ', tri_events[tri_events[:, 2] == tri_id].shape
                print 'Stim events after dropping ', events.shape
                print 'Lag ', lag.shape
                picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False,
                                        eog=False, exclude='bads')
                epochs = mne.Epochs(raw, events, tri_id, tmin,
                                    tmax, picks=picks, baseline=(None, 0),
                                    reject=dict(mag=4e-12), preload=True,
                                    reject_by_annotation=True)

                print 'Length of epochs ', len(epochs)
                if len(epochs) != 0:
                    epochs.save(fn_epo)
                else:
                    print 'INFO: No epochs for ', fn_epo
                    
###############################################
# Integrate chops and epochs visual inspection
#----------------------------------------------
filtered = False
if combine_epochs:
    subjects = ['203731', '201195', '203709', '203792', '203969', '203822', 
             '203929', '203867', '203147', '203267', '203840', '203780', '203288']
    for subject in subjects:
        for mycond in conditions:
            print 'running for condition ', mycond
            if filtered:
                fn_list = glob.glob(op.join(subjects_dir, subject, 'MEG/*bcc,nr,fibp1-45,ar,*,evt_%s_bc-epo.fif' % mycond))
            else:
                fn_list = glob.glob(op.join(subjects_dir, subject, 'MEG/*bcc,nr,ar,*,evt_%s_bc-epo.fif' % mycond))
            print len(fn_list), ' files for this condition.'
            # loop across chops per condition per subject (done for better ICA)
            chop_epochs_list = []
            for fn_epo in fn_list:
                epochs = mne.read_epochs(fn_epo, preload=True)
                chop_epochs_list.append(epochs)
                os.remove(fn_epo)
            final_epochs = mne.concatenate_epochs(chop_epochs_list)
            final_epochs_name = fn_epo.rsplit('ar')[0] + 'ar,evt_%s_bc-epo.fif' % mycond  # jumeg epocher
            print 'saving combined epochs as ', final_epochs_name
            final_epochs.save(final_epochs_name) 


if check_epochs:
    # for manual checking and identifying bad epochs
    if filtered:
        fn_list = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc,nr,fibp1-45,ar,evt_*bc-epo.fif') 
    else:
        fn_list = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc,nr,ar,evt_*bc-epo.fif')
    for fn_epo in fn_list:
        print 'processing file ', fn_epo
        epochs = mne.read_epochs(fn_epo, preload=True)
        epochs.plot(block=True, n_epochs=40)
        epochs.save(fn_epo) 

if ave_epo:
    fn_epos = glob.glob(subjects_dir + '*[0-9]/MEG/*rfDC,bcc,nr,fibp1-45,ar,evt_*bc-epo.fif') 
    for fn_epo in fn_epos:
        fn_ave = fn_epo[:fn_epo.rfind('-epo.fif')] + '-ave.fif'
        epo = mne.read_epochs(fn_epo)
        evo = epo.average()
        evo.save(fn_ave) 
