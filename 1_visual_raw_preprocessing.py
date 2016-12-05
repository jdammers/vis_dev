'''1. preprocessing pipeline:
      remove electrical noise(50Hz) and noise below 5Hz
      
'''
from jumeg.jumeg_preprocessing import apply_filter
from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising
from jumeg.decompose import ocarta
from jumeg import jumeg_plot
import os, glob

###########################
# Set your data path here
#--------------------------
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
MIN_path = subjects_dir + 'fsaverage'

###########################
# Set the parameters
#--------------------------
do_pre = False # data preprocessing
do_fil = True # data filtering
do_epo = True # Crop data into epochs
res_name, tri_name = 'STI 013', 'STI 014'

###################################
# Raw data preprocessing
#----------------------------------


if do_pre:
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*rfDC-raw.fif')
    #import pdb
    #pdb.set_trace()
    ##for i in [0, 2, 7, 10]:
    for fn_raw in fn_list:
        #import noise_reducer and plot_power_spectrum function
        #apply noise reducer for 50 Hz (and harmonics)
        fn_raw_nr = fn_raw[:fn_raw.rfind('-raw.fif')] + ',nr-raw.fif'
        noise_reducer(fn_raw, refnotch=50, detrending=False, fnout=fn_raw_nr) 
        #noise_reducer(fn_raw_nr, refnotch=60, detrending=False, fnout=fn_raw_nr)
        #apply noise reducer for frequencies below 5 Hz
        noise_reducer(fn_raw_nr, reflp=5, fnout=fn_raw_nr)# plot power spectrum
        fn_power_spect = fn_raw_nr[:fn_raw_nr.rfind('-raw.fif')] + ',denoising'
        ev_id = 1
        plot_denoising([fn_raw, fn_raw_nr], stim_name=tri_name,
                    event_id=ev_id, show=False, fnout=fn_power_spect,
                    tmin_stim=-0.4, tmax_stim=0.4)
        # import ocarta module
    
        # apply OCARTA
        ocarta_obj = ocarta.JuMEG_ocarta()
        #fn_ocarta = fn_raw_nr[:fn_raw_nr.rfind('-raw.fif')] + ',ocarta_perf'
        ocarta_obj.fit(fn_raw_nr, flow=1, fhigh=20)
    
    
    
''' Before data filtering, please make sure ECG and EOG cleared, and refer the script
    '2_bad_icacheck.py'
'''

###################################
# Filter the data
#----------------------------------

if do_fil:
    fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*ocarta-raw.fif')
    for fn_raw_cl in fn_list:
        fn_raw_fil = fn_raw_cl[:fn_raw_cl.rfind('-raw.fif')] + ',fibp1-45-raw.fif'
        apply_filter(fn_raw_cl, flow=1, fhigh=45, order=4, njobs=4)
        jumeg_plot.plot_compare_brain_responses(fn_raw, fn_raw_fil, event_id=ev_id, stim_name='trigger')
        
###################################
# Crop the data
#----------------------------------
