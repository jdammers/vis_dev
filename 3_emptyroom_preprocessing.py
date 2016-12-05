''' Empty room data is processed by noise_reducer (and filtering),
    Then bad channels will be view inspected and interpolated.
'''
from apply_empty import apply_empty, apply_cov
import os, glob, shutil
subjects_dir = '/home/uais/data/Chrono/18subjects/CAU/'
# Copy empty raw data into the individual path
#fn_list = glob.glob('/home/qdong/data/Chrono/18subjects/Empty_room/*')
#for fn_raw in fn_list:
#    name = os.path.basename(fn_raw)
#    subject = name.split('_')[0]
#    dist_path = subjects_dir + '/%s/MEG/' %subject
#    shutil.copy(fn_raw, dist_path)

#####################################################
# Apply noise reducer on the empty room data
#------------------------------------------
#fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*[0-9]*-empty.fif')
#apply_empty(fn_list)

#####################################################
# Bad channels will be interpolated
#------------------------------------------
#import mne
# Prepare interpolated emptyroom data for causality analysis 
#fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*[0-9]*nr-empty.fif')
# Prepare interpolated emptyroom data for ROIs definition
#fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*[0-9]*45-empty.fif')
#fn_raw = fn_list[-1]
#raw = mne.io.Raw(fn_raw, preload=True)
#raw = mne.io.Raw(fn_raw)
#raw.plot(duration=200)
#print raw.info['bads']
#raw.interpolate_bads(reset_bads=True)
#print raw.info['bads']
#raw.save(fn_raw, overwrite=True)
#####################################################
# Noise covariance calculation
#------------------------------------------
# Calculate noise_cov for unfiltered empty_room data
fn_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*[0-9]*nr-empty.fif')
apply_cov(fn_list, filtered = False)
# Calculate noise_cov for filtered empty_room data
fn_fil_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*[0-9]*45-empty.fif')
apply_cov(fn_fil_list)
