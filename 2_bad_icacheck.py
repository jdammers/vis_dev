''' This script confirms ECG and EOG are cleared by visual inspection, 
    if this step is done, you can make further processing.
'''
import mne
import numpy as np
from mne.preprocessing import ICA, read_ica
from jumeg.jumeg_preprocessing import get_ics_cardiac, get_ics_ocular
from jumeg.jumeg_plot import plot_performance_artifact_rejection


subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
subject = '201195'
path_data = subjects_dir + '%s/MEG/' %subject

# -------------------------------------------------------
# read continuous unfiltered raw data and apply filter
# -------------------------------------------------------
import glob
fn_raw = glob.glob(path_data + '*rfDC,nr-raw.fif')[0]
fn_raw_oca = fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta-raw.fif'
#fn_raw = path_data + '109077_Chrono01_110518_1415_1_c,rfDC,nr,ocarta-raw.fif'
raw_orig = mne.io.read_raw_fif(fn_raw, preload=True)
raw_orig.plot(start=0,duration=120)# Bad channels inspection


raw = raw_orig.copy()  # kepp a copy of the original data 
# Note, bad channels should be exclude -  which I did NOT do here 
picks_meg = mne.pick_types(raw.info, meg=True, exclude='bads')
#raw.plot(start=0,duration=120, n_channels=40)

# apply filter before ICA decomposition
raw.filter(1, 25, n_jobs=2, method='fft')
# -------------------------------------------------------
# apply ICA decomposition
# -------------------------------------------------------
#ica_method = 'fastica'
ica_method = 'extended-infomax'
decim = 3
ica = ICA(method=ica_method, n_components=0.99)
ica.fit(raw, picks=picks_meg, decim=decim)
# -------------------------------------------------------
# before visual inspection of ICA components
# try the automatic version to see what we get
# -------------------------------------------------------
# search for ECG artifact components (should already be removed by OCARTA)
idx_ecg = get_ics_cardiac(raw, ica, flow=10, fhigh=20, tmin=-0.3, tmax=0.3,
        name_ecg='ECG 001', use_CTPS=True, thresh=0.3)
# search for EOG artifact components
idx_eog1 = get_ics_ocular(raw, ica, flow=1, fhigh=10,
        name_eog_hor='EOG 001',score_func='pearsonr', thresh=0.25)
idx_eog2 = get_ics_ocular(raw, ica, flow=1, fhigh=10,
        name_eog_ver='EOG 002',score_func='pearsonr', thresh=0.25)

# check what we have in idx_ecg and idx_eog
# -------------------------------------------------------
# now do visual inspection of ICA components
# -------------------------------------------------------
sources = ica.get_sources(raw, add_channels=['EOG 002','ECG 001','EOG 001'])
sources.plot(n_channels=25, duration=120,scalings=dict(misc=3, eog=4e-4, ecg=1e-3))


''' The noisy indices should minus 1 before type them into the excluding list.
'''
ica.exclude = list(np.unique(list(idx_ecg) + list(idx_eog1) + list(idx_eog2) + list([])))


# -------------------------------------------------------
# apply ICA cleaning on original unfiltered data !!
# -------------------------------------------------------
raw_new = ica.apply(raw_orig.copy(), exclude=ica.exclude, n_pca_components=0.99)
# check your new raw cleaned data, and reject bad channels by visual inspection
#raw_new.plot(start=0,duration=120)
# -------------------------------------------------------
# save ICA object and plot ICA rejection performance
# -------------------------------------------------------
fn_ica = fn_raw[:-8]+',ica.fif'
fn_perf = fn_ica[:-4]+',perf'
ica.save(fn_ica)
# Note, there is a problem with the MNE peak detection of EOG peaks
# therefore the we only find a few peaks in the EOG signal
#raw_orig = mne.io.read_raw_fif(fn_raw, preload=True)
#plot_performance_artifact_rejection(raw_orig, ica, fn_perf, meg_clean=raw_new, show=True)
plot_performance_artifact_rejection(raw_orig, ica, fn_perf, show=True)

# -------------------------------------------------------
# save results
# -------------------------------------------------------
# maybe you keep a copy of the original OCARTA raw fileimport shutil
#import shutil
#shutil.copy2(fn_raw,fn_raw+'.orig' )

# save/overwrite cleaned data 
raw_new.save(fn_raw_oca, overwrite=True)


# now create epochs and check for bad epochs