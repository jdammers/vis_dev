from jumeg.jumeg_plot import plot_artefact_overview
import glob,os
from dirs_manage import set_directory
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
per_path = subjects_dir + 'performance/'
set_directory(per_path) 
fn_raw_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*rfDC-raw.fif')
for fn_raw in fn_raw_list:
    fn_raw_clean = fn_raw[:fn_raw.rfind('-raw.fif')] + ',nr,ocarta-raw.fif'
    subject = os.path.basename(fn_raw).split('_')[0]
    fn_out = per_path + subject
    plot_artefact_overview(fn_raw, fn_raw_clean, overview_fname=fn_out, 
                       ecg_ch='ECG 001', eog1_ch='EOG 001', eog2_ch='EOG 002')