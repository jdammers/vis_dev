from jumeg.jumeg_plot import plot_artefact_overview
import glob,os
from mne.io import read_raw_fif
from dirs_manage import set_directory
import matplotlib.pyplot as plt

subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
per_path = subjects_dir + 'psd/'
set_directory(per_path) 
fn_raw_list = glob.glob(subjects_dir + '/*[0-9]/MEG/*,nr,ocarta-raw.fif')
for fn_raw in fn_raw_list[5:]:
    subject = os.path.basename(fn_raw).split('_')[0]
    fn_path = os.path.dirname(fn_raw)
    fn_raw_empty = fn_path + '/%s_empty,nr-raw.fif' %subject
    raw = read_raw_fif(fn_raw)
    raw_empty = read_raw_fif(fn_raw_empty)
    fig1 = raw.plot_psd(fmin=1, fmax=100)
    fig2 = raw_empty.plot_psd(fmin=1, fmax=100)
    fn_out1 = per_path + subject
    fn_out2 = per_path + subject + '_empty'
    fig1.savefig(fn_out1, dpi=300)
    fig2.savefig(fn_out2, dpi=300)
    plt.close('all')