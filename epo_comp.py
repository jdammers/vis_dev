import os, glob
from jumeg.jumeg_plot import plot_compare_brain_responses
subjects_dir = os.environ['SUBJECTS_DIR']
fn_path = subjects_dir + '109077/MEG/'
fnraw = glob.glob(fn_path +'109077_Chrono01_110518_1415_1_c,rfDC,nr,ocarta,fibp1-45-raw.fif')[0]
fnorig = glob.glob(fn_path +'109077_Chrono01_110518_1415_1_c,rfDC-raw.fif')[0]
plot_compare_brain_responses(fnorig, fnraw, stim_name='trigger')

plot_compare_brain_responses(fnorig, fnraw, stim_name='response', event_id=8)