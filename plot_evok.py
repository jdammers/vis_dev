'''plot evoked data for each condition for each subject.
'''
import os, glob
from mne import read_evokeds
from apply_evo import plot_evo, plot_evo_allchs
from dirs_manage import set_directory
subjects_dir = os.environ['SUBJECTS_DIR']
fig_path = subjects_dir+ '/evo_plots/'
#set_directory(fig_path) 

st_list = ['LLst', 'RRst', 'RLst',  'LRst']
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt']
conf_per = True
conf_res = False
st_max = 0.4
st_min = -0.2
res_min = -0.3
res_max = 0.3 

if conf_per == True:
    evt_list = st_list
    tmin, tmax = st_min, st_max
    conf_type = 'conf_per' 
    baseline = True
    
#conflicts response
elif conf_res == True:
    evt_list = res_list
    tmin, tmax = res_min, res_max
    conf_type = 'conf_res'
    baseline = False
    
 
indi_list = glob.glob(subjects_dir+'[0-9]*[0-9]')

for indi_path in indi_list:
    subject = indi_path.split('subjects/')[1]
    evokeds = []
    for evt in evt_list:
        [fn_evt] = glob.glob(indi_path + '/MEG/*fibp1-45,evt_%s_bc-ave.fif' %evt)
        [evo] = read_evokeds(fn_evt)
        evo.crop(tmin, tmax)
        evokeds.append(evo)
    fn_fig1 = fig_path + '%s_%s.tif' %(subject, conf_type)
    fn_fig2 = fig_path + '%s_%s_allchs.tif' %(subject, conf_type)
    plot_evo(evokeds, evt_list, fn_fig1)
    plot_evo_allchs(evokeds, evt_list, fn_fig2)
    del evokeds
    #import pdb
    #pdb.set_trace()
