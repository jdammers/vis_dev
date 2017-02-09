import os
import numpy as np
from stat_cluster import ara_trivsres
from scipy import stats as stats
from mne import spatial_tris_connectivity, grade_to_tris
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
f_threshold = 0.01
st_max, st_min = 0.3, 0. # time period of stimulus
res_max, res_min = 0.1, -0.2 # time period of response
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
st_list = ['LLst', 'RRst', 'RLst',  'LRst'] # stimulus events
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt']
do_arange = True
if do_arange:
    tstep, X = ara_trivsres(st_list, res_list, st_min, st_max, res_min, res_max, stcs_path, subjects_dir)
else:
    res_mat = np.load(stcs_path + 'res.npz')
    tri_mat = np.load(stcs_path + 'tri.npz')
    X = [tri_mat['tri'], res_mat['res']]
    tstep = tri_mat['tstep']
    
fsave_vertices = [np.arange(10242), np.arange(10242)]
connectivity = spatial_tris_connectivity(grade_to_tris(5))
T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_test(X, n_permutations=8192/2, #step_down_p=0.001,
                                     connectivity=connectivity, n_jobs=1,
                                     # threshold=t_threshold, stat_fun=stats.ttest_ind)
                                     threshold=f_threshold)
np.savez(stcs_path+'trivsres.npz', clu=clu, tstep=tstep, fsave_vertices=fsave_vertices)