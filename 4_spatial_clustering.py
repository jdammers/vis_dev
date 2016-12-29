import glob, sys
from stat_cluster import set_directory, per2test
#####################################################
# Aplly spatio-temporal clustering for ROI defintion
#####################################################
''' From here data will be calculated in the source
    space.
'''
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'
MIN_path = subjects_dir + 'fsaverage'
do_make_inve = False # Generate inverse operator
do_inver_ave = False # make STC
do_morph_STC = False # Morph individual STC
do_group_STC = False # Group morphed STC into pre and post-stimuls (events)
do_t_test = True # Spatial clustering
do_2sample = True #2 sample test
do_clu2STC = True # Transfer cluster arrays into STC objects.

# Set the option for stimulus or response

conf_type = sys.argv[1]

print '>>>>processing significant clusters related with %s' %conf_type
stcs_path = subjects_dir + 'fsaverage/conf_stc/'
n_subjects = 13 # The amount of subjects
st_list = ['LLst', 'RRst', 'RLst',  'LRst'] # stimulus events
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt'] # response events
st_max, st_min = 0.3, 0. # time period of stimulus
res_max, res_min = 0.1, -0.2 # time period of response
set_directory(stcs_path)
# parameter of Morphing
grade = 5
method = 'dSPM'
snr = 3
template = 'fsaverage' # The common brain space

# Parameters of Moving average
mv_window = 10 # miliseconds
overlap = 0 # miliseconds
nfreqs = 678.17

# The parameters for clusterring test
permutation = 8192
p_th = 0.0001 # spatial p-value
p_v = 0.001 # comparisons corrected p-value



#stimulus
if conf_type == 'tri':
#if conf_per == 'True':
    evt_list = st_list
    tmin, tmax = st_min, st_max
    baseline = True

#response
elif conf_type == 'res':
#elif conf_res == 'True':
    evt_list = res_list
    tmin, tmax = res_min, res_max
    baseline = False


############################################
# make inverse operator of filtered evoked 
#------------------------------------------
if do_make_inve:
    from stat_cluster import apply_inverse_ave
    print '>>> Calculate inverse solution ....'
    fn_evt_list = glob.glob(subjects_dir+'*[0-9]/MEG/*bcc,nr,fibp1-45,ar,evt_LLst_bc-ave.fif')
    apply_inverse_ave(fn_evt_list, subjects_dir)
    print '>>> FINISHED with inverse solution.'
    print ''       
        

############################################
# inverse evo to the source space 
#------------------------------------------

if do_inver_ave:
    print '>>> Calculate STC ....'
    from stat_cluster import apply_STC_ave
    fn_evt_list = glob.glob(subjects_dir+'*[0-9]/MEG/*bcc,nr,fibp1-45,ar,evt_*_bc-ave.fif')
    apply_STC_ave(fn_evt_list, method=method, snr=snr)
    print '>>> FINISHED with STC generation.'
    print ''
    
    
###################################################
# Morph individual STC into the common brain space 
#--------------------------------------------------
if do_morph_STC:
    print '>>> Calculate morphed STC ....'
    from stat_cluster import morph_STC
    for evt in evt_list:
        #fn_stc_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-lh.stc' %evt)
        fn_stc_list = glob.glob(subjects_dir+'*[0-9]/MEG/*bcc,nr,fibp1-45,ar,evt_%s_bc-lh.stc' %evt)
        morph_STC(fn_stc_list, grade, subjects_dir, template, event=evt, baseline=baseline)
    print '>>> FINISHED with morphed STC generation.'
    print ''

###################################################
# Group STCs into pre- and post-stimulus (events)
#--------------------------------------------------
if do_group_STC:
    print '>>> Calculate Matrix for contrasts ....'
    from stat_cluster import Ara_contr_base
    Ara_contr_base(evt_list, tmin, tmax, stcs_path, n_subjects=n_subjects,
                   template='fsaverage', subjects_dir=subjects_dir)
    print '>>> FINISHED with a group matrix generation.'
    print ''

##################################################################
# Spatial clustering for significant clusters related with events
#----------------------------------------------------------------
if do_t_test:
    print '>>> ttest for clustering ....'
    from stat_cluster import clu2STC, stat_clus
    for evt in evt_list:
        evt = '1sample_%s' %(evt)
        fnmat = stcs_path + evt + '.npz'
        #conf_mark = 'ttest_' + conf_type
        print '>>> load Matrix for contrasts ....'
        import numpy as np
        npz = np.load(fnmat)
        tstep = npz['tstep'].flatten()[0]
        X = npz['X']
        print '>>> FINISHED with the group matrix loaded.'
        print ''
        fn_clu_out = stcs_path + '%d_pthr%.4f_%s.npz' %(permutation, p_th, evt)
        stat_clus(X, tstep, n_per=permutation, p_threshold=p_th, p=p_v,
                fn_clu_out=fn_clu_out)
        del X
    print '>>> FINISHED with the clusters generation.'
    print ''

###############################################################################
# Clustering using 2sample f-test
# -----------------
if do_2sample:
    ''' This comparison is suitable for the samples from different entireties
    '''
    print '>>> 2smpletest for clustering ....'
    for evt in evt_list:
        #evt = '2sample_%s_%s' %(conf_type, evt)
        evt = '2sample_%s' %(evt)
        print '>>> load Matrix for contrasts ....'
        import numpy as np
        fnmat = stcs_path + evt + '.npz'
        npz = np.load(fnmat)
        tstep = npz['tstep'].flatten()[0]
        X = npz['X']
        fn_clu_out = stcs_path + '%d_pthr%.4f_%s.npz' % (permutation/2, p_th, evt)
        per2test(X, p_thr=p_th, p=p_v, tstep=tstep, n_per=permutation/2,
                fn_clu_out=fn_clu_out)
        del X
        clu2STC(fn_clu_out, p_thre=p_v, tstep=0.01)
    print '>>> FINISHED with the clusters generation.'
    print ''

if do_clu2STC:
    print '>>> Transfer cluster to STC ....'
    from stat_cluster import clu2STC
    import os.path as op
    for evt in evt_list:
        evt1 = '1sample_%s' %(evt)
        fn_cluster1 = stcs_path + '%d_pthr%.4f_%s.npz' %(permutation, p_th, evt)
        if op.isfile(fn_cluster1):
            clu2STC(fn_cluster1, p_thre=p_v)
        evt2 = '2sample_%s' %(evt)
        fn_cluster2 = stcs_path + '%d_pthr%.4f_%s.npz' % (permutation/2, p_th, evt)
        if op.isfile(fn_cluster2):
            clu2STC(fn_cluster2, p_thre=p_v)
        