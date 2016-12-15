import glob

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
do_t_test = False # Spatial clustering
do_clu2STC = True # Transfer cluster arrays into STC objects.

# Set the option for stimulus or response
conf_per = True
conf_res = False
#conf_per = sys.argv[1]
#conf_res = sys.argv[2]

stcs_path = subjects_dir + 'fsaverage/conf_stc/'
n_subjects = 13 # The amount of subjects
st_list = ['LLst', 'RRst', 'RLst',  'LRst'] # stimulus events
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt'] # response events
st_max, st_min = 0.3, 0. # time period of stimulus
res_max, res_min = 0.1, -0.2 # time period of response

# parameter of Morphing
grade = 5
method = 'dSPM'
snr = 2
template = 'fsaverage' # The common brain space

# Parameters of Moving average
mv_window = 10 # miliseconds
overlap = 0 # miliseconds
nfreqs = 678.17

# The parameters for clusterring test
permutation = 16384
p_th = 0.0001 # spatial p-value
p_v = 0.005 # comparisons corrected p-value



#stimulus
if conf_per == True:
#if conf_per == 'True':
    evt_list = st_list
    tmin, tmax = st_min, st_max
    conf_type = 'sti'
    baseline = True

#response
elif conf_res == True:
#elif conf_res == 'True':
    evt_list = res_list
    tmin, tmax = res_min, res_max
    conf_type = 'res'
    baseline = False


############################################
# make inverse operator of filtered evoked 
#------------------------------------------
if do_make_inve:
    from stat_cluster import apply_inverse_ave
    print '>>> Calculate inverse solution ....'
    fn_evt_list = glob.glob(subjects_dir+'*[0-9]/MEG/*fibp1-45,evt_LLst_bc-ave.fif')
    apply_inverse_ave(fn_evt_list, subjects_dir)
    print '>>> FINISHED with inverse solution.'
    print ''       
        

############################################
# inverse evo to the source space 
#------------------------------------------

if do_inver_ave:
    print '>>> Calculate STC ....'
    from stat_cluster import apply_STC_ave
    fn_evt_list = glob.glob(subjects_dir+'*[0-9]/MEG/*fibp1-45,evt_*_bc-ave.fif')
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
        fn_stc_list = glob.glob(subjects_dir+'*[0-9]/MEG/*fibp1-45,evt_%s_bc-lh.stc' %evt)
        morph_STC(fn_stc_list, grade, subjects_dir, template, event=evt, baseline=baseline)
    print '>>> FINISHED with morphed STC generation.'
    print ''

###################################################
# Group STCs into pre- and post-stimulus (events)
#--------------------------------------------------
if do_group_STC:
    print '>>> Calculate Matrix for contrasts ....'
    from stat_cluster import Ara_contr_base
    Ara_contr_base(evt_list, tmin, tmax, conf_type, stcs_path, n_subjects=n_subjects,
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
        evt = '%s_%s' %(conf_type, evt)
        fnmat = stcs_path + evt + '.npz'
        permutation1 = permutation 
        #conf_mark = 'ttest_' + conf_type
        print '>>> load Matrix for contrasts ....'
        import numpy as np
        npz = np.load(fnmat)
        tstep = npz['tstep'].flatten()[0]
        X = npz['X']
        print '>>> FINISHED with the group matrix loaded.'
        print ''
        X1 = X[:, :, :n_subjects, 0]
        X2 = X[:, :, :n_subjects, 1]
        fn_clu_out = stcs_path + 'Ttestpermu%d_pthr%.4f_%s.npz' %(permutation1, p_th, evt)
        Y = X1 - X2  # make paired contrast
        stat_clus(Y, tstep, n_per=permutation1, p_threshold=p_th, p=p_v,
                fn_clu_out=fn_clu_out)
        print Y.shape
        del Y
        clu2STC(fn_clu_out, p_thre=p_v)
    print '>>> FINISHED with the clusters generation.'
    print ''

if do_clu2STC:
    print '>>> Transfer cluster to STC ....'
    from stat_cluster import clu2STC
    for evt in evt_list:
        evt = '%s_%s' %(conf_type, evt)
        fn_cluster = stcs_path + 'Ttestpermu%d_pthr%.4f_%s.npz' %(permutation, p_th, evt)
        clu2STC(fn_cluster, p_thre=p_v)
