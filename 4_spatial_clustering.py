import glob
from stat_cluster import set_directory
#####################################################
# Aplly spatio-temporal clustering for ROI defintion
#####################################################
''' From here data will be calculated in the source
    space. 
'''


do_make_inve = False # Generate inverse operator
do_inver_ave = False # make STC
do_morph_STC = False # Morph individual STC
do_group_STC = False # Group morphed STC into pre and post-stimuls (events)
do_t_test = True # Spatial clustering
do_2sample = False #2 sample test
do_clu2STC = False # Transfer cluster arrays into STC objects.

#The main path for ROI definition
subjects_dir = '/home/uais_common/dong/freesurfer/subjects/'

# parameter of Inversing and Morphing
method = 'dSPM'
snr = 3


# The parameters for clusterring test
permutation = 8192
perc = 99 # The percentile of baseline STCs distributions
p_v = 0.05 # comparisons corrected p-value

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
    fn_stc_list = glob.glob(subjects_dir+'*[0-9]/MEG/*bcc,nr,fibp1-45,ar,evt_*_bc-lh.stc')
    morph_STC(fn_stc_list, method, subjects_dir=subjects_dir)
    print '>>> FINISHED with morphed STC generation.'
    print ''

###################################################
# Group STCs into pre- and post-stimulus (events)
#--------------------------------------------------

#The path for storing the results related with ROIs
stcs_path = subjects_dir + 'fsaverage/conf_stc/'
set_directory(stcs_path)

if do_group_STC:
    print '>>> Calculate Matrix for contrasts ....'
    from stat_cluster import Ara_norm
    subjects = ['203731', '201195', '203709', '203792', '203969', '203822',
                    '203929', '203867', '203147', '203267', '203840', '203780', '203288']   
    ncond = 8 
    stcs_dir = subjects_dir + 'fsaverage/%s_ROIs' %method
    Ara_norm(subjects, ncond, stcs_dir, stcs_path)
    print '>>> FINISHED with a group matrix generation.'
    print ''

##################################################################
# Spatial clustering for significant clusters related with events
#----------------------------------------------------------------
if do_t_test:
    print '>>> ttest for clustering ....'
    from stat_cluster import sample1_clus 
    fn_list = glob.glob(stcs_path + 'Group_*.npz')
    sample1_clus(fn_list, n_per=permutation, pct=perc, p=p_v, del_vers=None)
    print '>>> FINISHED with the clusters generation.'
    print ''

###############################################################################
# Clustering using 2sample f-test
# -----------------
if do_2sample:
    ''' This comparison is suitable for the samples from different entireties
    '''
    print '>>> 2smpletest for clustering ....'
    from stat_cluster import sample2_clus 
    fn_list = glob.glob(stcs_path + 'Group_*.npz')
    sample2_clus(fn_list, n_per=permutation/2, pct=perc, p=p_v, del_vers=None)
    print '>>> FINISHED with the clusters generation.'
    print ''
    
###############################################################################
# Transfer significant cluster into STCs
# ----------------------------------------
if do_clu2STC:
    print '>>> Transfer cluster to STC ....'
    from stat_cluster import clu2STC
    fn_list = glob.glob(stcs_path + 'Group_*sample*.npz')
    clu2STC(fn_list, p_thre=p_v)
    
        