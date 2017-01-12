
import os
import numpy as np
#from scipy import stats as stats
#import glob

import mne
from mne import (spatial_tris_connectivity,
                 grade_to_tris)
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc, spatio_temporal_cluster_test)

from jumeg.jumeg_preprocessing import get_files_from_list

def reset_directory(path=None):
    """
    check whether the directory exits, if yes, recreat the directory
    ----------
    path : the target directory.
    """
    import shutil
    isexists = os.path.exists(path)
    if isexists:
        shutil.rmtree(path)
    os.makedirs(path)


def set_directory(path=None):
    """
    check whether the directory exits, if no, creat the directory
    ----------
    path : the target directory.

    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        
def find_files(rootdir='.', pattern='*'):
    import os, fnmatch
    files = []
    for root, dirnames, filenames in os.walk(rootdir):
      for filename in fnmatch.filter(filenames, pattern):
          files.append(os.path.join(root, filename))

    files = sorted(files)

    return files
                
def apply_inverse_ave(fnevo, subjects_dir):
    ''' Make individual inverse operator.

        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        subjects_dir: The total bath of all the subjects.

    '''
    from mne import make_forward_solution
    from mne.minimum_norm import make_inverse_operator, write_inverse_operator
    fnlist = get_files_from_list(fnevo)

    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        fn_inv = fn_path + '/%s_fibp1-45,ave-inv.fif' % subject
        subject_path = subjects_dir + '/%s' % subject

        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty,fibp1-45-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-5-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        [evoked] = mne.read_evokeds(fname)
        evoked.pick_types(meg=True, ref_meg=False)
        noise_cov = mne.read_cov(fn_cov)
        # noise_cov = dSPM.cov.regularize(noise_cov, evoked.info,
        #                               mag=0.05, grad=0.05, proj=True)
        fwd = make_forward_solution(evoked.info, fn_trans, fn_src, fn_bem)
        fwd['surf_ori'] = True
        inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2,
                                    depth=0.8, limit_depth_chs=False)
        write_inverse_operator(fn_inv, inv)


def apply_STC_ave(fnevo, method='dSPM', snr=3.0):
    ''' Inverse evoked data into the source space.
        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        method:string
            Inverse method, 'dSPM' or 'mne'
        snr: float
            Signal to noise ratio for inverse solution.
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import apply_inverse, read_inverse_operator
    fnlist = get_files_from_list(fnevo)
    # loop across all filenames
    for fname in fnlist:
        name = os.path.basename(fname)
        fn_path = os.path.split(fname)[0]
        fn_stc = fname[:fname.rfind('-ave.fif')]
        # fn_inv = fname[:fname.rfind('-ave.fif')] + ',ave-inv.fif'
        subject = name.split('_')[0]
        fn_inv = fn_path + '/%s_fibp1-45,ave-inv.fif' % subject
        snr = snr
        lambda2 = 1.0 / snr ** 2
        # noise_cov = mne.read_cov(fn_cov)
        [evoked] = mne.read_evokeds(fname)
        evoked.pick_types(meg=True, ref_meg=False)
        inv = read_inverse_operator(fn_inv)
        stc = apply_inverse(evoked, inv, lambda2, method,
                            pick_ori='normal')
        stc.save(fn_stc)


def morph_STC(fn_list, method, template='fsaverage', btmin=-0.3, btmax=0., 
               subjects_dir=None):
    '''
        Morph individual STC into the common brain space.

        Parameter
        ------------------------------------
        fn_list: list
            The paths of the individual STCs.
        subjects_dir: The total bath of all the subjects.
        template: string
            The subject name as the common brain.
        btmin, btmax: float
            If 'baseline' is True, baseline is croped using this period.

    '''
    from mne import read_source_estimate, morph_data
    for fname in fn_list:
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        cond = name.split('_')[-2]
        import pdb
        pdb.set_trace()
        stc_name = name[:name.rfind('-lh.stc')]
        min_dir = subjects_dir + '/%s' % template
        # this path used for ROI definition
        stc_path = min_dir + '/%_ROIs/%s' % (method, subject)
        # fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
        set_directory(stc_path)
        # Morph STC
        stc = read_source_estimate(fname)
        stc_morph = morph_data(subject, template, stc, grade=5, subjects_dir=subjects_dir)
        stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
        if cond[2:] == 'st':
            stc_base = stc_morph.crop(btmin, btmax)
            stc_base.save(stc_path + '/%s_%s_baseline' % (subject, cond[:2]),
                          ftype='stc')
                          
#################################################################################
# Spatial clustering
#################################################################################
    
def Ara_norm(subjects, ncond, stcs_dir, out_path):
    ''' 
        Arange group arrays for pre vs post stimulus, zscore them and make
        abs.
        
        Parameters:
        --------------
        subjects: list,
            the subjects list.
        ncond: int,
            the amount of experimental conditions.
        stcs_dir: string,
            the path for searching stcs of each condition.
        out_path: string,
            the path for storing group z-sored arrays.
    '''
    nsubjects = len(subjects)
    fn_list = find_files(stcs_dir, pattern='*evt*_bc-lh.stc') 
    fn_list = np.reshape(fn_list,(nsubjects,ncond)) 
    for icond in range(ncond):
        fn_tmp = fn_list[0, icond]
        name = os.path.basename(fn_tmp)
        cond = name.split('_')[-2]
        A_evt = []
        A_base = []
        for isubj in range(nsubjects):
            fn_stc = fn_list[isubj, icond]
            name = os.path.basename(fn_stc)
            stc = mne.read_source_estimate(fn_stc)
            if cond[2:] == 'st':
                stc.crop(0, 0.3)
            elif cond[2:] == 'rt':
                stc.crop(-0.2, 0.1)
            #data = stc.data.flatten()
            data = stc.data
            path = os.path.dirname(fn_stc)
            subject = name.split('_')[0]
            fn_base = path + '/%s_%s_baseline-lh.stc' %(subject, cond[:2])
            base_stc = mne.read_source_estimate(fn_base)
            #base_data = base_stc.data.flatten()
            base_data = base_stc.data
            A_evt.append(data)
            A_base.append(base_data)
              
        A_evt = np.array(A_evt)
        A_base = np.array(A_base)
        tstep = stc.tstep
        fsave_vertices = stc.vertices
        ctime = min([A_evt.shape[-1], A_base.shape[-1]])
        X = [A_evt[:, :, :ctime], A_base[:, :, :ctime]]
        del A_evt, A_base
        # save data matrix
        X = np.array(X)
        g_mean = X[1].mean()
        g_std = X[1].std()
        print g_mean, g_std, cond
        X[0] = (X[0] - g_mean) / g_std
        X[1] = (X[1] - g_mean) / g_std
        X = np.abs(X)  # only magnitude
        X = X.transpose(0,1,3,2)
        np.savez(out_path + 'Group_%s.npz' % (cond), X=X, tstep=tstep,
                 fsave_vertices=fsave_vertices)
        del X
                    

def _exclu_vers():    
    fn_lmedial = '/home/uais_common/dong/freesurfer/subjects/fsaverage/label/lh.Medial_wall.label'
    lh_medial = mne.read_label(fn_lmedial)
    lh_mvers = lh_medial.get_vertices_used()
    fn_rmedial = '/home/uais_common/dong/freesurfer/subjects/fsaverage/label/rh.Medial_wall.label'
    rh_medial = mne.read_label(fn_rmedial)
    rh_mvers = rh_medial.get_vertices_used()
    rh_mvers = rh_mvers + 10242
    del_vers = list(lh_mvers) + list(rh_mvers)
    return del_vers



def sample1_clus(fn_list, n_per=8192, pct=95, p=0.05, del_vers=_exclu_vers()):
    '''
      Calculate significant clusters using 1sample ttest.

      Parameter
      ---------
      fn_list: list
        Paths of group arrays
      n_per: int
        The permutation for ttest.
      pct: int or float.
        The percentile of the baseline distribution.
      p: float
        The corrected p_values for comparisons.
      del_vers: None or _exclu_vers
        If is '_exclu_vers', delete the vertices in the medial wall.
    '''

    print('Computing connectivity.')
    connectivity = spatial_tris_connectivity(grade_to_tris(5))

    # Using the percentile of baseline array as the distribution threshold
    for fn_npz in fn_list:
        fn_out = fn_npz[:fn_npz.rfind('.npz')] + '_1sample_%d_pct%.1f.npz' %(n_per, pct)
        npz = np.load(fn_npz)
        tstep = npz['tstep'].flatten()[0]
        #    Note that X needs to be a multi-dimensional array of shape
        #    samples (subjects) x time x space, so we permute dimensions
        X = npz['X']
        t_threshold = np.percentile(X[1], pct)
        X = X[0] - X[1]
        fsave_vertices = [np.arange(X.shape[-1]/2), np.arange(X.shape[-1]/2)]
    
        #    Now let's actually do the clustering. This can take a long time...
        #    Here we set the threshold quite high to reduce computation.
        #n_subjects = X.shape[0]
        #t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
        print('Clustering.')
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_1samp_test(X, connectivity=connectivity,
                                            n_jobs=1, threshold=t_threshold,
                                            n_permutations=n_per, spatial_exclude=del_vers)
    
        #    Now select the clusters that are sig. at p < 0.05 (note that this value
        #    is multiple-comparisons corrected).
        good_cluster_inds = np.where(cluster_p_values < p)[0]
        print 'the amount of significant clusters are: %d' %good_cluster_inds.shape
    
        # Save the clusters as stc file
        np.savez(fn_out, clu=clu, tstep=tstep, fsave_vertices=fsave_vertices)
        assert good_cluster_inds.shape != 0, ('Current p_threshold is %f %p_thr,\
                                    maybe you need to reset a lower p_threshold')



def sample2_clus(fn_list, n_per=8192, pct=95, p=0.05, del_vers=_exclu_vers()):
    '''
      Calculate significant clusters using 2 sample ftest.

      Parameter
      ---------
      fn_list: list
        Paths of group arrays
      n_per: int
        The permutation for ttest.
      pct: int or float.
        The percentile of the baseline distribution.
      p: float
        The corrected p_values for comparisons.
      del_vers: None or _exclu_vers
        If is '_exclu_vers', delete the vertices in the medial wall.
    '''
    for fn_npz in fn_list:
        fn_out = fn_npz[:fn_npz.rfind('.npz')] + '_2sample_%d_pct%.1f.npz' %(n_per, pct)
        npz = np.load(fn_npz)
        tstep = npz['tstep'].flatten()[0]
        #    Note that X needs to be a multi-dimensional array of shape
        #    samples (subjects) x time x space, so we permute dimensions
        X = npz['X']
        f_threshold = np.percentile(X[1], pct)
        fsave_vertices = [np.arange(X.shape[-1]/2), np.arange(X.shape[-1]/2)]
        print('Clustering...')
        connectivity = spatial_tris_connectivity(grade_to_tris(5))
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_test(X, n_permutations=n_per, #step_down_p=0.001,
                                        connectivity=connectivity, n_jobs=1,
                                        # threshold=t_threshold, stat_fun=stats.ttest_ind)
                                        threshold=f_threshold, spatial_exclude=del_vers)
    
        #    Now select the clusters that are sig. at p < 0.05 (note that this value
        #    is multiple-comparisons corrected).
        good_cluster_inds = np.where(cluster_p_values < p)[0]
        print 'the amount of significant clusters are: %d' % good_cluster_inds.shape
    
        # Save the clusters as stc file
        np.savez(fn_out, clu=clu, tstep=tstep, fsave_vertices=fsave_vertices)
        assert good_cluster_inds.shape != 0, ('Current p_threshold is %f %p_thr,\
                                    maybe you need to reset a lower p_threshold')


def clu2STC(fn_list, p_thre=0.05):
    '''
        Generate STCs from significant clusters

        Parameters
        -------
        fn_list: string
            The paths of significant clusters.
        p_thre: float
            The corrected p_values.
        
    '''
    for fn_cluster in fn_list:
        fn_stc_out = fn_cluster[:fn_cluster.rfind('.npz')] + ',pthre_%.3f' % (p_thre)
        npz = np.load(fn_cluster)
        clu = npz['clu']
        fsave_vertices = list(npz['fsave_vertices'])
        tstep = npz['tstep'].flatten()[0]
        stc_all_cluster_vis = summarize_clusters_stc(clu, p_thre, tstep=tstep,
                                                    vertices=fsave_vertices,
                                                    subject='fsaverage')
    
        stc_all_cluster_vis.save(fn_stc_out)
