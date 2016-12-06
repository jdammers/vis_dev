import os
import matplotlib.pyplot as pl
import mne
from jumeg.jumeg_utils import get_files_from_list
from jumeg.jumeg_preprocessing import apply_filter
#################################################################
#
# filename conventions
#
# >>> I assume that this will be provided in a different way <<<
# >>> probably by Frank's new routines (?) <<<
#
#################################################################

ext_empty_cov = '-cov.fif'
ext_empty_raw = '-empty.fif'
def apply_empty(fname_empty_room, require_filter=True):
    from jumeg.jumeg_noise_reducer import noise_reducer
    fner = get_files_from_list(fname_empty_room)
    nfiles = len(fner)
    # loop across all filenames
    for ifile in range(nfiles):
        fn_in = fner[ifile]
        path_in, name = os.path.split(fn_in)
        fn_empty_nr = fn_in[:fn_in.rfind(ext_empty_raw)] + ',nr-empty.fif'
        noise_reducer(fn_in, refnotch=50, detrending=False, fnout=fn_empty_nr)
        #noise_reducer(fn_empty_nr, refnotch=60, detrending=False, fnout=fn_empty_nr)
        noise_reducer(fn_empty_nr, reflp=5, fnout=fn_empty_nr)
        fn_in = fn_empty_nr
        if require_filter:
            print "Filtering with preset settings..."
            # filter empty room raw data
            apply_filter(fn_in, flow=1, fhigh=45, order=4, njobs=4)


def apply_cov(fname_empty_room, filtered = True):

    '''
    Creates the noise covariance matrix from an empty room file.

    Parameters
    ----------
    fname_empty_room : String containing the filename
        of the empty room file (must be a fif-file)
        File name should end with -raw.fif in order to have proper output filenames.
    require_filter: bool
        If true, the empy room file is filtered before calculating
        the covariance matrix. (Beware, filter settings are fixed.)
    require_noise_reducer: bool
        If true, a noise reducer is applied on the empty room file.
        The noise reducer frequencies are fixed to 50Hz, 60Hz and
        to frequencies less than 5Hz i.e. the reference channels are filtered to
        these frequency ranges and then signal obtained is removed from
        the empty room raw data. For more information please check the jumeg noise reducer.
    verbose : bool, str, int, or None
        If not None, override default verbose level
        (see mne.verbose).
        default: verbose=None
    '''

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne import compute_raw_covariance as cp_covariance
    from mne import write_cov, pick_types
    from mne.io import Raw
    fner = get_files_from_list(fname_empty_room)
    nfiles = len(fner)

    # loop across all filenames
    for ifile in range(nfiles):
        fn_in = fner[ifile]
        
        fn_fig1 = fn_in[:fn_in.rfind(ext_empty_raw)] + ',Magnetometers.tiff'
        fn_fig2 = fn_in[:fn_in.rfind(ext_empty_raw)] + ',Eigenvalue_index.tiff'
        #fn_out = fn_in[:fn_in.rfind(ext_empty_raw)] + ext_empty_cov
        path_in, name = os.path.split(fn_in)
        subject = name.split('_')[0]
        # read in data
        raw_empty = Raw(fn_in)

        # pick MEG channels only
        picks = pick_types(raw_empty.info, meg=True, exclude='bads')
        
        # calculate noise-covariance matrix
        noise_cov_mat = cp_covariance(raw_empty, tmin=None, tmax=None,
                                       tstep=0.2, picks=picks)
        #noise_cov_mat = cp_covariance(raw_empty, picks=picks)
        fig1, fig2 = mne.viz.plot_cov(noise_cov_mat, raw_empty.info)
        # write noise-covariance matrix to disk
        if filtered:
            fn_out = path_in + '/%s_empty,fibp1-45' %subject + ext_empty_cov
        else:
            fn_out = path_in + '/%s_empty' %subject + ext_empty_cov
        write_cov(fn_out, noise_cov_mat)
        fig1.savefig(fn_fig1)
        fig2.savefig(fn_fig2)
        pl.close('all')
