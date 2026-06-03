#%%###########################################################################
"""
WF correction analysis: compare uncorrected (Kolmogorov) and corrected
(TNN residual) PSF/PL propagation datasets.
"""

#%%########################################################################
# import IPython; _ip = IPython.get_ipython()
# if _ip:
#     _ip.run_line_magic('load_ext', 'autoreload')
#     _ip.run_line_magic('autoreload', '2')

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from seidr.seidr_functions_misc import plot_wf_psf_lp_pl, plot_histograms

#%%########################################################################
### Filenames ###

dir_data = "/import/roci1/nlon0790/Results/psf_prop/"
dir_plot  = Path("/suphys/nlon0790/Documents/python_code/seidr2.0/figures/")

## Uncorrected: Kolmogorov propagation dataset
f_uncorr = dir_data + "hms-pl6c_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_20260526-2331.npz"

## Corrected: TNN residual propagation dataset
f_corr   = dir_data + "hms-pl6c_preds_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_20260603-1240.npz"

## Corrected (reload): Kolmogorov + TNN residual re-propagation dataset
f_uncorr_reload = dir_data + "hms-pl6c_kolpreds_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_20260603-1318.npz"

#%%########################################################################
### Split / Sequence Parameters (must match those used during TNN training) ###

val_split  = 0.15
test_split = 0.15
seq_length = 50   # TNN input sequence length; predictions align to test[seq_length:]

#%%########################################################################
### Load Datasets ###

print("Loading uncorrected dataset (test split, seq-aligned)...")
_uncorr_full = np.load(f_uncorr, allow_pickle=True)
modelabels   = _uncorr_full['modelabels']

_N      = _uncorr_full['pupil_wf'].shape[0]
_n_val  = int(_N * val_split)
_n_test = int(_N * test_split)
# test occupies [_i1 : _i2]; TNN predicts output[j] = test[j + seq_length],
# so aligned uncorrected comparison starts at test[seq_length:]
_i1, _i2 = _n_val + seq_length, _n_val + _n_test
print(f"  Full dataset: {_N} samples  →  test slice [{_n_val}:{_n_val+_n_test}]"
      f"  →  seq-aligned [{_i1}:{_i2}] ({_i2-_i1} samples)")

uncorr = {
    'pupil_wf':       _uncorr_full['pupil_wf'][_i1:_i2],
    'psf_fields':     _uncorr_full['psf_fields'][_i1:_i2],
    'lp_powers':      _uncorr_full['lp_powers'][_i1:_i2],
    'pl_powers':      _uncorr_full['pl_powers'][_i1:_i2],
    'total_coupling': _uncorr_full['total_coupling'][_i1:_i2],
    'modelabels':     modelabels,
}
print(f"  pupil_wf:       {uncorr['pupil_wf'].shape}")
print(f"  psf_fields:     {uncorr['psf_fields'].shape}")
print(f"  lp_powers:      {uncorr['lp_powers'].shape}")
print(f"  pl_powers:      {uncorr['pl_powers'].shape}")
print(f"  total_coupling: {uncorr['total_coupling'].shape}")

print("\nLoading corrected dataset...")
_corr = np.load(f_corr, allow_pickle=True)

corr = {
    'pupil_wf':       _corr['pupil_wf'],
    'psf_fields':     _corr['psf_fields'],
    'lp_powers':      _corr['lp_powers'],
    'pl_powers':      _corr['pl_powers'],
    'total_coupling': _corr['total_coupling'],
    'modelabels':     modelabels,
}
print(f"  pupil_wf:       {corr['pupil_wf'].shape}")
print(f"  psf_fields:     {corr['psf_fields'].shape}")
print(f"  lp_powers:      {corr['lp_powers'].shape}")
print(f"  pl_powers:      {corr['pl_powers'].shape}")
print(f"  total_coupling: {corr['total_coupling'].shape}")

print("\nLoading uncorr_reload dataset...")
_reload = np.load(f_uncorr_reload, allow_pickle=True)

uncorr_reload = {
    'pupil_wf':       _reload['pupil_wf'],
    'psf_fields':     _reload['psf_fields'],
    'lp_powers':      _reload['lp_powers'],
    'pl_powers':      _reload['pl_powers'],
    'total_coupling': _reload['total_coupling'],
    'modelabels':     modelabels,
}
print(f"  pupil_wf:       {uncorr_reload['pupil_wf'].shape}")
print(f"  psf_fields:     {uncorr_reload['psf_fields'].shape}")
print(f"  lp_powers:      {uncorr_reload['lp_powers'].shape}")
print(f"  pl_powers:      {uncorr_reload['pl_powers'].shape}")
print(f"  total_coupling: {uncorr_reload['total_coupling'].shape}")

n_samples = uncorr['pupil_wf'].shape[0]
assert corr['pupil_wf'].shape[0] == n_samples, \
    f"Dataset sizes do not match: uncorr {n_samples} vs corr {corr['pupil_wf'].shape[0]}"
assert uncorr_reload['pupil_wf'].shape[0] == n_samples, \
    f"Dataset sizes do not match: uncorr {n_samples} vs uncorr_reload {uncorr_reload['pupil_wf'].shape[0]}"
print(f"\n{n_samples} matched samples loaded (test[{seq_length}:]).")

#%%########################################################################
### Plot Example Row ###

idx = np.random.randint(0, n_samples)
print(f"Example index: {idx}")

plot_wf_psf_lp_pl(uncorr, idx=idx)

#%%########################################################################

plot_wf_psf_lp_pl(corr, idx=idx)

#%%########################################################################

plot_wf_psf_lp_pl(uncorr_reload, idx=idx)

#%%########################################################################
### Calculate the ratio of power in core 0

core0_ratio_uncorr        = uncorr['pl_powers'][:, 0]        / uncorr['pl_powers'].sum(axis=1)
core0_ratio_corr          = corr['pl_powers'][:, 0]          / corr['pl_powers'].sum(axis=1)
core0_ratio_uncorr_reload = uncorr_reload['pl_powers'][:, 0] / uncorr_reload['pl_powers'].sum(axis=1)

## Calculate mean and std of the core 0 ratio
mean_uncorr        = core0_ratio_uncorr.mean()
std_uncorr         = core0_ratio_uncorr.std()
mean_corr          = core0_ratio_corr.mean()
std_corr           = core0_ratio_corr.std()
mean_uncorr_reload = core0_ratio_uncorr_reload.mean()
std_uncorr_reload  = core0_ratio_uncorr_reload.std()

print(f"Core 0 ratio (uncorrected):    mean={mean_uncorr:.4f}, std={std_uncorr:.4f}")
print(f"Core 0 ratio (corrected):      mean={mean_corr:.4f}, std={std_corr:.4f}")
print(f"Core 0 ratio (uncorr_reload):  mean={mean_uncorr_reload:.4f}, std={std_uncorr_reload:.4f}")


#%%#########################################################################
### Histograms ###


plot_histograms(
    [core0_ratio_uncorr, core0_ratio_corr],
    labels=['Uncorrected', 'Corrected'],
    colors=["#D55E00", "#0072B2"],  # dark teal / coral
    bins=100,
    figsize=(8, 4),
    xlim=None,
    xlabel='Mode Selective Core Power Ratio ($P_{ms} / P_{all}$)',
    ylabel='Count',
    title=None,
    show_mean=False,
    save_path=None,
    dpi=150,
)

# %%
