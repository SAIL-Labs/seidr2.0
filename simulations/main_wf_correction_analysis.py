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
import matplotlib.pyplot as plt
import zernikePSF

from seidr.seidr_functions_misc import plot_wf_psf_lp_pl, plot_histograms, \
                                        load_dataset, make_pupil_mask, strehl_ratio

#%%###########################################################################

plt.style.use(['seaborn-v0_8-paper']) #-v0_8-paper

plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 5.5
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"

#%%########################################################################
### Paths ###

dir_data = "/import/roci1/nlon0790/Results/psf_prop/"
dir_plot = '/suphys/nlon0790/Documents/python_code/seidr2.0/figures/'

#%%########################################################################
### Control ###

wf_type = "baldr"   # "kolmogorov" or "tiptilt" or "baldr"

if wf_type == 'baldr':
    sequential_length = 100    # length of input sequences for sequential models
else:
    sequential_length = 50    # length of input sequences for sequential models
    
val_split   = 0.15
test_split  = 0.15

#%%########################################################################
### Dataset Registry ###
# Each entry: (filepath, is_raw)
#   is_raw=True  → uncorrected raw dataset; slice to test window [n_val : n_val+n_test]
#   is_raw=False → nn prediction dataset (already the test set); load all

_DATASETS_KOL = {
    ## uncorrected (raw)
    "kol_contig":     (dir_data + "hms-pl6c_kol_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_20260526-2331.npz",
                       True),
    "kol_rand":       (dir_data + "hms-pl6c_kol_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_rand_20260610-0917.npz",
                       True),
    ## corrected (nn predictions re-propagated)
    "tnn_kol_contig": (dir_data + "hms-pl6c_pred_tnn_kol_contig_wf_psf_lp_dataset_20260609-2055.npz",
                       False),
    "cnn_kol_contig": (dir_data + "hms-pl6c_pred_cnn_kol_contig_wf_psf_lp_dataset_20260609-2119.npz",
                       False),
    "tnn_kol_rand":   (dir_data + "hms-pl6c_pred_tnn_kol_rand_wf_psf_lp_dataset_20260610-1303.npz",
                       False),
    "cnn_kol_rand":   (dir_data + "hms-pl6c_pred_cnn_kol_rand_wf_psf_lp_dataset_20260610-1256.npz",
                       False),
}

_DATASETS_BALDR = {
    ## uncorrected (raw)
    "baldr_contig":     (dir_data + "hms-pl6c_baldr_wf_psf_lp_dataset_20260701-2025.npz",
                         True),
    ## corrected (tnn predictions re-propagated); no CNN or rand variant yet
    "tnn_baldr_contig": (dir_data + "hms-pl6c_pred_tnn_baldr_contig_wf_psf_lp_dataset_20260720-1302.npz",
                         False),
    "cnn_baldr_contig": (dir_data + "hms-pl6c_pred_cnn_baldr_contig_wf_psf_lp_dataset_20260720-1320.npz",
                         False),
}

_DATASETS_TIPTILT = {
    ## uncorrected (raw)
    "tiptilt_contig": (dir_data + "hms-pl6c_tiptilt_wf_psf_lp_dataset_contig_20260608-1913.npz",
                       True),
    "tiptilt_rand":   (dir_data + "hms-pl6c_tiptilt_wf_psf_lp_dataset_rand_20260609-0903.npz",
                       True),
    
    ## corrected (nn predictions re-propagated)
    "tnn_tiptilt_contig": (dir_data + "hms-pl6c_pred_tnn_tiptilt_contig_wf_psf_lp_dataset_20260610-1249.npz",
                           False),
    "cnn_tiptilt_contig": (dir_data + "hms-pl6c_pred_cnn_tiptilt_contig_wf_psf_lp_dataset_20260610-0959.npz",
                           False),
    "tnn_tiptilt_rand":   (dir_data + "hms-pl6c_pred_tnn_tiptilt_rand_wf_psf_lp_dataset_20260609-2221.npz",
                           False),
    "cnn_tiptilt_rand":   (dir_data + "hms-pl6c_pred_cnn_tiptilt_rand_wf_psf_lp_dataset_20260609-2227.npz",
                           False),
}

if wf_type == "kolmogorov":
    _DATASETS    = _DATASETS_KOL
    _k_contig    = 'kol_contig'
    _k_rand      = 'kol_rand'
    _k_tnn_contig = 'tnn_kol_contig'
    _k_cnn_contig = 'cnn_kol_contig'
    _k_tnn_rand  = 'tnn_kol_rand'
    _k_cnn_rand  = 'cnn_kol_rand'
    _wft          = 'Kol'

elif wf_type == "tiptilt":
    _DATASETS    = _DATASETS_TIPTILT
    _k_contig    = 'tiptilt_contig'
    _k_rand      = 'tiptilt_rand'
    _k_tnn_contig = 'tnn_tiptilt_contig'
    _k_cnn_contig = 'cnn_tiptilt_contig'
    _k_tnn_rand  = 'tnn_tiptilt_rand'
    _k_cnn_rand  = 'cnn_tiptilt_rand'
    _wft          = 'TipTilt'
    
elif wf_type == "baldr":
    _DATASETS     = _DATASETS_BALDR
    _k_contig     = 'baldr_contig'
    _k_rand       = 'baldr_contig'       # no rand variant
    _k_tnn_contig = 'tnn_baldr_contig'
    _k_cnn_contig = 'cnn_baldr_contig'
    _k_tnn_rand   = 'tnn_baldr_contig'   # no rand variant
    _k_cnn_rand   = 'cnn_baldr_contig'   # no rand variant
    _wft          = 'Post-Baldr'

#%%########################################################################
### Load Datasets ###

_KEYS = ['pupil_wf', 'psf_fields', 'lp_powers', 'pl_powers', 'total_coupling']


datasets = {}
for label, (fpath, is_raw) in _DATASETS.items():
    print(f"Loading {label} ...")
    datasets[label] = load_dataset(fpath, is_raw, 
                                   val_split, test_split, _KEYS)
    print(f"  pupil_wf: {datasets[label]['pupil_wf'].shape}")

first_label = next(iter(datasets))
modelabels = datasets[first_label]['modelabels']
print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")

# Override pupil_wf for prediction datasets with the matching raw input wavefront so all
# datasets carry the same original wavefront. residual_wf retains the NN residual.
# TNN contig predictions are offset by seq_length (sliding window), so the raw slice
# starts at offset to align physical frames.
_raw_wf_lookup = {
    'tnn_kol_contig':     ('kol_contig',     sequential_length),
    'cnn_kol_contig':     ('kol_contig',     0),
    'tnn_kol_rand':       ('kol_rand',       0),
    'cnn_kol_rand':       ('kol_rand',       0),
    'tnn_tiptilt_contig': ('tiptilt_contig', sequential_length),
    'cnn_tiptilt_contig': ('tiptilt_contig', 0),
    'tnn_tiptilt_rand':   ('tiptilt_rand',   0),
    'cnn_tiptilt_rand':   ('tiptilt_rand',   0),
    'tnn_baldr_contig':   ('baldr_contig',   sequential_length),
    'cnn_baldr_contig':   ('baldr_contig',   0),
}

for label, (raw_label, offset) in _raw_wf_lookup.items():
    if label not in datasets or raw_label not in datasets:
        continue
    n_pred = datasets[label]['pupil_wf'].shape[0]
    datasets[label]['pupil_wf'] = datasets[raw_label]['pupil_wf'][offset : offset + n_pred]

#%%########################################################################
### Core 0 power ratio for all datasets ###

core0_ratios = {
    label: ds['pl_powers'][:, 0] / ds['pl_powers'].sum(axis=1)
    for label, ds in datasets.items()
}

print("\nCore 0 power ratio summary:")
for label, ratio in core0_ratios.items():
    print(f"  {label:20s}  mean={ratio.mean():.4f}  std={ratio.std():.4f}")


#%%#########################################################################
### % improvement in mode-selective core power ratio ###

_comparisons = [
    (_k_contig, _k_cnn_contig, 'CNN (contig)'),
    (_k_contig, _k_tnn_contig, 'TNN (contig)'),
    (_k_rand,   _k_cnn_rand,   'CNN (rand)'),
    (_k_rand,   _k_tnn_rand,   'TNN (rand)'),
]

print("\nMode-selective core power ratio improvement:")
print(f"  {'Model':<20s}  {'Raw mean':>10s}  {'Corr mean':>10s}  {'% increase':>10s}")
for raw_key, corr_key, name in _comparisons:
    mu_raw  = core0_ratios[raw_key].mean()
    mu_corr = core0_ratios[corr_key].mean()
    pct = 100 * (mu_corr - mu_raw) / mu_raw
    print(f"  {name:<20s}  {mu_raw:>10.4f}  {mu_corr:>10.4f}  {pct:>+10.2f}%")


#%%#########################################################################
### Strehl ratio (Marechal approximation) for all datasets ###

_pupil_mask = make_pupil_mask(datasets[first_label]['pupil_wf'].shape[-2:])

def strehl_marechal(wf, mask):
    """Strehl ratio via the Marechal approximation: S = exp(-sigma^2),
    where sigma^2 is the piston-removed phase variance over the pupil [rad^2]."""
    wf_px = wf[:, mask]                                   # (N, n_px)
    wf_px = wf_px - wf_px.mean(axis=-1, keepdims=True)    # remove piston
    sigma2 = wf_px.var(axis=-1)
    return np.exp(-sigma2)

strehl_ratios = {
    # corrected datasets carry the post-correction residual in 'residual_wf';
    # their 'pupil_wf' was overwritten above with the raw input wavefront for
    # side-by-side plotting, so it must NOT be used here.
    label: strehl_marechal(ds.get('residual_wf', ds['pupil_wf']), _pupil_mask)
    for label, ds in datasets.items()
}

#%%#############################################################################
print("\nStrehl ratio (Marechal approximation) summary:")
for label, s in strehl_ratios.items():
    print(f"  {label:20s}  mean={s.mean():.4f}  std={s.std():.4f}")

print("\nStrehl ratio improvement:")
print(f"  {'Model':<20s}  {'Raw mean':>10s}  {'Corr mean':>10s}  {'% increase':>10s}")
for raw_key, corr_key, name in _comparisons:
    mu_raw  = strehl_ratios[raw_key].mean()
    mu_corr = strehl_ratios[corr_key].mean()
    pct = 100 * (mu_corr - mu_raw) / mu_raw
    print(f"  {name:<20s}  {mu_raw:>10.4f}  {mu_corr:>10.4f}  {pct:>+10.2f}%")


#%%#########################################################################
### RMS Values Post-Wavefront Correction ###

def wf_rms(wf, mask):
    """Residual wavefront RMS [rad]: sqrt of the piston-removed phase
    variance over the pupil."""
    wf_px = wf[:, mask]                                   # (N, n_px)
    wf_px = wf_px - wf_px.mean(axis=-1, keepdims=True)    # remove piston
    return np.sqrt(wf_px.var(axis=-1))

wf_rms_residuals = {
    # same residual_wf vs pupil_wf caveat as the Strehl calculation above
    label: wf_rms(ds.get('residual_wf', ds['pupil_wf']), _pupil_mask)
    for label, ds in datasets.items()
}

#%%
print("\nResidual wavefront RMS [rad] summary:")
for label, r in wf_rms_residuals.items():
    print(f"  {label:20s}  mean={r.mean():.4f}  std={r.std():.4f}")

print("\nResidual wavefront RMS reduction:")
print(f"  {'Model':<20s}  {'Raw mean':>10s}  {'Corr mean':>10s}  {'% decrease':>10s}")
for raw_key, corr_key, name in _comparisons:
    mu_raw  = wf_rms_residuals[raw_key].mean()
    mu_corr = wf_rms_residuals[corr_key].mean()
    pct = 100 * (mu_raw - mu_corr) / mu_raw
    print(f"  {name:<20s}  {mu_raw:>10.4f}  {mu_corr:>10.4f}  {pct:>+10.2f}%")


#%%########################################################################
### Plot Example Row ###

idx = 500 if wf_type == "baldr" else 12781
print(f"Example index: {idx}")

# TNN contig predictions are indexed from the sliding window: tnn_contig[i] covers
# test frame i+seq_length. Subtract seq_length so all datasets plot the same physical frame.
_seq_offset = {
    'tnn_kol_contig':     -sequential_length,
    'tnn_tiptilt_contig': -sequential_length,
    'tnn_baldr_contig':   -sequential_length,
}

for label, ds in datasets.items():
    effective_idx = idx + _seq_offset.get(label, 0)
    n = ds['pupil_wf'].shape[0]
    if effective_idx < 0 or effective_idx >= n:
        continue
    print(f"\nPlotting example {idx} from dataset '{label}' (N={n}, access_idx={effective_idx}) ...")
    plot_wf_psf_lp_pl(ds, idx=effective_idx, save_plot=False,
                      fname_plot=dir_plot + f"seidr_wf_psf_lp_pl_{label}_example_{idx}.pdf")



#%%#########################################################################
### Histograms: NN Kol Contig vs Raw Kol Contig ###

_contig_keys   = [_k_contig, _k_tnn_contig, _k_cnn_contig]
_contig_labels = [f'Raw {_wft} (contig)', f'TNN {_wft} (contig)', f'CNN {_wft} (contig)']

_rand_keys   = [_k_rand, _k_tnn_rand, _k_cnn_rand]
_rand_labels = [f'Raw {_wft} (rand)', f'TNN {_wft} (rand)', f'CNN {_wft} (rand)']

#%%#########################################################################
### Histograms: NN Kol Contig vs Raw Kol Contig ###

plot_histograms(
    [core0_ratios[k] for k in _contig_keys],
    labels=_contig_labels,
    bins=100,
    figsize=(8, 4),
    xlim=None,
    xlabel='Mode-Selective Core Power Ratio ($P_{ms} / P_{all}$)',
    ylabel='PDF',
    title=f'{_wft} Contig: Uncorrected vs NN Corrected',
    show_mean=True,
    save_path=dir_plot + f'seidr_ms_core_ratio_{wf_type}_contig_nn_vs_raw.pdf',
    dpi=150,
)

#%%#########################################################################
### Histograms: Strehl Ratio, NN Contig vs Raw Contig ###

plot_histograms(
    [strehl_ratios[k] for k in _contig_keys],
    labels=_contig_labels,
    bins=100,
    figsize=(8, 4),
    xlim=[0.5, 0.95],
    xlabel='Strehl Ratio (Marechal approximation)',
    ylabel='PDF',
    title=f'{_wft} Contig: Uncorrected vs NN Corrected',
    show_mean=True,
    save_path=dir_plot + f'seidr_strehl_ratio_{wf_type}_contig_nn_vs_raw.pdf',
    dpi=150,
)

#%%#########################################################################
### Histograms: NN Kol Rand vs Raw Kol Rand ###

plot_histograms(
    [core0_ratios[k] for k in _rand_keys],
    labels=_rand_labels,
    bins=100,
    figsize=(8, 4),
    xlim=None,
    xlabel='Mode-Selective Core Power Ratio ($P_{ms} / P_{all}$)',
    ylabel='PDF',
    title=f'{_wft} Rand: Uncorrected vs NN Corrected',
    show_mean=True,
    save_path=dir_plot + f'seidr_ms_core_ratio_{wf_type}_rand_nn_vs_raw.pdf',
    dpi=150,
)

#%%#########################################################################
### Histograms: Strehl Ratio, NN Rand vs Raw Rand ###

plot_histograms(
    [strehl_ratios[k] for k in _rand_keys],
    labels=_rand_labels,
    bins=100,
    figsize=(8, 4),
    xlim=None,
    xlabel='Strehl Ratio (Marechal approximation)',
    ylabel='PDF',
    title=f'{_wft} Rand: Uncorrected vs NN Corrected',
    show_mean=True,
    save_path=dir_plot + f'seidr_strehl_ratio_{wf_type}_rand_nn_vs_raw.pdf',
    dpi=150,
)

#%%#########################################################################
### Histograms: Contig + Rand Combined ###

# Random first, Temporal second; plotting order Raw → CNN → TNN
_groups = {
    'Random':   ([_k_rand,  _k_tnn_rand, _k_cnn_rand],
                 ['Uncorrected', 'TNN', 'CNN']),
    'Temporal': ([_k_contig, _k_tnn_contig, _k_cnn_contig],
                 ['Uncorrected', 'TNN', 'CNN']),
}

# gray (uncorrected reference) + blue (CNN) + deep orange (TNN)
# gray reads as a neutral baseline; blue/orange are complementary and don't mix to mud
_colors = ['#969696', '#4393C3', '#D6604D']

fig, axes = plt.subplots(1, 2,
                         figsize=(10.5, 5.5),
                         sharey=True,
                         tight_layout=True)

for ax, (title, (keys, labels)) in zip(axes, _groups.items()):
    for key, label, color in zip(keys, labels, _colors):
        data = core0_ratios[key]
        weights = np.ones_like(data) / len(data)
        ax.hist(data, bins=100, histtype='stepfilled', alpha=0.4,
                color=color, label=label, weights=weights)
        ax.hist(data, bins=100, histtype='step', linewidth=1.8,
                color=color, weights=weights)
        ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2, alpha=0.8)

    ax.plot([], [], color='k', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
    ax.set_title('')
    ax.set_xlabel('')
    ax.legend(prop={'size': 17})
    ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', labelsize=14)
    # ax.set_title(title, fontsize=15, weight='semibold')
    # ax.set_xlabel('Mode-Selective Core Power Ratio', 
                #   fontsize=15)
    # ax.legend(prop={'size': 15, 'weight': 'semibold'})
    # ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)

axes[0].set_ylabel('')
# axes[0].set_ylabel('PDF', fontsize=15, weight='semibold')
fig.savefig(dir_plot + f'seidr_ms_core_ratio_{wf_type}_contig_rand_combined.svg', dpi=150)
plt.show()

#%%#########################################################################
### Histograms: Strehl Ratio, Contig + Rand Combined ###

fig, axes = plt.subplots(1, 2,
                         figsize=(12, 5.5),
                         sharey=True,
                         tight_layout=True)

for ax, (title, (keys, labels)) in zip(axes, _groups.items()):
    for key, label, color in zip(keys, labels, _colors):
        data = strehl_ratios[key]
        weights = np.ones_like(data) / len(data)
        ax.hist(data, bins=100, histtype='stepfilled', alpha=0.4,
                color=color, label=label, weights=weights)
        ax.hist(data, bins=100, histtype='step', linewidth=1.8,
                color=color, weights=weights)
        ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2, alpha=0.8)

    ax.plot([], [], color='k', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
    ax.set_xlabel('Strehl Ratio', fontsize=17)
    ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', labelsize=14)

axes[0].set_xlim([-0.01, 1.01])
axes[1].set_xlim([-0.01, 1.01])
axes[0].set_title('Random', fontsize=17)
axes[1].set_title('Temporal', fontsize=17)
axes[0].legend(prop={'size': 14})
axes[0].set_ylabel('PDF', fontsize=17)
fig.savefig(dir_plot + f'seidr_strehl_ratio_{wf_type}_contig_rand_combined.svg', dpi=150)
plt.show()

#%%#########################################################################
### Histogram: TNN Kol Contig vs Raw Kol Contig ###

_pairs = [(_k_contig, 'Uncorrected'), (_k_tnn_contig, 'Corrected')]
_pair_colors = ['#969696', '#D6604D']

fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
for (key, label), color in zip(_pairs, _pair_colors):
    data = core0_ratios[key]
    weights = np.ones_like(data) / len(data)
    ax.hist(data, bins=100, histtype='stepfilled', alpha=0.4,
            color=color, label=label, weights=weights)
    ax.hist(data, bins=100, histtype='step', linewidth=1.8,
            color=color, weights=weights)
    ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2, alpha=0.8)

ax.plot([], [], color='k', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
ax.set_title('')
ax.set_xlabel('Ratio of power in mode-selective core', # ($P_{ms} / P_{all}$)',
              fontsize=17)
ax.set_ylabel('PDF', fontsize=17)
ax.legend(prop={'size': 15})
ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='both', labelsize=14)
fig.savefig(dir_plot + f'seidr_ms_core_ratio_{wf_type}_contig_tnn_vs_raw.svg', dpi=150)
plt.show()

#%%#########################################################################
### Histogram: Strehl Ratio, TNN Kol Contig vs Raw Kol Contig ###

fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
for (key, label), color in zip(_pairs, _pair_colors):
    data = strehl_ratios[key]
    weights = np.ones_like(data) / len(data)
    ax.hist(data, bins=100, histtype='stepfilled', alpha=0.4,
            color=color, label=label, weights=weights)
    ax.hist(data, bins=100, histtype='step', linewidth=1.8,
            color=color, weights=weights)
    ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2, alpha=0.8)

ax.plot([], [], color='k', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
ax.set_title('')
ax.set_xlabel('Strehl Ratio (Marechal approximation)', fontsize=17)
ax.set_ylabel('PDF', fontsize=17)
ax.legend(prop={'size': 15})
ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='both', labelsize=14)
fig.savefig(dir_plot + f'seidr_strehl_ratio_{wf_type}_contig_tnn_vs_raw.pdf', dpi=150)
plt.show()

#%%#########################################################################
### Histogram: Strehl Ratio + Power Ratio side by side, TNN vs Raw Contig ###

fig, axes = plt.subplots(1, 2, 
                         figsize=(10, 5), 
                         sharey=True, 
                         tight_layout=True)

_subplot_data = [
    (axes[0], strehl_ratios, 'Strehl Ratio'),
    (axes[1], core0_ratios,  'Ratio of power in mode-selective core'),
]

for ax, ratios, xlabel in _subplot_data:
    for (key, label), color in zip(_pairs, _pair_colors):
        data = ratios[key]
        weights = np.ones_like(data) / len(data)
        ax.hist(data, bins=100, histtype='stepfilled', alpha=0.4,
                color=color, label=label, weights=weights)
        ax.hist(data, bins=100, histtype='step', linewidth=1.8,
                color=color, weights=weights)
        ax.axvline(data.mean(), color=color, linestyle='--', 
                   linewidth=2, alpha=0.8)
    ax.plot([], [], color='k', linestyle='--', 
            linewidth=2, alpha=0.8, label='Mean')
    ax.set_xlabel(xlabel, fontsize=15)
    ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', labelsize=12)

axes[1].legend(prop={'size': 13})
axes[0].set_ylabel('PDF', fontsize=15)
axes[0].set_xlim([-0.01, 1.01])
axes[1].set_xlim([-0.01, 1.01])
fig.savefig(dir_plot + f'seidr_strehl_power_ratio_{wf_type}_contig_tnn_vs_raw.svg', dpi=150)
plt.show()


#%%#########################################################################
### Grouped bar chart: mean core-0 ratio by dataset ###

_bar_groups = {
    'Random': [_k_rand,    _k_cnn_rand,    _k_tnn_rand],
    'Contig': [_k_contig,  _k_cnn_contig,  _k_tnn_contig],
}
_bar_labels = ['Uncorrected', 'CNN', 'TNN']
_bar_colors = ['#969696', '#4393C3', '#D6604D']

n_models  = len(_bar_labels)
n_groups  = len(_bar_groups)
_bar_x    = np.arange(n_groups)
width     = 0.18   # bar width
spacing   = 0.24   # centre-to-centre distance (> width → gap between bars)
offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * spacing

fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)

for i, (label, color) in enumerate(zip(_bar_labels, _bar_colors)):
    means = [core0_ratios[keys[i]].mean() for keys in _bar_groups.values()]
    ax.bar(_bar_x + offsets[i], means, width,
           color=color, alpha=0.85, label=label)

# individual bar labels: True / CNN / TNN repeated per group
_tick_pos    = [_bar_x[g] + offsets[i]
                for g in range(n_groups) for i in range(n_models)]
_tick_labels = ['Uncorrected', 'CNN', 'TNN'] * n_groups
ax.set_xticks(_tick_pos)
ax.set_xticklabels(_tick_labels, fontsize=13)


ax.set_ylim([0, 1])
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', length=0)
ax.set_ylabel('', fontsize=13)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

fig.savefig(dir_plot + f'seidr_ms_core_ratio_{wf_type}_bar.svg', dpi=150,
            bbox_inches='tight')
plt.show()


#%%#########################################################################
### Grouped bar chart: mean Strehl ratio by dataset ###

_bar_groups = {
    'Random': [_k_rand,    _k_cnn_rand,    _k_tnn_rand],
    'Contig': [_k_contig,  _k_cnn_contig,  _k_tnn_contig],
}
_bar_labels = ['Uncorrected', 'CNN', 'TNN']
_bar_colors = ['#969696', '#4393C3', '#D6604D']

n_models  = len(_bar_labels)
n_groups  = len(_bar_groups)
_bar_x    = np.arange(n_groups)
width     = 0.18   # bar width
spacing   = 0.24   # centre-to-centre distance (> width → gap between bars)
offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * spacing


fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)

for i, (label, color) in enumerate(zip(_bar_labels, _bar_colors)):
    means = [strehl_ratios[keys[i]].mean() for keys in _bar_groups.values()]
    ax.bar(_bar_x + offsets[i], means, width,
           color=color, alpha=0.85, label=label)

# individual bar labels: True / CNN / TNN repeated per group
_tick_pos    = [_bar_x[g] + offsets[i]
                for g in range(n_groups) for i in range(n_models)]
_tick_labels = ['Uncorrected', 'CNN', 'TNN'] * n_groups
ax.set_xticks(_tick_pos)
ax.set_xticklabels(_tick_labels, fontsize=13)

ax.set_ylim([0, 1.05])
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', length=0)
ax.set_ylabel('', fontsize=13)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

fig.savefig(dir_plot + f'seidr_strehl_ratio_{wf_type}_bar.svg', dpi=150,
            bbox_inches='tight')
plt.show()


#%%#########################################################################
### Grouped bar chart: mean residual wavefront RMS by dataset ###

_bar_groups = {
    'Random': [_k_rand,    _k_cnn_rand,    _k_tnn_rand],
    'Contig': [_k_contig,  _k_cnn_contig,  _k_tnn_contig],
}
_bar_labels = ['Uncorrected', 'CNN', 'TNN']
_bar_colors = ['#969696', '#D6604D', '#4393C3']

n_models  = len(_bar_labels)
n_groups  = len(_bar_groups)
_bar_x    = np.arange(n_groups)
width     = 0.18   # bar width
spacing   = 0.24   # centre-to-centre distance (> width → gap between bars)
offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * spacing


fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)

for i, (label, color) in enumerate(zip(_bar_labels, _bar_colors)):
    means = [wf_rms_residuals[keys[i]].mean() for keys in _bar_groups.values()]
    ax.bar(_bar_x + offsets[i], means, width,
           color=color, alpha=0.85, label=label)

# individual bar labels: True / CNN / TNN repeated per group
_tick_pos    = [_bar_x[g] + offsets[i]
                for g in range(n_groups) for i in range(n_models)]
_tick_labels = ['Uncorrected', 'CNN', 'TNN'] * n_groups
ax.set_xticks(_tick_pos)
ax.set_xticklabels(_tick_labels, fontsize=15)
# ax.set_yscale('log')
ax.set_ylabel('Mean Tip/Tilt RMS Error [rad]', fontsize=15)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', length=0)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

# group titles above each cluster of bars
_group_titles = ['Random', 'Temporal']
for g, title in enumerate(_group_titles):
    ax.text(_bar_x[g], 1.05, title, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=16,
            clip_on=False)

fig.savefig(dir_plot + f'seidr_wf_rms_{wf_type}_bar.svg', dpi=150,
            bbox_inches='tight')
plt.show()


#%%###########################################################################
### Load True Preds ###

_dir_preds = '/import/roci1/nlon0790/Results/proteus/outputs/'

_preds_files = {
    'tnn_kol_contig':     _dir_preds + 'seidr_tnn_seq_plcin_wfout_kol_contig_npl6_20260609-1555_preds.npz',
    'cnn_kol_contig':     _dir_preds + 'seidr_cnn_plcin_wfout_kol_contig_npl6_20260609-1826_preds.npz',
    'tnn_baldr_contig':   _dir_preds + 'seidr_tnn_seq_plcin_wfout_baldr_contig_npl6_20260615-1640_preds.npz',
}

# Keys available: pred_wf_array, true_wf_array, residual_wf_array (all shape (N,64,64))
#                 X_test, predictions_wf, y_test_wf, pupil_mask, normfacts_PL/WF, history_loss
_preds_keys = ['pred_wf_array', 'true_wf_array', 'residual_wf_array', 'pupil_mask']

preds = {}
for label, fpath in _preds_files.items():
    print(f"Loading preds: {label} ...")
    d = np.load(fpath, allow_pickle=True)
    preds[label] = {k: d[k] for k in _preds_keys}
    print(f"  pred_wf_array:     {preds[label]['pred_wf_array'].shape}")
    print(f"  true_wf_array:     {preds[label]['true_wf_array'].shape}")
    print(f"  residual_wf_array: {preds[label]['residual_wf_array'].shape}")

#%%#########################################################################
### Wavefront grid: True / TNN pred / CNN pred (from preds, 4 random frames) ###

rng    = np.random.default_rng()
n_cols = 4
# TNN[i] == CNN[i + sequential_length]: pick from TNN range, offset CNN indices
n_tnn    = preds[_k_tnn_contig]['pred_wf_array'].shape[0]

# tnn_idxs = rng.integers(0, n_tnn, size=n_cols)
# cnn_idxs = tnn_idxs + sequential_length

tnn_idxs = [13012, 4586, 13466, 14348]
cnn_idxs = [13062, 4636, 13516, 14398]

print(f"Selected TNN indices: {tnn_idxs}")
print(f"Corresponding CNN indices: {cnn_idxs}")

true_wfs = preds[_k_tnn_contig]['true_wf_array'][tnn_idxs]
tnn_pred = preds[_k_tnn_contig]['pred_wf_array'][tnn_idxs]
cnn_pred = preds[_k_cnn_contig]['pred_wf_array'][cnn_idxs]

# Selected TNN indices: [13012 337 8474 4586 13466 6454
# Corresponding CNN indices: [13062 387 8524 4636 13516 6504

rows       = [true_wfs, tnn_pred, cnn_pred]
row_labels = ['True WF', 'TNN WF Est.', 'CNN WF Est.']
cmap = 'twilight'
vmin, vmax = -np.pi, np.pi
cbar_ticks      = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
cbar_ticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']

fig, axes = plt.subplots(
    3, n_cols + 1,
    figsize=(12, 7),
    gridspec_kw={'width_ratios': [1]*n_cols + [0.06],
                 'hspace': 0.04, 'wspace': 0.04},
)

for row_idx, (wfs, rlabel) in enumerate(zip(rows, row_labels)):
    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]
        im = ax.imshow(wfs[col_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(rlabel, fontsize=15)
        # if row_idx == 0:
        #     ax.set_title(f'Frame {tnn_idxs[col_idx]}', fontsize=11)
    cbar_ax = axes[row_idx, n_cols]
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_ticks(cbar_ticks)
    cb.set_ticklabels(cbar_ticklabels, fontsize=10)

fig.savefig(dir_plot + f'seidr_wf_grid_true_tnn_cnn_{wf_type}_contig_preds.svg', dpi=150,
            bbox_inches='tight')
plt.show()


#%%#########################################################################
### Wavefront grid: True / TNN pred / CNN pred (from preds, 4 random frames) ###
## Paper version ##

rng    = np.random.default_rng()
n_cols = 4
# TNN[i] == CNN[i + sequential_length]: pick from TNN range, offset CNN indices
n_tnn    = preds[_k_tnn_contig]['pred_wf_array'].shape[0]

# tnn_idxs = rng.integers(0, n_tnn, size=n_cols)
# cnn_idxs = tnn_idxs + sequential_length

tnn_idxs = [13012, 4586, 13466, 14348]
cnn_idxs = [13062, 4636, 13516, 14398]

print(f"Selected TNN indices: {tnn_idxs}")
print(f"Corresponding CNN indices: {cnn_idxs}")

true_wfs = preds[_k_tnn_contig]['true_wf_array'][tnn_idxs]
tnn_pred = preds[_k_tnn_contig]['pred_wf_array'][tnn_idxs]
cnn_pred = preds[_k_cnn_contig]['pred_wf_array'][cnn_idxs]

# Selected TNN indices: [13012 337 8474 4586 13466 6454
# Corresponding CNN indices: [13062 387 8524 4636 13516 6504

rows       = [true_wfs, tnn_pred, cnn_pred]
row_labels = ['True WF', 'TNN WF Est.', 'CNN WF Est.']
cmap = 'twilight'
vmin, vmax = -np.pi, np.pi
cbar_ticks      = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
cbar_ticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']

fig, axes = plt.subplots(
    3, n_cols + 1,
    figsize=(12, 7),
    gridspec_kw={'width_ratios': [1]*n_cols + [0.06],
                 'hspace': 0.04, 'wspace': 0.04},
)

for row_idx, (wfs, rlabel) in enumerate(zip(rows, row_labels)):
    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]
        im = ax.imshow(wfs[col_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(rlabel, fontsize=15)
        # if row_idx == 0:
        #     ax.set_title(f'Frame {tnn_idxs[col_idx]}', fontsize=11)
    cbar_ax = axes[row_idx, n_cols]
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_ticks(cbar_ticks)
    cb.set_ticklabels(cbar_ticklabels, fontsize=10)
    if row_idx == 0:
        cbar_ax.set_title('Phase [rad]', fontsize=10)

fig.savefig(dir_plot + f'seidr_wf_grid_true_tnn_cnn_{wf_type}_contig_preds.svg', dpi=150,
            bbox_inches='tight')
plt.show()
# %%
