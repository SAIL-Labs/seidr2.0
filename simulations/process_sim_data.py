#%%###########################################################################
'''
This is the preprocessing script for simulated data for the transformer model
designed for Seidr. The script loads the raw data, normalises it, and splits
it into training and testing sets. The processed data is then saved in a format
suitable for training the transformer model using the style adopted by PL_NN.
'''

#%%###########################################################################
### Import Libraries and Modules ###
import IPython; _ip = IPython.get_ipython()
if _ip:
    _ip.run_line_magic('load_ext', 'autoreload')
    _ip.run_line_magic('autoreload', '2')
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from seidr.temp_preprocess_sim_data_funcs import (
    load_compact_sim_data, make_pupil_mask, wf_to_vector,
    normalize_data, split_train_test, vector_to_wf,
    denormalize_data
)
from seidr.seidr_functions_misc import plot_wf_psf_lp_pl


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

#%%###########################################################################
### Parameters ###

datadir  = '/import/roci1/nlon0790/Results/psf_prop/'  # directory containing the raw simulation data
sims_fname = 'hms-pl6c_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_20260526-2331.npz' #seidr_wf_psf_lp_dataset_test_20260515-1430.npz'
tnndir   = '/import/roci1/nlon0790/Results/proteus/outputs/'
preds_fname = 'seidr_tnn_plcin_wfout_npl6_20260601-1220_preds.npz' #seidr_tnn_plcin_wfout_npl6_20260601-1220_preds.npz
outname  = 'seidr_postprocessed_hms-pl6c_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_20260526-2331' #'preprocessed_seidr_data.npz'
figdir  = Path(__file__).parents[1] / "figures"

wf_type = 'kolmogorov'  # 'zernike' or 'kolmogorov'

test_split  = 0.2
stat_frms   = 10000   # frames used to compute normalisation statistics


#%%###########################################################################
### Load Data ###

raw = load_compact_sim_data(datadir, sims_fname, wf_type=wf_type)

pl_powers  = raw['pl_powers'].astype('float32')   # PL port powers -> inputs
pupil_wf   = raw['pupil_wf'].astype('float32')     # wavefront -> targets
psf_fields = raw['psf_fields']                     # complex PSF fields
lp_powers  = raw['lp_powers'].astype('float32')    # LP modal powers

print(f"pl_powers shape:  {pl_powers.shape}")
print(f"pupil_wf shape:   {pupil_wf.shape}")
print(f"psf_fields shape: {psf_fields.shape}")
print(f"lp_powers shape:  {lp_powers.shape}")

#%%###########################################################################
### Plot Example Row ###

idx_example = np.random.randint(0, len(pupil_wf))
idx_example = 27308
print(f"Example index: {idx_example}")
plot_wf_psf_lp_pl(raw, idx=idx_example, save_plot=False, 
                  fname_plot=figdir / f"seidr_wf_psf_lp_pl_example_{idx_example}.pdf")
plt.show()


#%%###########################################################################
### Load TNN Predictions ###

# tnn_preds = np.load(os.path.join(tnndir, preds_fname))

npf_npl6 = np.load(tnndir + preds_fname, allow_pickle=True)
predictions_wf_npl6 = npf_npl6['predictions_wf']   # (N, H*W)
y_test_wf_npl6      = npf_npl6['y_test_wf']        # (N, H*W)
X_test_npl6         = npf_npl6['X_test']           # (N, n_pl_ports)
wf_shape_npl6       = tuple(npf_npl6['wf_shape'])  # (H, W)
pupil_mask_npl6     = npf_npl6['pupil_mask']       # (H, W) bool
normfacts_PL_npl6   = npf_npl6['normfacts_PL']    # [mean, 0, std]
print(f"npl6: {predictions_wf_npl6.shape[0]} samples, {X_test_npl6.shape[1]} ports")
print(f"  WF shape: {predictions_wf_npl6.shape}")


#%%############################################################################



#%%############################################################################
### OLD PROCESSING CODE BELOW - IGNORE ###
################################################################################
#%%###########################################################################
### Preprocess for NN Training ###

# Store original WF spatial shape before flattening
wf_shape = pupil_wf.shape[1:]
mask = make_pupil_mask(wf_shape)

# Flatten 2D wavefronts to 1D vectors, keeping only in-pupil pixels
wf_vec = wf_to_vector(pupil_wf, mask=mask)
wf_vec_orig = wf_vec.copy()
pl_powers_orig = pl_powers.copy()

# Normalise
pl_norm, wf_norm, normfacts = normalize_data(pl_powers, wf_vec,
                                             stat_frms=stat_frms)

print(f"PL norm: mean={normfacts['PL'][0]:.4f}, std={normfacts['PL'][2]:.4f}")
print(f"WF norm: mean={normfacts['WF'][0]:.4f}, std={normfacts['WF'][2]:.4f}")

#%%###########################################################################
### Train / Test Split ###

split = split_train_test(pl_norm, wf_norm, test_split=test_split)

print(f"Train samples: {split['X_train'].shape[0]}")
print(f"Test samples:  {split['X_test'].shape[0]}")

#%%###########################################################################
### Save ###

os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, outname)

np.savez(
    outpath,
    X_train=split['X_train'],
    y_train_wf=split['y_train_wf'],
    X_test=split['X_test'],
    y_test_wf=split['y_test_wf'],
    wf_shape=np.array(wf_shape),
    pupil_mask=mask,
    normfacts_PL=normfacts['PL'],
    normfacts_WF=normfacts['WF'],
)

print(f"Saved preprocessed data to {outpath}")

#%%

pupil_wf_reconstructed = vector_to_wf(wf_vec_orig, wf_shape, mask=mask)


#%%

num = np.random.randint(0, len(pupil_wf))

print('Original')
plt.figure(figsize=(4, 4))
plt.imshow(pupil_wf[num], cmap="twilight", 
           vmin=-np.pi, vmax=np.pi)
plt.colorbar()
plt.show()

print('Reconstructed')
plt.figure(figsize=(4, 4))
plt.imshow(pupil_wf_reconstructed[num], cmap="twilight", 
           vmin=-np.pi, vmax=np.pi)
plt.colorbar()
plt.show()

print('Difference')
plt.figure(figsize=(4, 4))
plt.imshow(pupil_wf[num] - pupil_wf_reconstructed[num], cmap="twilight", 
           vmin=-np.pi, vmax=np.pi)
plt.colorbar()
plt.show()

#%%#########################################################################
## Check denormalization

pl_denorm, wf_denorm = denormalize_data(pl_norm, wf_norm, normfacts)

#%%

print('Difference Original vs Denormalized PL powers')

print('diff = denorm - original')
plt.figure(figsize=(4, 4))
plt.bar(np.arange(pl_powers_orig.shape[1]), pl_denorm[num] - pl_powers_orig[num])
plt.xticks(np.arange(pl_powers.shape[1]))
plt.title("Difference in PL core powers (denorm - original)")
plt.tight_layout()
plt.show()

#%%

plt.figure(figsize=(4, 4))
n_ports = pl_powers.shape[1]
plt.bar(np.arange(n_ports), pl_powers[num])
plt.xticks(np.arange(n_ports))
plt.title("PL core powers")
plt.tight_layout()
plt.show()

#%%#########################################################################
## Check WF denormalization

wf_denorm_2d = vector_to_wf(wf_denorm, wf_shape, mask=mask)

print('WF denorm max abs error:', np.max(np.abs(wf_denorm_2d - pupil_wf)))

plt.figure(figsize=(4, 4))
plt.imshow(wf_denorm_2d[num] - pupil_wf[num], cmap='twilight')
plt.title("WF difference (denorm - original)")
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
 # field = results["fields"][idx_rand]
    # wf = results["pupil_wf"][idx_rand]
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(wf, cmap="twilight", 
    #            vmin=-np.pi, vmax=np.pi)
    # plt.title("Wavefront")
    # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.imshow(np.abs(field)**2)
    # plt.title("PSF intensity at HMSPL input")
    # plt.colorbar()

    # plt.subplot(1, 3, 3)
    # plt.bar(np.arange(results["nmodes"]), results["lp_powers"][idx])
    # plt.xticks(np.arange(results["nmodes"]), results["modelabels"], rotation=90)
    # plt.title("LP modal powers")
    # plt.tight_layout()
    # plt.show()



    # ## Plot Zernike coefficients
    # plt.figure(1)
    # plt.plot(results["zernikes"][:100,:], '-o',markersize=2)
    # plt.xlabel('Simulation Step')
    # plt.ylabel('Zernike Coefficient Value')
    # # plt.ylim([-1, 1])
    # plt.grid(':', linewidth=0.5, alpha=0.5)
    # plt.legend(['%s' % (zernike_mode_labels[k]) for k in range(1, n_zernikes+1)],
    #         loc='best', 
    #         fontsize=8,
    #         ncol=4)
    # plt.show()