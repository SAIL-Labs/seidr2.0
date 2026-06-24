#%%###########################################################################
"""
Generate PSFs for a lantern fiber, using the 
lanternfiber and zernikePSF classes 
"""

#%%########################################################################
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import datetime

from seidr.source2pl import source2pl_zernike, \
    source2pl_kolmogorov
from seidr.seidr_functions_misc import plot_wf_psf_zernike_lp, \
    make_wf_psf_zernike_video_square, make_wf_psf_lp_pl_zernike_video_row, \
    plot_wf_psf_lp_pl, make_wf_psf_lp_pl_video_row, get_filenames

#%%########################################################################
### Define Simulation Type ###

# whether to run temporal simulations with evolving wavefronts, or just static sims with one wavefront per simulation
temporal = True

wf_type = "nn_predictions" # "zernike" or "kolmogorov" or "nn_predictions" or "tiptilt" or "baldr"

if wf_type == "nn_predictions":
    pred_type = "baldr_contig" # only used when wf_type == "nn_predictions"
                                    # "kolmogorov_contig", "kolmogorov_rand", "tiptilt_contig", "tiptilt_rand", "baldr_contig"

    nn_type = "cnn" # only used when wf_type == "nn_predictions": "tnn" or "cnn"

else:
    pred_type = None
    nn_type = None


#%%########################################################################
### Filenames ###

print("Loading datasets and transfer matrix from disk...")

outname_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")

dir = "/import/roci1/nlon0790/Results/psf_prop/"
dir_plot = "/suphys/nlon0790/Documents/python_code/seidr2.0/figures/"

# f_data = dir + "seidr_wf_psf_lp_dataset_test_" + outname_datetime + ".npz"

## transfer matrix 
f_pl_name = "hms-pl6c"
f_pl_path = "/import/roci1/nlon0790/Results/hms-pl6/cores/"

## zernike wavefronts
f_zern_pref = "_zernike" 

## tiptilt wavefronts
f_tt_pref = "_tiptilt" 

## kolmogorov wavefronts
f_wf_path = '/import/roci2/sail/data/PL_IRTestbed/2024_files-combined/'
f_kol_pref = '_kol'
if temporal:
    f_wf_name = 'slmcube_202400708_seeing_0.4-10-scl1_contig-flatn1000_10K_01_files-combined.npz'
else:
    f_wf_name = 'slmcube_202400708_seeing_0.4-10-scl1_rand-flatn2_10K_01_files-combined.npz'

## baldr wavefronts
f_baldr_path = '/import/roci1/nlon0790/Results/phase_screens/baldr/post_baldr_residual_cube_parts0000_0009_noresets.fits'


## predictions from TNN
tnndir   = '/import/roci1/nlon0790/Results/proteus/outputs/'
f_pred_pref = '_preds'

#%%########################################################################
### Set HMSPL and Simulation Parameters ##

print("Setting HMSPL and simulation parameters...")

## Simulation parameters

# number of sims
if wf_type == "zernike" or wf_type == "tiptilt":
    n_sims = 100000 # number of simulations to run
else:
    n_sims = None # use all available phase screens in the dataset
                  # (set below once the data file is loaded)

wavel = 1.55 # wavelenth [um]

wf_npixels = 64  # number of pixels across the pupil plane for SeidrSim optics
psf_npixels = 128  # number of pixels across the PSF plane for SeidrSim optics

focal_length = 20000 # focal length of the optics in um
pupil_diameter = 4500 #256*17  # diameter of the pupil in um
f_number = focal_length / pupil_diameter  # f-number of the optics


## HMSPL parameters ##

## Transfer Matrix Parameters
n_cores = 6 # number of cores in the PL
wavel = 1.55 # wavelenth [um]
r_ms = 8.2 # radius of mode selective core [um]
ds = 0.25 # x-y step size for LB sims [um]
dz = 2 # z-step size for LB sims [um]
rv = 1 # reference value for LB sims
z_len = 50000 # length of PL [um]
taper_ratio = 20 # taper ratio

r_clad_out = 155 # [um]

xywidth = 2 * r_clad_out + 10 # width of the simulation window [um]

## Input Radii ##
r_core_mm = r_clad_out/taper_ratio # cladding radius [um]
d_core_mm = 2 * r_core_mm  # MMF end core diameter [um]

# numerical window / padding scaling factor
max_r = 3 # maximum radius to calculate fiber modes, in units of core radius

## Refractive Indices ##
n_core_mm = 1.444
n_clad_mm = 1.435


#%%########################################################################
## Set Phase Screen Type ##


f_data, f_plot, f_plot_row, f_video = get_filenames(
    wf_type, pred_type, nn_type, temporal,
    dir, dir_plot, f_pl_name,
    f_zern_pref, f_tt_pref, f_kol_pref, f_pred_pref,
    outname_datetime,
)

if wf_type == "zernike":
    print("Using Zernike wavefronts with random coefficients.")
    n_zernikes = 30  # number of Zernike modes to include in the random aberrations

    print(f"  Including Zernike modes 2 to {n_zernikes+1}.")
    max_rms_per_mode = 7e-8 #0.4 [m]
    min_rms_per_mode = 1e-8 #0.05 [m]
    smooth_amt = 7 # Gaussian kernel samples / time steps

elif wf_type == "tiptilt":
    print("Using tiptilt wavefronts with random coefficients.")
    n_zernikes = 3  # number of Zernike modes to include in the random aberrations

    print("  Only including tip/tilt (Zernike modes 2, 3).")
    max_rms_per_mode = 1e-7 #7e-8 #0.4 [m]
    min_rms_per_mode = 1e-7 #1e-8 #0.05 [m]
    smooth_amt = 7 # Gaussian kernel samples / time steps

elif wf_type == "kolmogorov":
    print("Using Kolmogorov wavefronts with random temporal evolution.")
    wf_data = np.load(f_wf_path + f_wf_name)
    if temporal:
        phase_screens = wf_data['all_pupphase'][1:]             # (n_sims, 64, 64) [rad]
    else:
        # rand file has flat reference frames at every even index (all identical);
        # odd indices are the actual varied phase screens
        phase_screens = wf_data['all_pupphase'][1::2][:100000]  # (100000, 64, 64) [rad]
    if n_sims is None:
        n_sims = phase_screens.shape[0]
        print(f"Setting n_sims to {n_sims} based on the number of available phase screens in the dataset.")

elif wf_type == "baldr":
    from astropy.io import fits
    print(f"Using Baldr residual OPD screens from {f_baldr_path}.")
    with fits.open(f_baldr_path) as hdul:
        opd_nm = hdul[0].data.astype(np.float64)   # (n_frames, 68, 68), nm OPD
    # Replace NaN (non-pupil pixels from pupil mask) with zero phase
    opd_nm = np.nan_to_num(opd_nm, nan=0.0)
    # Convert nm OPD -> radians at science wavelength
    phase_screens = opd_nm * 1e-9 * (2 * np.pi / (wavel * 1e-6))  # (n_frames, 68, 68) [rad]
    if n_sims is None:
        n_sims = phase_screens.shape[0]
        print(f"Setting n_sims to {n_sims} based on the number of available Baldr frames.")


elif wf_type == "nn_predictions":
    print(f"Using NN predictions for wavefronts (pred_type={pred_type}).")
    _preds_fnames = {
        "tnn": {
            "kolmogorov_contig": 'seidr_tnn_seq_plcin_wfout_kol_contig_npl6_20260609-1555_preds.npz',
            "kolmogorov_rand":       'seidr_tnn_plcin_wfout_kol_rand_npl6_20260610-1005_preds.npz',                          # placeholder
            "tiptilt_contig":    'seidr_tnn_seq_plcin_wfout_tiptilt_contig_npl6_20260609-2003_preds.npz',                       # placeholder
            "tiptilt_rand":          'seidr_tnn_plcin_wfout_tiptilt_rand_npl6_20260609-2203_preds.npz',                  # placeholder
            "baldr_contig":          'seidr_tnn_seq_plcin_wfout_baldr_contig_npl6_20260615-1640_preds.npz',
        },
        "cnn": {
            "kolmogorov_contig":     'seidr_cnn_plcin_wfout_kol_contig_npl6_20260609-1826_preds.npz',                    # placeholder
            "kolmogorov_rand":       'seidr_cnn_plcin_wfout_kol_rand_npl6_20260610-1010_preds.npz',                      # placeholder
            "tiptilt_contig":        'seidr_cnn_plcin_wfout_tiptilt_contig_npl6_20260609-2030_preds.npz',                       # placeholder
            "tiptilt_rand":          'seidr_cnn_plcin_wfout_tiptilt_rand_npl6_20260609-2158_preds.npz',                  # placeholder
            "baldr_contig":          'seidr_cnn_plcin_wfout_baldr_contig_npl6_20260616-1418_preds.npz',                                                                                   # placeholder
        },
    }
    preds_fname = _preds_fnames[nn_type][pred_type]
    preds_data = np.load(tnndir + preds_fname)
    phase_screens = preds_data['residual_wf_array']  # (n_sims, 64, 64) [rad]
    if n_sims is None:
        n_sims = phase_screens.shape[0]
        print(f"Setting n_sims to {n_sims} based on the number of available phase screens in the dataset.")

#%%
# if wf_type == "kolmogorov" or wf_type == "nn_predictions":
#     num = np.random.randint(0, phase_screens.shape[0]-3)
#     fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
#     # vmin, vmax = phase_screens[:3].min(), phase_screens[:3].max()
#     for k, ax in enumerate(axes):
#         im = ax.imshow(phase_screens[num+k], 
#                        cmap="twilight", 
#                        vmin=-np.pi, vmax=np.pi, 
#                        origin="lower")
#         ax.set_title(f"Phase Screen {num+k}")
#         ax.set_xticks([])
#         ax.set_yticks([])

#     fig.colorbar(im, ax=axes, label="phase [rad]", fraction=0.02, pad=0.04)
#     plt.show()



#%%#########################################################################
save_data = True # whether to save the generated dataset to disk
plot_example = False # whether to plot one example of the generated PSF, wavefront, and LP powers
save_video = False # whether to save the video to disk

#%%########################################################################
if __name__ == "__main__":

    print("Running simulations...")
    
    if wf_type == "zernike" or wf_type == "tiptilt":
        if temporal:
            print(f"Generating temporally-evolving Zernike coefficients with max RMS {max_rms_per_mode} m and min RMS {min_rms_per_mode} m, smoothed with a Gaussian kernel of {smooth_amt} samples.")
        else:
            print(f"Generating independently random Zernike coefficients with max RMS {max_rms_per_mode} m and min RMS {min_rms_per_mode} m (no temporal correlation).")
        lf, results = source2pl_zernike(
            n_sims=n_sims,
            wavel=wavel,
            f_number=f_number,
            pupil_diameter=pupil_diameter,
            n_core=n_core_mm,
            n_cladding=n_clad_mm,
            core_diameter=d_core_mm,
            max_r=max_r,
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            n_zernikes=n_zernikes,
            max_rms_perterm=max_rms_per_mode,
            min_rms_perterm=min_rms_per_mode,
            temporal=temporal,
            smooth_amt=smooth_amt,
            f_pl_path=f_pl_path,
            f_pl_name=f_pl_name,
            r_core=r_ms,
            ds=ds,
            dz=dz,
            rv=rv, 
            xywidth=xywidth,
            z_len=z_len, 
            tr=taper_ratio,
        )

        print("zernike coeffs shape :", results["zernike_coeffs"].shape)

    elif wf_type == "kolmogorov":
        print(f"Using Kolmogorov wavefronts loaded from {f_wf_path + f_wf_name} with shape {phase_screens.shape}.")
        lf, results = source2pl_kolmogorov(
            phase_screens=phase_screens,
            wavel=wavel,
            f_number=f_number,
            pupil_diameter=pupil_diameter,
            n_core=n_core_mm,
            n_cladding=n_clad_mm,
            core_diameter=d_core_mm,
            max_r=max_r,
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            f_pl_path=f_pl_path,
            f_pl_name=f_pl_name,
            r_core=r_ms,
            ds=ds,
            dz=dz,
            rv=rv, 
            xywidth=xywidth,
            z_len=z_len, 
            tr=taper_ratio,
        )

    elif wf_type == "nn_predictions":
        print(f"Using NN predictions loaded from {tnndir + preds_fname} with shape {phase_screens.shape}.")
        lf, results = source2pl_kolmogorov(
            phase_screens=phase_screens,
            wavel=wavel,
            f_number=f_number,
            pupil_diameter=pupil_diameter,
            n_core=n_core_mm,
            n_cladding=n_clad_mm,
            core_diameter=d_core_mm,
            max_r=max_r,
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            f_pl_path=f_pl_path,
            f_pl_name=f_pl_name,
            r_core=r_ms,
            ds=ds,
            dz=dz,
            rv=rv,
            xywidth=xywidth,
            z_len=z_len,
            tr=taper_ratio,
        )

    elif wf_type == "baldr":
        print(f"Using Baldr OPD screens from {f_baldr_path} with shape {phase_screens.shape}.")
        lf, results = source2pl_kolmogorov(
            phase_screens=phase_screens,
            wavel=wavel,
            f_number=f_number,
            pupil_diameter=pupil_diameter,
            n_core=n_core_mm,
            n_cladding=n_clad_mm,
            core_diameter=d_core_mm,
            max_r=max_r,
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            f_pl_path=f_pl_path,
            f_pl_name=f_pl_name,
            r_core=r_ms,
            ds=ds,
            dz=dz,
            rv=rv,
            xywidth=xywidth,
            z_len=z_len,
            tr=taper_ratio,
        )



    print("powers shape :", results["lp_powers"].shape)
    print("coeffs shape :", results["lp_coeffs"].shape)
    print("fields shape :", results["psf_fields"].shape)
    print("pupil wfs shape :", results["pupil_wf"].shape)
    print("number of LP modes :", results["nmodes"])
    print("PL outputs shape :", results["pl_outputs"].shape)

    ##########################################################################
    ### Plotting ###

    ## Plot one example
    idx_rand = np.random.randint(0, n_sims) # pick a random simulation to plot

    if plot_example:
        if wf_type == "zernike" or wf_type == "tiptilt":
            plot_wf_psf_zernike_lp(results, idx=idx_rand, save_plot=True,
                                fname_plot=f_plot)
        elif wf_type == "kolmogorov" or wf_type == "nn_predictions" or wf_type == "baldr":
            plot_wf_psf_lp_pl(results, idx=idx_rand, save_plot=True,
                                fname_plot=f_plot)

    if save_video:
        if wf_type == "zernike" or wf_type == "tiptilt":
            # make_wf_psf_zernike_video_square(results, outname=f_video,
            #                   save_video=save_video, fps=30, dpi=150)
            
            make_wf_psf_lp_pl_zernike_video_row(results, 
                                        outname=f_video.replace(".gif", "_row.gif"),
                                        save_video=save_video, fps=30, dpi=150)
        
        elif wf_type == "kolmogorov" or wf_type == "baldr":
            make_wf_psf_lp_pl_video_row(results,
                                        outname=f_video.replace(".gif", "_row.gif"),
                                        save_video=save_video, fps=30, dpi=150)


    ########################################################################
    ### Save Dataset ###
    if save_data:
        print("Saving dataset to disk...")
        if wf_type == "zernike" or wf_type == "tiptilt":
            np.savez(
                f_data,
                total_coupling=results["total_coupling"],
                lp_powers=results["lp_powers"],
                lp_coeffs=results["lp_coeffs"],
                zernike_coeffs=results["zernike_coeffs"],
                modelabels=results["modelabels"],
                psf_fields=results["psf_fields"],
                pupil_wf=results["pupil_wf"],
                pl_outputs=results["pl_outputs"],
                pl_powers=results["pl_powers"],
            )

        elif wf_type == "kolmogorov" or wf_type == "nn_predictions" or wf_type == "baldr":
            np.savez(
                f_data,
                total_coupling=results["total_coupling"],
                lp_powers=results["lp_powers"],
                lp_coeffs=results["lp_coeffs"],
                modelabels=results["modelabels"],
                psf_fields=results["psf_fields"],
                pupil_wf=results["pupil_wf"],
                pl_outputs=results["pl_outputs"],
                pl_powers=results["pl_powers"],
            )

        print(f"Saved dataset to {f_data}")

#%%

# idx_example = np.random.randint(0, n_sims)
# plt.figure(figsize=(8, 4))
# plt.bar(range(n_cores), np.abs(results["pl_outputs"][idx_example, :])**2)
# plt.xlabel('Core Index')
# plt.ylabel('Output Intensity')
# plt.title('Propagation of PSF through PL - Example Simulation')
# plt.grid(':', linewidth=0.5, alpha=0.5)
# plt.show()


#%%

# idx_example = np.random.randint(0, n_sims)

# print(np.sum(np.abs(results["psf_fields"][idx_example, :])**2))
# print(np.sum(results["lp_powers"][idx_example, :]))
# print(np.sum(np.abs(results["pl_outputs"][idx_example, :])**2))

#%%
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# plt.imshow(np.abs(field)**2)
# plt.title("PSF intensity at HMSPL input")
# plt.colorbar()

# plt.subplot(1, 3, 2)
# plt.imshow(np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
# plt.title("PSF phase")
# plt.colorbar()

# plt.subplot(1, 3, 3)
# plt.bar(np.arange(results["nmodes"]), results["lp_powers"][idx])
# plt.xticks(np.arange(results["nmodes"]), results["modelabels"], rotation=90)
# plt.title("LP modal powers")
# plt.tight_layout()
# plt.show()

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