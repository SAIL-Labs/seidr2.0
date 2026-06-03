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

from seidr.source2pl import source2pl_zernike_temporal, \
    source2pl_kolmogorov_temporal
from seidr.seidr_functions_misc import plot_wf_psf_zernike_lp, \
    make_wf_psf_zernike_video_square, make_wf_psf_lp_pl_zernike_video_row, \
    plot_wf_psf_lp_pl, make_wf_psf_lp_pl_video_row

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
f_zern_pref = "_zern"


## kolmogorov wavefronts
f_wf_path = '/import/roci2/sail/data/PL_IRTestbed/2024_files-combined/'
f_wf_name = 'slmcube_202400708_seeing_0.4-10-scl1_contig-flatn1000_10K_01_files-combined.npz'
f_kol_pref = '_kol'

## predictions from TNN
tnndir   = '/import/roci1/nlon0790/Results/proteus/outputs/'
preds_fname = 'seidr_tnn_plcin_wfout_npl6_20260602-1523_preds.npz'
f_pred_pref = '_preds'

#%%########################################################################
### Set HMSPL and Simulation Parameters ##

print("Setting HMSPL and simulation parameters...")

## Simulation parameters
n_sims = 100000 #None # number of simulations to run

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

wf_type = "zernike" # "zernike" or "kolmogorov" or "tnn_predictions"

if wf_type == "zernike":
    print("Using Zernike wavefronts with random coefficients.")
    f_data = dir + f_pl_name + f_zern_pref + "_wf_psf_lp_dataset_" + outname_datetime + ".npz"


    f_plot = dir_plot + "seidr_wf_psf_lp_zernike_example.pdf" #+ outname_datetime + ".pdf"
    f_plot_row = dir_plot + "seidr_wf_psf_lp_row_zernike_example.pdf" #+ outname_datetime + "_row.pdf"
    f_video = dir_plot + "seidr_wf_psf_lp_zernike_evolution.gif" #+ outname_datetime + ".gif"

    n_zernikes = 3  # number of Zernike modes to include in the random aberrations

    # Vary RMS per zernike mode, with a linear drop-off from start to end mode
    # if max_rms_per_mode == min_rms_per_mode, then only defining tip/tilt
    if n_zernikes == 3:
        print("  Only including tip/tilt and defocus (Zernike modes 2, 3, and 4).")
        max_rms_per_mode = 3e-8 #7e-8 #0.4 [m]
        min_rms_per_mode = 3e-8 #1e-8 #0.05 [m]
    else:
        print(f"  Including Zernike modes 2 to {n_zernikes+1}.")
        max_rms_per_mode = 7e-8 #0.4 [m]
        min_rms_per_mode = 1e-8 #0.05 [m]

    ## Define Zernike coefficient wavefront error RMS values using Gaussian kernel smoothing
    smooth_amt = 7 # Gaussian kernel samples / time steps
        
    # ## Wavefront error RMS
    # tiptilt_rms = 3e-8 # m None
    # ho_rms = 5e-8 # m

elif wf_type == "kolmogorov":
    print("Using Kolmogorov wavefronts with random temporal evolution.")
    f_data = dir + f_pl_name + f_kol_pref + "_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_" + outname_datetime + ".npz"


    f_plot = dir_plot + "seidr_wf_psf_lp_example.pdf" #+ outname_datetime + ".pdf"
    f_plot_row = dir_plot + "seidr_wf_psf_lp_row_example.pdf" #+ outname_datetime + "_row.pdf"
    f_video = dir_plot + "seidr_wf_psf_lp_evolution.gif" #+ outname_datetime + ".gif"

    wf_data = np.load(f_wf_path + f_wf_name)
    phase_screens = wf_data['all_pupphase'][1:]  # (n_sims, 64, 64) [rad]
    
    if n_sims is None:
        n_sims = phase_screens.shape[0]
        print(f"Setting n_sims to {n_sims} based on the number of available phase screens in the dataset.")


elif wf_type == "tnn_predictions":
    print("Using TNN predictions for wavefronts.")
    f_data = dir + f_pl_name + f_pred_pref + "_wf_psf_lp_dataset_slmcube_202400708_seeing_0.4-10-scl1_contig_" + outname_datetime + ".npz"


    f_plot = dir_plot + "seidr_wf_psf_lp_pred_example.pdf" #+ outname_datetime + ".pdf"
    f_plot_row = dir_plot + "seidr_wf_psf_lp_row_pred_example.pdf" #+ outname_datetime + "_row.pdf"
    f_video = dir_plot + "seidr_wf_psf_lp_pred_evolution.gif" #+ outname_datetime + ".gif"

    preds_data = np.load(tnndir + preds_fname)
    phase_screens = preds_data['residual_wf_array']  # (n_sims, 64, 64) [rad]

    if n_sims is None:
        n_sims = phase_screens.shape[0]
        print(f"Setting n_sims to {n_sims} based on the number of available phase screens in the dataset.")

else:
    raise ValueError(f"Invalid wf_type: {wf_type}. Must be 'zernike' or 'kolmogorov'.")


#%%
if wf_type == "kolmogorov" or wf_type == "tnn_predictions":
    num = np.random.randint(0, phase_screens.shape[0]-3)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    # vmin, vmax = phase_screens[:3].min(), phase_screens[:3].max()
    for k, ax in enumerate(axes):
        im = ax.imshow(phase_screens[num+k], 
                       cmap="twilight", 
                       vmin=-np.pi, vmax=np.pi, 
                       origin="lower")
        ax.set_title(f"Phase Screen {num+k}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, label="phase [rad]", fraction=0.02, pad=0.04)
    plt.show()



#%%#########################################################################
save_data = True # whether to save the generated dataset to disk
plot_example = True # whether to plot one example of the generated PSF, wavefront, and LP powers
save_video = False # whether to save the video to disk

#%%########################################################################
if __name__ == "__main__":

    print("Running simulations...")
    
    if wf_type == "zernike":
        print(f"Generating random Zernike coefficients with max RMS {max_rms_per_mode} m and min RMS {min_rms_per_mode} m, smoothed with a Gaussian kernel of {smooth_amt} samples.")
        lf, results = source2pl_zernike_temporal(
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
        lf, results = source2pl_kolmogorov_temporal(
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

    elif wf_type == "tnn_predictions":
        print(f"Using TNN predictions loaded from {tnndir + preds_fname} with shape {phase_screens.shape}.")
        lf, results = source2pl_kolmogorov_temporal(
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
        if wf_type == "zernike":
            plot_wf_psf_zernike_lp(results, idx=idx_rand, save_plot=True,
                                fname_plot=f_plot)
        elif wf_type == "kolmogorov" or wf_type == "tnn_predictions":
            plot_wf_psf_lp_pl(results, idx=idx_rand, save_plot=True,
                                fname_plot=f_plot_row)

    if save_video:
        if wf_type == "zernike":
            # make_wf_psf_zernike_video_square(results, outname=f_video,
            #                   save_video=save_video, fps=30, dpi=150)
            
            make_wf_psf_lp_pl_zernike_video_row(results, 
                                        outname=f_video.replace(".gif", "_row.gif"),
                                        save_video=save_video, fps=30, dpi=150)
        
        elif wf_type == "kolmogorov":
            make_wf_psf_lp_pl_video_row(results, 
                                        outname=f_video.replace(".gif", "_row.gif"),
                                        save_video=save_video, fps=30, dpi=150)


    ########################################################################
    ### Save Dataset ###
    if save_data:
        print("Saving dataset to disk...")
        if wf_type == "zernike":
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

        elif wf_type == "kolmogorov" or wf_type == "tnn_predictions":
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