#%%
"""
Defines Functions for Use in End-to-End Seidr Simulations
"""

#%% Import Libraries and Modules

import os
import h5py
import jax
import jax.numpy as jnp
from jax import random
import h5py
from scipy import ndimage
import numpy as np


import dLux.utils as dlu
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from mpl_toolkits.axes_grid1 import make_axes_locatable

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

#%% Define Functions

##############################################################################
def correlated_noise(correlation_time, rms_amplitudes, sample_times, 
                     key=random.PRNGKey(0)):
    """
        Used to generate correlated aberrations
    """
    ## Sample the correlated aberrations
    
    del_ts = jnp.diff(sample_times)

    ## Draw the initial Gaussian sample
    key, subkey = random.split(key)
    noise = random.normal(subkey, (1, len(rms_amplitudes))) \
        * rms_amplitudes

    rs = jnp.exp(-del_ts / correlation_time)
    tilde_sigs = jnp.sqrt(1 - rs**2)[:, None] * rms_amplitudes[None, :]

    for i in range(1, len(sample_times)):

        key, subkey = random.split(key)
        noise = jnp.vstack(
            [
                noise,
                rs[i - 1] * noise[-1]
                + tilde_sigs[i - 1]
                * random.normal(subkey, (1, len(rms_amplitudes))),
            ]
        )

    return noise


##############################################################################
def electric_field(I, Phi):
    """
    Electric Field Reconstruction Function
    ----------------------------------------------------------------------
    This function calculates the electric field using beam intensity and 
    phase information.

    Arguments:
        I                      : Beam intensity profile
        Phi                    : Beam phase profile
        
    Returns:
        E                      : Beam electric field
    """ 
    
    # Electric field
    E = np.sqrt(I) * ( np.cos(Phi) + np.sin(Phi)*1j )

    return E

##############################################################################
def load_lb_transfer_matrix(f_path, f_pl_name, 
                            wl, r_ms, ds, 
                            dz, rv, xywidth, 
                            z_len, tr):
    """
        Used to load complex transfer matrices for the lantern fiber,
        as calculated using lightbea.
    """

    f_name = f_path + f_pl_name + '_C_lm_array_wl=' + str(wl) \
        + '_rms=' + str(r_ms) + '_ds=' + str(ds) + '_dz=' + str(dz) \
        + '_rv=' + str(rv) + '_xyw=' + str(xywidth) + '_zlen=' + str(z_len) \
        + '_tr=' + str(tr) + '.h5'

    C_lm_data = h5py.File(f_name, 'r')
    C_lm_array = C_lm_data['C_lm'][:]
    C_lm_data.close()

    return C_lm_array


##############################################################################
def make_smoothrand(nsteps, nvecs=1, smthamt=10., 
                    finalsd=1.):

    smthrand_all = np.zeros((nsteps, nvecs))

    for k in range(nvecs):
        noisevec = np.random.randn(nsteps)
        smthrand = ndimage.gaussian_filter1d(noisevec, smthamt)
        smthrand = smthrand / np.std(smthrand) * finalsd
        smthrand_all[:,k] = smthrand

    return smthrand_all


##############################################################################
def make_smoothrand_multi(nsteps, nvecs=1, smthamt=10., 
                          finalsds=1.):
    
    smthrand_all = np.zeros((nsteps, nvecs))

    for k in range(nvecs):
        noisevec = np.random.randn(nsteps)
        smthrand = ndimage.gaussian_filter1d(noisevec, smthamt)
        smthrand = smthrand / np.std(smthrand) * finalsds[k]
        smthrand_all[:,k] = smthrand
        
    return smthrand_all

##############################################################################
def zernike_rms_per_mode(max_rms_perterm, min_rms_perterm, n_zernikes):

   # Vary RMS per zernike mode, with a linear drop-off from start to end mode 
    rms_perterm_multi = np.linspace(max_rms_perterm, min_rms_perterm, 
                                    n_zernikes-1)
    
    # Add a zero for the piston mode
    rms_perterm_multi = np.concatenate(([0], rms_perterm_multi))

    # make tip / tilt the same
    rms_perterm_multi[1] = rms_perterm_multi[2]
        
    return rms_perterm_multi


##############################################################################
# def norm_coeffs(coeffs_in):
#     # Normalise cofficients so polynomials are [-1,1], like zernfun.m
#     coeffs_out = np.zeros_like(coeffs_in)
#     for k in range(coeffs_in.shape[1]):
#         n = cart.ntab[k]
#         m = cart.mtab[k]
#         if m == 0:
#             normfact = np.sqrt(n + 1)
#         else:
#             normfact = np.sqrt(2 * (n + 1))
#         coeffs_out[:, k] = coeffs_in[:, k] / normfact
#     return coeffs_out



##############################################################################
# def make_psf_lp_video(results, outname="psf_lp_evolution.gif", 
#                       save_video=False, fps=15, dpi=150):

#     fields = np.asarray(results["fields"])
#     lp_powers = np.asarray(results["lp_powers"])
#     modelabels = np.asarray(results["modelabels"])
#     nmodes = int(results["nmodes"])

#     n_sims = fields.shape[0]

#     # Fixed limits so the movie does not flicker frame-to-frame
#     intensity_all = np.abs(fields)**2
#     intensity_vmax = np.max(intensity_all)
#     lp_vmax = np.max(lp_powers)

#     fig = plt.figure(figsize=(12, 4))

#     # ------------------------------------------------------------------
#     # Panel 1: PSF intensity
#     # ------------------------------------------------------------------
#     ax1 = plt.subplot(1, 3, 1)
#     im1 = ax1.imshow(
#         intensity_all[0],
#         origin="lower",
#         vmin=0,
#         vmax=intensity_vmax,
#     )
#     ax1.set_title("PSF intensity at HMSPL input")
#     cbar1 = plt.colorbar(im1, ax=ax1)

#     # ------------------------------------------------------------------
#     # Panel 2: PSF phase
#     # ------------------------------------------------------------------
#     ax2 = plt.subplot(1, 3, 2)
#     im2 = ax2.imshow(
#         np.angle(fields[0]),
#         origin="lower",
#         cmap="twilight",
#         vmin=-np.pi,
#         vmax=np.pi,
#     )
#     ax2.set_title("PSF phase")
#     cbar2 = plt.colorbar(im2, ax=ax2)

#     # ------------------------------------------------------------------
#     # Panel 3: LP modal powers
#     # ------------------------------------------------------------------
#     ax3 = plt.subplot(1, 3, 3)
#     x = np.arange(nmodes)
#     bars = ax3.bar(x, lp_powers[0])
#     ax3.set_xticks(x)
#     ax3.set_xticklabels(modelabels[:nmodes], rotation=90)
#     ax3.set_ylim(0, 1.05 * lp_vmax)
#     ax3.set_title("LP modal powers")

#     frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

#     plt.tight_layout()

#     def update(idx):
#         field = fields[idx]

#         # Update PSF intensity
#         im1.set_data(np.abs(field)**2)

#         # Update PSF phase
#         im2.set_data(np.angle(field))

#         # Update LP bar heights
#         for bar, height in zip(bars, lp_powers[idx]):
#             bar.set_height(height)

#         frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

#         return [im1, im2, frame_title, *bars]

#     anim = FuncAnimation(
#         fig,
#         update,
#         frames=n_sims,
#         interval=1000 / fps,
#         blit=False,
#     )

#     if save_video:
#         writer = PillowWriter(fps=fps)
#         # else:
#         #     writer = FFMpegWriter(fps=fps, bitrate=3000)

#         anim.save(outname, writer=writer, dpi=dpi)
#         plt.close(fig)

#         print(f"Saved video to {outname}")



##############################################################################
def make_n_distinct_colors(n, cmap="turbo"):
    """
    Return n visually distinct RGBA colours from a given colormap.

    Parameters
    ----------
    n : int
        Number of colours.
    cmap : str or Colormap
        Matplotlib colormap name or object (e.g. "viridis", "plasma", plt.cm.tab10).

    Returns
    -------
    colors : (n, 4) ndarray
        RGBA colours.
    """
    cmap_obj = plt.get_cmap(cmap)
    return cmap_obj(np.linspace(0, 1, n, endpoint=False))


##############################################################################
def plot_wf_psf_zernike_lp(
    results,
    idx=0,
    figsize=(12, 10),
    wf_key="pupil_wf",
    zernike_key="zernike_coeffs",
    power_key="lp_powers",
    field_key="psf_fields",
    save_plot=False,
    fname_plot='wf_psf_zernike_lp_example.png'
):
    """
    2x2 layout with aligned columns and square panels.

        Top-left     : Pupil wavefront
        Top-right    : PSF intensity
        Bottom-left  : Zernike coefficient bar chart
        Bottom-right : LP modal power bar chart
    """

    field = np.asarray(results[field_key][idx])
    wf = np.asarray(results[wf_key][idx])
    z = np.asarray(results[zernike_key][idx])
    lp = np.asarray(results[power_key][idx])

    nmodes = int(results["nmodes"])
    modelabels = np.asarray(results["modelabels"])

    # Colour sets for bars
    z_colors = make_n_distinct_colors(len(z), cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 1],
    )

    # ==========================================================
    # Top-left : Pupil wavefront
    # ==========================================================
    ax00 = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])

    im1 = ax00.imshow(
        wf,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        aspect="equal",
    )
    ax00.set_title("Wavefront")
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.set_box_aspect(1)

    cb1 = fig.colorbar(im1, cax=cax00)
    cb1.ax.set_title("phase [rad]", fontsize=10, pad=8)

    # ==========================================================
    # Top-right : PSF intensity
    # ==========================================================
    ax01 = fig.add_subplot(gs[0, 2])
    cax01 = fig.add_subplot(gs[0, 3])

    im2 = ax01.imshow(
        np.abs(field)**2,
        origin="lower",
        aspect="equal",
    )
    ax01.set_title("PSF at HMSPL Input")
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.set_box_aspect(1)

    cb2 = fig.colorbar(im2, cax=cax01)
    cb2.ax.set_title("intensity", fontsize=10, pad=8)

    # ==========================================================
    # Bottom-left : Zernike coefficients
    # ==========================================================
    ax10 = fig.add_subplot(gs[1, 0])
    zx = np.arange(len(z))

    ax10.bar(zx, z, width=0.8, color=z_colors)
    ax10.set_title("Zernike Coefficients")
    ax10.set_xlabel("Mode Index")
    ax10.set_ylabel("Coefficient")
    ax10.set_xticks(zx)
    ax10.set_xticklabels(zx + 1, rotation=90)
    ax10.grid(":", linewidth=0.5, alpha=0.4)
    ax10.set_box_aspect(1)

    ax_blank1 = fig.add_subplot(gs[1, 1])
    ax_blank1.axis("off")

    # ==========================================================
    # Bottom-right : LP modal powers
    # ==========================================================
    ax11 = fig.add_subplot(gs[1, 2])
    x = np.arange(nmodes)

    ax11.bar(x, lp, width=0.8, color=lp_colors)
    ax11.set_title("LP Modal Powers")
    ax11.set_xlabel("LP Mode")
    ax11.set_ylabel("Coupled Power")
    ax11.set_xticks(x)
    ax11.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax11.grid(":", linewidth=0.5, alpha=0.4)
    ax11.set_box_aspect(1)

    ax_blank2 = fig.add_subplot(gs[1, 3])
    ax_blank2.axis("off")

    if save_plot:
        plt.savefig(fname_plot, dpi=150)

    plt.show()


##############################################################################
def plot_wf_psf_lp_pl(
    results,
    idx=0,
    figsize=(12, 3),
    wf_key="pupil_wf",
    power_key="lp_powers",
    field_key="psf_fields",
    pl_power_key="pl_powers",
    save_plot=False,
    fname_plot='wf_psf_lp_pl_example.png'
):
    """
    1x4 row layout:

        1. Pupil wavefront
        2. PSF intensity at HMSPL input
        3. LP modal power bar chart
        4. PL output power bar chart
    """

    field = np.asarray(results[field_key][idx])
    wf = np.asarray(results[wf_key][idx])
    lp = np.asarray(results[power_key][idx])
    pl = np.asarray(results[pl_power_key][idx])

    nmodes = int(len(results["modelabels"]))
    modelabels = np.asarray(results["modelabels"])

    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")
    pl_colors = make_n_distinct_colors(len(pl), cmap="viridis")

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # ==========================================================
    # 1. Pupil wavefront
    # ==========================================================
    ax_wf = axes[0]

    im1 = ax_wf.imshow(
        wf,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        aspect="equal",
    )
    ax_wf.set_title("Wavefront")
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_wf.set_box_aspect(1)

    div1 = make_axes_locatable(ax_wf)
    cax1 = div1.append_axes("right", size="5%", pad=0.08)
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.ax.set_title("phase\n[rad]", fontsize=10, pad=8)

    # ==========================================================
    # 2. PSF intensity
    # ==========================================================
    ax_psf = axes[1]

    im2 = ax_psf.imshow(
        np.abs(field)**2,
        origin="lower",
        aspect="equal",
    )
    ax_psf.set_title("PSF")
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    ax_psf.set_box_aspect(1)

    div2 = make_axes_locatable(ax_psf)
    cax2 = div2.append_axes("right", size="5%", pad=0.08)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.ax.set_title("intensity", fontsize=10, pad=8)
    cb2.ax.tick_params(labelsize=8)

    # ==========================================================
    # 3. LP modal powers
    # ==========================================================
    ax_lp = axes[2]
    x = np.arange(nmodes)

    ax_lp.bar(x, lp, width=0.8, color=lp_colors)
    ax_lp.set_title("LP Modal Powers")
    ax_lp.set_xlabel("LP Mode")
    ax_lp.set_ylabel("Coupled Power")
    ax_lp.set_xticks(x)
    ax_lp.set_xticklabels(modelabels[:nmodes], rotation=45)
    ax_lp.set_box_aspect(1)
    ax_lp.grid(":", linewidth=0.5, alpha=0.4)

    # ==========================================================
    # 4. PL output powers
    # ==========================================================
    ax_pl = axes[3]
    cx = np.arange(len(pl))

    ax_pl.bar(cx, pl, width=0.8, color=pl_colors)
    ax_pl.set_title("PL Output Powers")
    ax_pl.set_xlabel("Core Index")
    ax_pl.set_ylabel("Power")
    ax_pl.set_xticks(cx)
    ax_pl.set_xticklabels(cx, rotation=0)
    ax_pl.set_box_aspect(1)
    ax_pl.grid(":", linewidth=0.5, alpha=0.4)

    plt.tight_layout(w_pad=0.5)

    if save_plot:
        plt.savefig(fname_plot, dpi=150, bbox_inches="tight")

    plt.show()


##############################################################################
def make_wf_psf_zernike_video_square(
    results,
    outname="wf_psf_evolution.gif",
    save_video=False,
    fps=30,
    dpi=150,
    figsize=(12, 10),
    psf_key="psf_fields",
    wf_key="pupil_wf",
    zernike_key="zernike_coeffs",
    power_key="lp_powers",
):
    """
    Create a video of the 2x2 evolution plot:

        Top-left     : Pupil wavefront
        Top-right    : PSF intensity
        Bottom-left  : Zernike coefficient bar chart
        Bottom-right : LP modal power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wf = np.asarray(results[wf_key])
    zernikes = np.asarray(results[zernike_key])
    lp_powers = np.asarray(results[power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]

    intensity_all = np.abs(fields)**2

    # Fixed colour/axis limits so the movie does not flicker
    intensity_vmax = np.nanmax(intensity_all)
    wf_vmin = -np.pi
    wf_vmax = np.pi

    z_absmax = np.nanmax(np.abs(zernikes))
    lp_vmax = np.nanmax(lp_powers)

    z_colors = make_n_distinct_colors(zernikes.shape[1], cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 1],
    )

    # ==========================================================
    # Top-left : Pupil wavefront
    # ==========================================================
    ax00 = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])

    im_wf = ax00.imshow(
        pupil_wf[0],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )
    ax00.set_title("Wavefront")
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.set_box_aspect(1)

    cb1 = fig.colorbar(im_wf, cax=cax00)
    cb1.ax.set_title("phase [rad]", fontsize=10, pad=8)

    # ==========================================================
    # Top-right : PSF intensity
    # ==========================================================
    ax01 = fig.add_subplot(gs[0, 2])
    cax01 = fig.add_subplot(gs[0, 3])

    im_psf = ax01.imshow(
        intensity_all[0],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )
    ax01.set_title("PSF at HMSPL Input")
    ax01.set_xticks([])
    ax01.set_yticks([])
    ax01.set_box_aspect(1)

    cb2 = fig.colorbar(im_psf, cax=cax01)
    cb2.ax.set_title("intensity", fontsize=10, pad=8)

    # ==========================================================
    # Bottom-left : Zernike coefficients
    # ==========================================================
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.axhline(y=0, xmin=0, xmax=len(zernikes)+1,
                 color="k", 
                 linestyle="--", 
                 linewidth=0.5, 
                 alpha=0.7)
    zx = np.arange(zernikes.shape[1])

    z_bars = ax10.bar(
        zx,
        zernikes[0],
        width=0.8,
        color=z_colors,
    )
    ax10.set_title("Zernike Coefficients")
    ax10.set_xlabel("Mode Index")
    ax10.set_ylabel("Coefficient")
    ax10.set_xticks(zx)
    ax10.set_xticklabels(zx + 1, rotation=90)
    ax10.set_ylim(-1.05 * z_absmax, 1.05 * z_absmax)
    # ax10.grid(":", linewidth=0.5, alpha=0.4)
    ax10.set_box_aspect(1)

    ax_blank1 = fig.add_subplot(gs[1, 1])
    ax_blank1.axis("off")

    # ==========================================================
    # Bottom-right : LP modal powers
    # ==========================================================
    ax11 = fig.add_subplot(gs[1, 2])
    x = np.arange(nmodes)

    lp_bars = ax11.bar(
        x,
        lp_powers[0],
        width=0.8,
        color=lp_colors,
    )
    ax11.set_title("LP Modal Powers")
    ax11.set_xlabel("LP Mode")
    ax11.set_ylabel("Coupled Power")
    ax11.set_xticks(x)
    ax11.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax11.set_ylim(0, 1.05 * lp_vmax)
    # ax11.grid(":", linewidth=0.5, alpha=0.4)
    ax11.set_box_aspect(1)

    ax_blank2 = fig.add_subplot(gs[1, 3])
    ax_blank2.axis("off")

    frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

    def update(idx):
        # Update wavefront
        im_wf.set_data(pupil_wf[idx])

        # Update PSF intensity
        im_psf.set_data(intensity_all[idx])

        # Update Zernike bar heights
        for bar, height in zip(z_bars, zernikes[idx]):
            bar.set_height(height)

        # Update LP bar heights
        for bar, height in zip(lp_bars, lp_powers[idx]):
            bar.set_height(height)

        frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

        return [im_wf, im_psf, frame_title, *z_bars, *lp_bars]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_sims,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        if outname.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
        # else:
        #     writer = FFMpegWriter(fps=fps, bitrate=3000)

        anim.save(outname, writer=writer, dpi=dpi)
        plt.close(fig)

        print(f"Saved video to {outname}")
    else:
        plt.show()

    return anim


##############################################################################
def make_wf_psf_lp_pl_zernike_video_row(
    results,
    outname="zernike_wf_psf_lp_pl_evolution.gif",
    save_video=True,
    fps=30,
    dpi=150,
    figsize=(22, 4.5),
    psf_key="psf_fields",
    wf_key="pupil_wf",
    zernike_key="zernike_coeffs",
    lp_power_key="lp_powers",
    pl_power_key="pl_powers",
):
    """
    Create a 1x5 video:

        1. Zernike coefficient bar chart
        2. Pupil wavefront
        3. PSF intensity at HMSPL input
        4. LP modal power bar chart
        5. PL output core power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wf = np.asarray(results[wf_key])
    zernikes = np.asarray(results[zernike_key])
    lp_powers = np.asarray(results[lp_power_key])
    pl_powers = np.asarray(results[pl_power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]
    n_zernikes = zernikes.shape[1]
    n_pl_cores = pl_powers.shape[1]

    intensity_all = np.abs(fields) ** 2

    # Fixed limits to avoid flickering
    z_absmax = np.nanmax(np.abs(zernikes))
    wf_vmin, wf_vmax = -np.pi, np.pi
    intensity_vmax = np.nanmax(intensity_all)
    lp_vmax = np.nanmax(lp_powers)
    pl_vmax = np.nanmax(pl_powers)

    z_colors = make_n_distinct_colors(n_zernikes, cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")
    pl_colors = make_n_distinct_colors(n_pl_cores, cmap="cividis")

    fig, axes = plt.subplots(
        1, 5,
        figsize=figsize,
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # 1. Zernike coefficients
    # ----------------------------------------------------------
    ax_z = axes[0]
    zx = np.arange(n_zernikes)

    z_bars = ax_z.bar(
        zx,
        zernikes[0],
        width=0.8,
        color=z_colors,
    )
    ax_z.set_title("Zernike Coefficients")
    ax_z.set_xlabel("Mode Index")
    ax_z.set_ylabel("Coefficient")
    ax_z.set_xticks(zx)
    ax_z.set_xticklabels(zx + 1, rotation=90)
    ax_z.set_ylim(-1.05 * z_absmax, 1.05 * z_absmax)
    # ax_z.grid(":", linewidth=0.5, alpha=0.4)
    ax_z.set_box_aspect(1)

    # ----------------------------------------------------------
    # 2. Pupil wavefront
    # ----------------------------------------------------------
    ax_wf = axes[1]

    im_wf = ax_wf.imshow(
        pupil_wf[0],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )
    ax_wf.set_title("Wavefront")
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_wf.set_box_aspect(1)

    cb_wf = fig.colorbar(im_wf, ax=ax_wf, fraction=0.046, pad=0.04)
    cb_wf.ax.set_title("phase [rad]", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 3. PSF intensity
    # ----------------------------------------------------------
    ax_psf = axes[2]

    im_psf = ax_psf.imshow(
        intensity_all[0],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )
    ax_psf.set_title("PSF Intensity")
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    ax_psf.set_box_aspect(1)

    cb_psf = fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
    cb_psf.ax.set_title("intensity", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 4. LP modal powers
    # ----------------------------------------------------------
    ax_lp = axes[3]
    lx = np.arange(nmodes)

    lp_bars = ax_lp.bar(
        lx,
        lp_powers[0],
        width=0.8,
        color=lp_colors,
    )
    ax_lp.set_title("LP Modal Powers")
    ax_lp.set_xlabel("LP Mode")
    ax_lp.set_ylabel("Coupled Power")
    ax_lp.set_xticks(lx)
    ax_lp.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax_lp.set_ylim(0, 1.05 * lp_vmax)
    # ax_lp.grid(":", linewidth=0.5, alpha=0.4)
    ax_lp.set_box_aspect(1)

    # ----------------------------------------------------------
    # 5. PL output core powers
    # ----------------------------------------------------------
    ax_pl = axes[4]
    px = np.arange(n_pl_cores)

    pl_bars = ax_pl.bar(
        px,
        pl_powers[0],
        width=0.8,
        color=pl_colors,
    )
    ax_pl.set_title("PL Output Core Powers")
    ax_pl.set_xlabel("Core Index")
    ax_pl.set_ylabel("Power")
    ax_pl.set_xticks(px)
    ax_pl.set_xticklabels(px + 1)
    ax_pl.set_ylim(0, 1.05 * pl_vmax)
    # ax_pl.grid(":", linewidth=0.5, alpha=0.4)
    ax_pl.set_box_aspect(1)

    frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

    def update(idx):
        im_wf.set_data(pupil_wf[idx])
        im_psf.set_data(intensity_all[idx])

        for bar, height in zip(z_bars, zernikes[idx]):
            bar.set_height(height)

        for bar, height in zip(lp_bars, lp_powers[idx]):
            bar.set_height(height)

        for bar, height in zip(pl_bars, pl_powers[idx]):
            bar.set_height(height)

        frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

        return [
            im_wf,
            im_psf,
            frame_title,
            *z_bars,
            *lp_bars,
            *pl_bars,
        ]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_sims,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        if outname.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)

        anim.save(outname, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Saved video to {outname}")
    else:
        plt.show()

    return anim


###############################################################################
def make_wf_psf_lp_pl_video_row(
    results,
    outname="wf_psf_lp_pl_evolution.gif",
    save_video=True,
    fps=30,
    dpi=150,
    figsize=(18, 4.5),
    psf_key="psf_fields",
    wf_key="pupil_wf",
    lp_power_key="lp_powers",
    pl_power_key="pl_powers",
):
    """
    Create a 1x4 animated video:

        1. Pupil wavefront
        2. PSF intensity at HMSPL input
        3. LP modal power bar chart
        4. PL output core power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wf = np.asarray(results[wf_key])
    lp_powers = np.asarray(results[lp_power_key])
    pl_powers = np.asarray(results[pl_power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]
    n_pl_cores = pl_powers.shape[1]

    intensity_all = np.abs(fields) ** 2

    wf_vmin, wf_vmax = -np.pi, np.pi
    intensity_vmax = np.nanmax(intensity_all)
    lp_vmax = np.nanmax(lp_powers)
    pl_vmax = np.nanmax(pl_powers)

    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")
    pl_colors = make_n_distinct_colors(n_pl_cores, cmap="cividis")

    fig, axes = plt.subplots(
        1, 4,
        figsize=figsize,
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # 1. Pupil wavefront
    # ----------------------------------------------------------
    ax_wf = axes[0]

    im_wf = ax_wf.imshow(
        pupil_wf[0],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )
    ax_wf.set_title("Wavefront")
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_wf.set_box_aspect(1)

    cb_wf = fig.colorbar(im_wf, ax=ax_wf, fraction=0.046, pad=0.04)
    cb_wf.ax.set_title("phase [rad]", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 2. PSF intensity
    # ----------------------------------------------------------
    ax_psf = axes[1]

    im_psf = ax_psf.imshow(
        intensity_all[0],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )
    ax_psf.set_title("PSF Intensity")
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    ax_psf.set_box_aspect(1)

    cb_psf = fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
    cb_psf.ax.set_title("intensity", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 3. LP modal powers
    # ----------------------------------------------------------
    ax_lp = axes[2]
    lx = np.arange(nmodes)

    lp_bars = ax_lp.bar(
        lx,
        lp_powers[0],
        width=0.8,
        color=lp_colors,
    )
    ax_lp.set_title("LP Modal Powers")
    ax_lp.set_xlabel("LP Mode")
    ax_lp.set_ylabel("Coupled Power")
    ax_lp.set_xticks(lx)
    ax_lp.set_xticklabels(modelabels[:nmodes], rotation=90)
    ax_lp.set_ylim(0, 1.05 * lp_vmax)
    ax_lp.set_box_aspect(1)

    # ----------------------------------------------------------
    # 4. PL output core powers
    # ----------------------------------------------------------
    ax_pl = axes[3]
    px = np.arange(n_pl_cores)

    pl_bars = ax_pl.bar(
        px,
        pl_powers[0],
        width=0.8,
        color=pl_colors,
    )
    ax_pl.set_title("PL Output Core Powers")
    ax_pl.set_xlabel("Core Index")
    ax_pl.set_ylabel("Power")
    ax_pl.set_xticks(px)
    ax_pl.set_xticklabels(px + 1)
    ax_pl.set_ylim(0, 1.05 * pl_vmax)
    ax_pl.set_box_aspect(1)

    frame_title = fig.suptitle(f"Simulation 0 / {n_sims - 1}")

    def update(idx):
        im_wf.set_data(pupil_wf[idx])
        im_psf.set_data(intensity_all[idx])

        for bar, height in zip(lp_bars, lp_powers[idx]):
            bar.set_height(height)

        for bar, height in zip(pl_bars, pl_powers[idx]):
            bar.set_height(height)

        frame_title.set_text(f"Simulation {idx} / {n_sims - 1}")

        return [im_wf, im_psf, frame_title, *lp_bars, *pl_bars]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_sims,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        if outname.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)

        anim.save(outname, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Saved video to {outname}")
    else:
        plt.show()

    return anim


###############################################################################
def plot_wf_psf_lp_pl_figure_row(
    results,
    sim_idx=0,
    outname="zernike_wf_psf_lp_pl_single.png",
    savefig=True,
    dpi=150,
    figsize=(22, 4.5),
    psf_key="psf_fields",
    wf_key="pupil_wf",
    zernike_key="zernike_coeffs",
    lp_power_key="lp_powers",
    pl_power_key="pl_powers",
):
    """
    Create a single 1x5 static figure for one simulation:

        1. Zernike coefficient bar chart
        2. Pupil wavefront
        3. PSF intensity at HMSPL input
        4. LP modal power bar chart
        5. PL output core power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wf = np.asarray(results[wf_key])
    zernikes = np.asarray(results[zernike_key])
    lp_powers = np.asarray(results[lp_power_key])
    pl_powers = np.asarray(results[pl_power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]
    n_zernikes = zernikes.shape[1]
    n_pl_cores = pl_powers.shape[1]

    if sim_idx < 0 or sim_idx >= n_sims:
        raise ValueError(f"sim_idx must be between 0 and {n_sims - 1}, got {sim_idx}")

    intensity_all = np.abs(fields) ** 2

    # Fixed limits, matching the video version
    z_absmax = np.nanmax(np.abs(zernikes))
    wf_vmin, wf_vmax = -np.pi, np.pi
    intensity_vmax = np.nanmax(intensity_all)
    lp_vmax = np.nanmax(lp_powers)
    pl_vmax = np.nanmax(pl_powers)

    z_colors = make_n_distinct_colors(n_zernikes, cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")
    pl_colors = make_n_distinct_colors(n_pl_cores, cmap="cividis")

    fig, axes = plt.subplots(
        1, 5,
        figsize=figsize,
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # 1. Zernike coefficients
    # ----------------------------------------------------------
    ax_z = axes[0]
    zx = np.arange(n_zernikes)

    ax_z.bar(
        zx,
        zernikes[sim_idx],
        width=0.8,
        color=z_colors,
    )

    ax_z.set_title("Zernike Coefficients")
    ax_z.set_xlabel("Mode Index")
    ax_z.set_ylabel("Coefficient")
    ax_z.set_xticks(zx)
    ax_z.set_xticklabels(zx + 1, rotation=90)

    if z_absmax > 0:
        ax_z.set_ylim(-1.05 * z_absmax, 1.05 * z_absmax)

    ax_z.set_box_aspect(1)

    # ----------------------------------------------------------
    # 2. Pupil wavefront
    # ----------------------------------------------------------
    ax_wf = axes[1]

    im_wf = ax_wf.imshow(
        pupil_wf[sim_idx],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )

    ax_wf.set_title("Wavefront")
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_wf.set_box_aspect(1)

    cb_wf = fig.colorbar(im_wf, ax=ax_wf, fraction=0.046, pad=0.04)
    cb_wf.ax.set_title("phase [rad]", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 3. PSF intensity
    # ----------------------------------------------------------
    ax_psf = axes[2]

    im_psf = ax_psf.imshow(
        intensity_all[sim_idx],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )

    ax_psf.set_title("PSF Intensity")
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    ax_psf.set_box_aspect(1)

    cb_psf = fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
    cb_psf.ax.set_title("intensity", fontsize=9, pad=8)

    # ----------------------------------------------------------
    # 4. LP modal powers
    # ----------------------------------------------------------
    ax_lp = axes[3]
    lx = np.arange(nmodes)

    ax_lp.bar(
        lx,
        lp_powers[sim_idx],
        width=0.8,
        color=lp_colors,
    )

    ax_lp.set_title("LP Modal Powers")
    ax_lp.set_xlabel("LP Mode")
    ax_lp.set_ylabel("Coupled Power")
    ax_lp.set_xticks(lx)
    ax_lp.set_xticklabels(modelabels[:nmodes], rotation=90)

    if lp_vmax > 0:
        ax_lp.set_ylim(0, 1.05 * lp_vmax)

    ax_lp.set_box_aspect(1)

    # ----------------------------------------------------------
    # 5. PL output core powers
    # ----------------------------------------------------------
    ax_pl = axes[4]
    px = np.arange(n_pl_cores)

    ax_pl.bar(
        px,
        pl_powers[sim_idx],
        width=0.8,
        color=pl_colors,
    )

    ax_pl.set_title("PL Output Core Powers")
    ax_pl.set_xlabel("Core Index")
    ax_pl.set_ylabel("Power")
    ax_pl.set_xticks(px)
    ax_pl.set_xticklabels(px + 1)

    if pl_vmax > 0:
        ax_pl.set_ylim(0, 1.05 * pl_vmax)

    ax_pl.set_box_aspect(1)

    fig.suptitle(f"Simulation {sim_idx} / {n_sims - 1}")

    if savefig:
        fig.savefig(outname, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {outname}")
    else:
        plt.show()

    return fig, axes


###############################################################################
def plot_wf_psf_lp_figure_row(
    results,
    sim_idx=0,
    outname="zernike_wf_psf_lp_single.png",
    savefig=True,
    dpi=150,
    figsize=(18, 4.5),
    psf_key="psf_fields",
    wf_key="pupil_wf",
    zernike_key="zernike_coeffs",
    lp_power_key="lp_powers",
):
    """
    Create a single 1x4 static figure for one simulation:

        1. Zernike coefficient bar chart
        2. Pupil wavefront
        3. PSF intensity at HMSPL input
        4. LP modal power bar chart
    """

    fields = np.asarray(results[psf_key])
    pupil_wf = np.asarray(results[wf_key])
    zernikes = np.asarray(results[zernike_key])
    lp_powers = np.asarray(results[lp_power_key])

    modelabels = np.asarray(results["modelabels"])
    nmodes = int(results["nmodes"])

    n_sims = fields.shape[0]
    n_zernikes = zernikes.shape[1]

    if sim_idx < 0 or sim_idx >= n_sims:
        raise ValueError(f"sim_idx must be between 0 and {n_sims - 1}, got {sim_idx}")

    intensity_all = np.abs(fields) ** 2

    z_absmax = np.nanmax(np.abs(zernikes))
    wf_vmin, wf_vmax = -np.pi, np.pi
    intensity_vmax = np.nanmax(intensity_all)
    lp_vmax = np.nanmax(lp_powers)

    z_colors = make_n_distinct_colors(n_zernikes, cmap="turbo")
    lp_colors = make_n_distinct_colors(nmodes, cmap="magma")

    fig, axes = plt.subplots(
        1, 4,
        figsize=figsize,
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # 1. Zernike coefficients
    # ----------------------------------------------------------
    ax_z = axes[0]
    zx = np.arange(n_zernikes)

    ax_z.bar(
        zx,
        zernikes[sim_idx],
        width=0.8,
        color=z_colors,
    )

    ax_z.set_title("Zernike Coefficients", fontsize=16)
    ax_z.set_xlabel("Mode Index", fontsize=14)
    ax_z.set_ylabel("Coefficient", fontsize=14)
    ax_z.set_xticks(zx)
    ax_z.set_xticklabels(zx + 1, fontsize=8, rotation=90)

    if z_absmax > 0:
        ax_z.set_ylim(-1.05 * z_absmax, 1.05 * z_absmax)

    ax_z.grid(":", linewidth=0.5, alpha=0.4)
    ax_z.set_box_aspect(1)

    # ----------------------------------------------------------
    # 2. Pupil wavefront
    # ----------------------------------------------------------
    ax_wf = axes[1]

    im_wf = ax_wf.imshow(
        pupil_wf[sim_idx],
        cmap="twilight",
        vmin=wf_vmin,
        vmax=wf_vmax,
        origin="lower",
        aspect="equal",
    )

    ax_wf.set_title("Wavefront", fontsize=16)
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_wf.set_box_aspect(1)

    divider_wf = make_axes_locatable(ax_wf)
    cax_wf = divider_wf.append_axes("right", size="4%", pad=0.08)

    cb_wf = fig.colorbar(im_wf, cax=cax_wf)
    cb_wf.ax.set_title("phase [rad]", fontsize=11, pad=8)

    # ----------------------------------------------------------
    # 3. PSF intensity
    # ----------------------------------------------------------
    ax_psf = axes[2]

    im_psf = ax_psf.imshow(
        intensity_all[sim_idx],
        origin="lower",
        vmin=0,
        vmax=intensity_vmax,
        aspect="equal",
    )

    ax_psf.set_title("PSF Intensity", fontsize=16)
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    ax_psf.set_box_aspect(1)

    divider_psf = make_axes_locatable(ax_psf)
    cax_psf = divider_psf.append_axes("right", size="4%", pad=0.08)

    cb_psf = fig.colorbar(im_psf, cax=cax_psf)
    cb_psf.ax.set_title("intensity", fontsize=11, pad=8)

    # ----------------------------------------------------------
    # 4. LP modal powers
    # ----------------------------------------------------------
    ax_lp = axes[3]
    lx = np.arange(nmodes)

    ax_lp.bar(
        lx,
        lp_powers[sim_idx],
        width=0.8,
        color=lp_colors,
    )

    ax_lp.set_title("LP Modal Powers", fontsize=16)
    ax_lp.set_xlabel("LP Mode", fontsize=14)
    ax_lp.set_ylabel("Coupled Power", fontsize=14)
    ax_lp.set_xticks(lx)
    ax_lp.set_xticklabels(modelabels[:nmodes], rotation=0)
    ax_lp.grid(":", linewidth=0.5, alpha=0.4)

    if lp_vmax > 0:
        ax_lp.set_ylim(0, 1.05 * lp_vmax)

    ax_lp.set_box_aspect(1)

    # fig.suptitle(f"Simulation {sim_idx} / {n_sims - 1}")

    if savefig:
        fig.savefig(outname, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {outname}")
    else:
        plt.show()

    return fig, axes


##############################################################################
def animate_lp_wf_predictions(
    X_test,
    y_test_wf,
    predictions_wf,
    wf_shape,
    mask=None,
    normfacts_PL=None,
    n_frames=None,
    start_idx=0,
    figsize=(15, 5),
    interval=100,
    cmap_wf='twilight',
    save_gif=False,
    fname_gif='wf_animation.gif',
    fps=10,
):
    """
    Animate PL input powers, true wavefront, and predicted wavefront.

    Subplots (left to right):
        1 : PL modal powers — bar chart (n_pl_ports bars)
        2 : True wavefront  — imshow
        3 : Predicted wavefront — imshow

    Parameters
    ----------
    X_test          : (N, seq_len, n_pl_ports) or (N, n_pl_ports) PL power inputs
    y_test_wf       : (N, H*W) true wavefront vectors
    predictions_wf  : (N, H*W) predicted wavefront vectors
    wf_shape        : (H, W) reshape target
    mask            : (H, W) boolean pupil mask, or None
    normfacts_PL    : [mean, _, std] normalisation factors; if given, PL powers
                      are unnormalised before plotting
    n_frames        : number of samples to animate (default: all)
    start_idx       : first sample index
    figsize         : figure size
    interval        : ms between frames
    cmap_wf         : colormap for wavefront images
    save_gif        : write GIF to disk
    fname_gif       : output filename
    fps             : frames per second for saved GIF
    """

    def _unnorm(pl):
        if normfacts_PL is not None:
            return pl * normfacts_PL[2] + normfacts_PL[0]
        return pl

    N_total = X_test.shape[0]
    end_idx = N_total if n_frames is None else min(start_idx + n_frames, N_total)
    indices = np.arange(start_idx, end_idx)

    n_ports = X_test.shape[-1]
    x_bars = np.arange(n_ports)

    # Pre-compute all wavefronts so reshaping isn't done per frame
    true_wfs = vector_to_wf(y_test_wf[start_idx:end_idx], wf_shape, mask=mask)
    pred_wfs = vector_to_wf(predictions_wf[start_idx:end_idx], wf_shape, mask=mask)

    # Fixed y-limits from unnormalised values across all animated frames
    X_window = X_test[start_idx:end_idx]
    pl_max = _unnorm(X_window.max())
    pl_min = _unnorm(X_window.min())

    fig, (ax_lp, ax_true, ax_pred) = plt.subplots(1, 3, figsize=figsize)


    # --- Subplot 1: PL powers bar chart ---
    _pl0 = X_test[start_idx, -1, :] if X_test.ndim == 3 else X_test[start_idx]
    bars = ax_lp.bar(x_bars, _unnorm(_pl0),
                     width=1.0,
                     color='steelblue',
                     linewidth=0)

    ax_lp.set_xlim(-0.5, n_ports - 0.5)
    ax_lp.set_ylim(pl_min * 1.05, pl_max * 1.05)

    ax_lp.set_title('PL Powers')
    ax_lp.set_xlabel('Core index')
    ax_lp.set_ylabel('Power (unnorm.)' if normfacts_PL is not None else 'Power')

    ax_lp.tick_params(labelsize=7)


    # --- Subplot 2: True wavefront ---
    wf0 = true_wfs[0]
    # clim0 = max(abs(wf0.min()), abs(wf0.max()))
    im_true = ax_true.imshow(wf0, 
                             cmap=cmap_wf, 
                             vmin=-np.pi, 
                             vmax=np.pi, 
                             origin='lower')
    
    ax_true.set_title('True WF')
    ax_true.axis('off')
    cb_true = fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
    cb_true.ax.tick_params(labelsize=7)

    # --- Subplot 3: Predicted wavefront ---
    im_pred = ax_pred.imshow(pred_wfs[0], 
                             cmap=cmap_wf, 
                             vmin=-np.pi, 
                             vmax=np.pi, 
                             origin='lower')
    
    ax_pred.set_title('Predicted WF')
    ax_pred.axis('off')
    cb_pred = fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    cb_pred.ax.tick_params(labelsize=7)

    title = fig.suptitle(f'Sample {indices[0]}', fontsize=10)
    plt.tight_layout()

    def _update(frame):
        _lp = X_test[start_idx + frame]
        lp = _unnorm(_lp[-1, :] if _lp.ndim == 2 else _lp)
        for bar, h in zip(bars, lp):
            bar.set_height(h)

        im_true.set_data(true_wfs[frame])
        im_true.set_clim(-np.pi, np.pi)
        im_pred.set_data(pred_wfs[frame])
        im_pred.set_clim(-np.pi, np.pi)

        title.set_text(f'Sample {indices[frame]}')
        return bars.patches + [im_true, im_pred, title]

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(indices),
        interval=interval,
        blit=True,
    )

    if save_gif:
        anim.save(fname_gif, writer='pillow', fps=fps)
        print(f"Saved GIF to {fname_gif}")

    plt.show()
    return anim


###########################################################################
def plot_wf_predictions(
    fpath,
    predictions_wf,
    y_test_wf,
    X_test,
    wf_shape,
    mask=None,
    n_show=5,
    start_idx=0,
    cmap_wf='twilight',
    save_fig=True,
    save_path=None,
    dpi=150,
):
    """
    Plot PL inputs, true wavefronts, and predicted wavefronts for a set of samples.

    Rows (top to bottom) per sample column:
        0 : PL modal powers — bar chart
        1 : True wavefront  — imshow
        2 : Predicted wavefront — imshow

    Parameters
    ----------
    fpath           : path to the source _preds.npz file (used for title/save path)
    predictions_wf  : (N, H*W) predicted wavefront vectors
    y_test_wf       : (N, H*W) true wavefront vectors
    X_test          : (N, n_pl_ports) PL power inputs
    wf_shape        : (H, W) reshape target
    n_show          : number of samples to plot
    start_idx       : index of first sample
    cmap_wf         : colormap for wavefront images
    save_fig        : save figure alongside source file
    dpi             : resolution for saved figure
    """
    print(f"\nPlotting {os.path.basename(fpath)}")

    pl_colors = make_n_distinct_colors(X_test.shape[-1], cmap="viridis")

    n_show_actual = min(n_show, predictions_wf.shape[0] - start_idx)
    indices = range(start_idx, start_idx + n_show_actual)

    fig = plt.figure(figsize=(n_show_actual * 3, 9))
    fig.suptitle(os.path.basename(fpath), fontsize=10)
    gs = fig.add_gridspec(3, n_show_actual + 1,
                          width_ratios=[1] * n_show_actual + [0.05],
                          wspace=0.05, hspace=0.15)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_show_actual)]
                     for r in range(3)])
    cb_ax1 = fig.add_subplot(gs[1, -1])
    cb_ax2 = fig.add_subplot(gs[2, -1])

    for col, idx in enumerate(indices):
        true_wf = vector_to_wf(y_test_wf[idx:idx+1],      wf_shape, mask=mask)[0]
        pred_wf = vector_to_wf(predictions_wf[idx:idx+1], wf_shape, mask=mask)[0]
        resid   = true_wf - pred_wf

        clim = [true_wf.min(), true_wf.max()]

        pl_frame = X_test[idx, -1, :] if X_test.ndim == 3 else X_test[idx]
        axes[0, col].bar(range(pl_frame.shape[0]), pl_frame, color=pl_colors)
        axes[0, col].set_title(f'PL inputs [{idx}]', fontsize=8)
        axes[0, col].tick_params(labelsize=6)

        im1 = axes[1, col].imshow(true_wf, vmin=-np.pi, vmax=np.pi, cmap=cmap_wf)
        axes[1, col].axis('off')

        im2 = axes[2, col].imshow(pred_wf, vmin=-np.pi, vmax=np.pi, cmap=cmap_wf)
        axes[2, col].axis('off')

    fig.colorbar(im1, cax=cb_ax1)
    cb_ax1.tick_params(labelsize=7)
    fig.colorbar(im2, cax=cb_ax2)
    cb_ax2.tick_params(labelsize=7)

    if save_fig:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved plot to {os.path.basename(save_path)}")

    return fig


###########################################################################
def plot_wf_prediction(
    predictions_wf,
    y_test_wf,
    X_test,
    wf_shape,
    mask=None,
    normfacts_PL=None,
    idx=0,
    cmap_wf='twilight',
    figsize=(9.6, 3.3),
    title=None,
    save_path=None,
    dpi=150,
):
    """
    Plot a single example as a row: PL powers | predicted WF | true WF.

    Parameters
    ----------
    predictions_wf  : (N, H*W) predicted wavefront vectors
    y_test_wf       : (N, H*W) true wavefront vectors
    X_test          : (N, seq_len, n_pl_ports) or (N, n_pl_ports) PL inputs
    wf_shape        : (H, W) reshape target
    mask            : (H, W) boolean pupil mask, or None
    normfacts_PL    : [mean, _, std] normalisation factors; if given, PL powers
                      are unnormalised before plotting
    idx             : sample index to plot
    cmap_wf         : colormap for wavefront images
    figsize         : figure size
    title           : figure suptitle, or None
    save_path       : save figure to this path if provided
    dpi             : resolution for saved figure
    """

    pred_wf = vector_to_wf(predictions_wf[idx:idx+1], wf_shape, mask=mask)[0]
    true_wf = vector_to_wf(y_test_wf[idx:idx+1],      wf_shape, mask=mask)[0]
    pl_frame = X_test[idx, -1, :] if X_test.ndim == 3 else X_test[idx]
    if normfacts_PL is not None:
        pl_frame = pl_frame * normfacts_PL[2] + normfacts_PL[0]
    pl_colors = make_n_distinct_colors(len(pl_frame), cmap="viridis")

    fig, (ax_pl, ax_pred, ax_true) = plt.subplots(1, 3, figsize=figsize)

    ax_pl.bar(range(len(pl_frame)), pl_frame, color=pl_colors)
    ax_pl.set_title('PL Powers')
    ax_pl.set_xlabel('Core index')
    ax_pl.set_ylabel('Power' if normfacts_PL is not None else 'Power (normalised)')
    ax_pl.tick_params(labelsize=8)

    ax_pred.imshow(pred_wf, 
                   vmin=-np.pi, 
                   vmax=np.pi, 
                   cmap=cmap_wf, 
                   origin='lower')
    ax_pred.set_title('Estimated WF')
    ax_pred.axis('off')

    im_true = ax_true.imshow(true_wf, 
                             vmin=-np.pi, 
                             vmax=np.pi, 
                             cmap=cmap_wf, 
                             origin='lower')
    ax_true.set_title('True WF')
    ax_true.axis('off')
    cb = fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.ax.set_title('phase [rad]', fontsize=9)

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved plot to {os.path.basename(save_path)}")

    return fig


###########################################################################
def plot_rmse_compare_histograms(
    rmse_wf_list,
    labels,
    colors=None,
    bins=100,
    figsize=(8, 4),
    xlim=None,
    title='TNN WF Estimation Errors',
    save_path=None,
    dpi=150,
):
    """
    Plot overlaid WF RMSE histograms for multiple models/conditions.

    Parameters
    ----------
    rmse_wf_list : list of (N,) arrays, one per model/condition
    labels       : list of legend label strings
    colors       : list of colors (defaults to tab10)
    bins         : number of histogram bins
    figsize      : figure size
    xlim         : (xmin, xmax) x-axis limits, or None for auto
    title        : figure suptitle
    save_path    : save figure to this path if provided
    dpi          : resolution for saved figure
    """
    n = len(rmse_wf_list)
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(n)]

    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

    for rmse_wf, label, color in zip(rmse_wf_list, labels, colors):
        weights = np.ones_like(rmse_wf) / len(rmse_wf)
        ax.hist(rmse_wf,
                bins=bins,
                weights=weights,
                histtype='step',
                alpha=0.8,
                linewidth=1.5,
                color=color,
                label=label)

    ax.set_xlabel(r'$\mathrm{Wavefront \ RMSE \ [rad]}$')
    ax.set_ylabel(r'$\mathrm{PDF}$')
    ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    if title:
        plt.suptitle(title, fontsize=14)

    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved histogram to {os.path.basename(save_path)}")

    return fig, ax


###########################################################################
def plot_rmse_histograms(
    rmse_wf_list,
    labels,
    colors=None,
    bins=100,
    figsize=(8, 4),
    xlim=None,
    title='TNN WF RMSE Distribution',
    save_path=None,
    dpi=150,
):
    """
    Plot overlaid RMSE histograms for one or more models/conditions.

    Parameters
    ----------
    rmse_wf_list : list of (N,) arrays, one per model/condition
    labels       : list of legend label strings
    colors       : list of colors (defaults to tab10)
    bins         : number of histogram bins
    figsize      : figure size
    xlim         : (xmin, xmax) x-axis limits, or None for auto
    title        : figure title
    save_path    : save figure to this path if provided
    dpi          : resolution for saved figure
    """
    n = len(rmse_wf_list)
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    for rmse, label, color in zip(rmse_wf_list, labels, colors):
        ax.hist(rmse, bins=bins, histtype='stepfilled',
                alpha=0.4, color=color, label=label)
        ax.hist(rmse, bins=bins, histtype='step',
                linewidth=1.5, color=color)
        ax.axvline(rmse.mean(), color=color, linestyle='--',
                   linewidth=1.2, label=f'{label} mean={rmse.mean():.3f}')

    ax.set_xlabel(r'Wavefront RMSE [rad]')
    ax.set_ylabel('Count')
    ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    if title:
        ax.set_title(title, fontsize=12)

    ax.legend(fontsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved histogram to {os.path.basename(save_path)}")

    return fig, ax


##############################################################################
def make_pupil_mask(shape):
    """Boolean circular pupil mask for a (H, W) wavefront grid.

    Replicates the circular aperture used in SeidrSim / atmospherics:
    True inside the inscribed circle, False outside.
    """
    H, W = shape
    y = np.arange(H) - H // 2
    x = np.arange(W) - W // 2
    X, Y = np.meshgrid(x, y)
    return X**2 + Y**2 <= (min(H, W) / 2) ** 2


##############################################################################
def wf_to_vector(wf, mask=None):
    """Flatten (N, H, W) wavefront array to (N, n_px).

    If mask is provided, only pixels inside the pupil are kept,
    reducing n_px from H*W to the number of True pixels in mask.
    """
    if mask is None:
        return wf.reshape(wf.shape[0], -1)
    return wf[:, mask]


def vector_to_wf(vec, wf_shape, mask=None):
    """Reconstruct (N, H, W) wavefront from a (N, n_px) vector.

    If mask is provided, pixels are placed back into their original
    positions; out-of-pupil pixels are filled with zero.
    """
    if mask is None:
        return vec.reshape(vec.shape[0], *wf_shape)
    out = np.zeros((vec.shape[0], *wf_shape), dtype=vec.dtype)
    out[:, mask] = vec
    return out


###########################################################################
def plot_histograms(
    data_list,
    labels,
    colors=None,
    bins=100,
    figsize=(8, 4),
    xlim=None,
    xlabel='Value',
    ylabel='Count',
    title=None,
    show_mean=True,
    save_path=None,
    dpi=150,
):
    """
    Plot overlaid histograms for one or more datasets.

    Parameters
    ----------
    data_list  : list of (N,) arrays, one per condition
    labels     : list of legend label strings
    colors     : list of colors (defaults to tab10)
    bins       : number of histogram bins
    figsize    : figure size
    xlim       : (xmin, xmax) x-axis limits, or None for auto
    xlabel     : x-axis label
    ylabel     : y-axis label
    title      : figure title, or None
    show_mean  : draw a dashed vertical line at each dataset's mean
    save_path  : save figure to this path if provided
    dpi        : resolution for saved figure
    """
    n = len(data_list)
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    for data, label, color in zip(data_list, labels, colors):
        ax.hist(data, bins=bins, histtype='stepfilled',
                alpha=0.4, color=color, label=label)
        ax.hist(data, bins=bins, histtype='step',
                linewidth=1.5, color=color)
        if show_mean:
            ax.axvline(data.mean(), color=color, linestyle='--',
                       linewidth=1.2, label=f'{label} mean={data.mean():.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    if title:
        ax.set_title(title, fontsize=12)

    ax.legend(fontsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved histogram to {os.path.basename(save_path)}")

    return fig, ax


###########################################################################
def plot_wf_predictions_compare(
    predictions_wf_list,
    y_test_wf,
    X_test,
    wf_shape,
    labels,
    mask=None,
    n_show=5,
    start_idx=0,
    cmap_wf='twilight',
    save_fig=True,
    save_path=None,
    dpi=150,
):
    """
    Compare WF predictions from two models against the same ground truth.

    Rows per sample column:
        0 : PL inputs (X_test, first bar red)
        1 : True wavefront
        2 : Predicted WF for labels[0]
        3 : Predicted WF for labels[1]

    Parameters
    ----------
    predictions_wf_list : [preds_a, preds_b], each (N, H*W)
    y_test_wf           : (N, H*W) shared ground truth wavefront vectors
    X_test              : (N, n_ports) PL power inputs (first bar plotted red)
    wf_shape            : (H, W) reshape target
    labels              : [label_a, label_b] model label strings
    n_show              : number of sample columns to plot
    start_idx           : index of first sample
    cmap_wf             : colormap for wavefront images
    save_fig            : save figure to save_path
    save_path           : output file path
    dpi                 : resolution for saved figure
    """
    n_show_actual = min(n_show, y_test_wf.shape[0] - start_idx)
    indices = range(start_idx, start_idx + n_show_actual)

    n_ports = X_test.shape[1]
    bar_colors = ['tab:red'] + ['steelblue'] * (n_ports - 1)
    clim = [-np.pi, np.pi]

    fig = plt.figure(figsize=(n_show_actual * 3, 12))
    gs = fig.add_gridspec(4, n_show_actual + 1,
                          width_ratios=[1] * n_show_actual + [0.05],
                          left=0.1, right=0.98,
                          wspace=0.05, hspace=0.15)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_show_actual)]
                     for r in range(4)])
    cb_axes = [fig.add_subplot(gs[r, -1]) for r in [1, 2, 3]]

    for col, idx in enumerate(indices):
        true_wf   = vector_to_wf(y_test_wf[idx:idx+1], 
                                 wf_shape, 
                                 mask=mask)[0]
        pred_wf_a = vector_to_wf(predictions_wf_list[0][idx:idx+1], 
                                 wf_shape, mask=mask)[0]
        pred_wf_b = vector_to_wf(predictions_wf_list[1][idx:idx+1], 
                                 wf_shape, 
                                 mask=mask)[0]
        
        pl_frame = X_test[idx, -1, :] if X_test.ndim == 3 else X_test[idx]
        axes[0, col].bar(range(pl_frame.shape[0]), 
                         pl_frame, 
                         color=bar_colors)

        im0 = axes[1, col].imshow(true_wf, 
                                  vmin=clim[0], 
                                  vmax=clim[1], 
                                  cmap=cmap_wf)
        axes[1, col].axis('off')

        im1 = axes[2, col].imshow(pred_wf_a, 
                                  vmin=clim[0], 
                                  vmax=clim[1], 
                                  cmap=cmap_wf)
        axes[2, col].axis('off')

        im2 = axes[3, col].imshow(pred_wf_b, 
                                  vmin=clim[0], 
                                  vmax=clim[1], 
                                  cmap=cmap_wf)
        axes[3, col].axis('off')

    for cax, im in zip(cb_axes, [im0, im1, im2]):
        fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=7)

    row_labels = ['PL inputs', 'True WF',
                  f'{labels[0]} Pred WF', f'{labels[1]} Pred WF']
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.08, 0.5, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=9, va='center', ha='right',
                          rotation=90, fontweight='bold')

    if save_fig and save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved comparison plot to {os.path.basename(save_path)}")

    return fig