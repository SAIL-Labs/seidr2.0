import numpy as np
import matplotlib.pyplot as plt

import lanternfiber
import zernikePSF

from seidr.seidr_functions_misc import make_smoothrand_multi,\
      zernike_rms_per_mode, load_lb_transfer_matrix


#%%###########################################################################
def source2pl_zernike_temporal(
    n_sims=100,
    wavel=1.55,
    f_number=4.5,
    pupil_diameter=1.8,
    n_core=1.4440,
    n_cladding=1.4400,
    core_diameter=18.0,
    max_r=3.0,
    wf_npixels=None,
    psf_npixels=256,
    n_zernikes=30,
    max_rms_perterm=0.4,
    min_rms_perterm=0.05,
    smooth_amt=7,
    f_pl_path=None,
    f_pl_name=None,
    r_core=None,
    ds=None,
    dz=None,
    rv=None,
    xywidth=None,
    z_len=None,
    tr=None,
):
    """
    Generate point-source PSFs using zernikePSF (poppy), distorted by random
    Zernike wavefront aberrations, then decompose the focal-plane field into
    LP modes using lanternfiber.

    Parameters
    ----------
    wf_npixels : int
        Pupil-plane array size passed to poppy's OpticalSystem (sets the
        resolution of the pupil wavefront stored in results["pupil_wf"]).

    Returns
    -------
    lf : lanternfiber
        Configured lanternfiber object with computed LP mode fields.
    results : dict
        Dataset containing PSFs, Zernikes, LP powers, and LP complex coefficients.
    """

    ##########################################################################
    ### Build lanternfiber and zernikePSF objects

    lf = lanternfiber.lanternfiber(
        n_core=n_core,
        n_cladding=n_cladding,
        core_radius=core_diameter / 2,
        wavelength=wavel,
    )
    lf.find_fiber_modes()
    lf.make_fiber_modes(npix=psf_npixels // 2, show_plots=False, max_r=max_r)

    # Physical pixel scale at focal plane [um/pix] must match lanternfiber mode
    # fields: microns_per_pixel = max_r * core_diameter / psf_npixels.
    # Convert to arcsec/pix for poppy: pixscale = (scale_um / f_um) * (648000/pi)
    focal_length = f_number * pupil_diameter  # [um]
    target_pixel_scale = max_r * core_diameter / psf_npixels  # [um/pix]
    pixscale_arcsec = target_pixel_scale / focal_length * (648000 / np.pi)

    zpsf = zernikePSF.zernikePSF(
        radius=pupil_diameter / 2 * 1e-6,  # aperture radius [m]
        wavelength=wavel * 1e-6,            # wavelength [m]
        pixscale=pixscale_arcsec,
        FOV_pixels=psf_npixels,
        wf_pixels=wf_npixels,
    )

    ##########################################################################
    ### Setup

    mode_numbers = list(range(len(lf.allmodefields_rsoftorder)))

    ## Initialize lists to store results
    all_zernikes = []
    all_total_coupling = []
    all_lp_powers = []
    all_lp_coeffs = []
    all_fields = []
    all_pupil_wf = []
    all_pl_outputs = []
    all_pl_powers = []

    ## Generate Zernike coefficient wavefront error RMS values
    # using Gaussian kernel smoothing

    # check if tip/tilt only (modes 2 and 3), or if including higher-order modes
    if n_zernikes == 3:
        rms_per_mode = np.concatenate(([0], [max_rms_perterm, min_rms_perterm]))
    else:
        rms_per_mode = zernike_rms_per_mode(max_rms_perterm, min_rms_perterm,
                                            n_zernikes)

    zernike_coef_array = make_smoothrand_multi(n_sims, n_zernikes,
                                               finalsds=rms_per_mode,
                                               smthamt=smooth_amt)

    ## Load LB transfer matrix for PL propagation
    transfer_matrix = load_lb_transfer_matrix(f_pl_path, f_pl_name,
                                                wavel, r_core, ds,
                                                dz, rv, xywidth,
                                                z_len, tr)

    ##########################################################################
    ### Main simulation loop
    for i in range(n_sims):

        # Zernike coefficients (converted from metres OPD to microns)
        coeffs_um = zernike_coef_array[i, :] * 1e6

        ### Propagate point source through Zernike-aberrated aperture
        zpsf.makeZernikePSF(coeffs=coeffs_um, units='microns')

        ### Focal-plane complex field E(x,y) = A(x,y) exp(i phi(x,y))
        field = zpsf.wf.amplitude * np.exp(1j * zpsf.wf.phase)

        # Poppy uses internal oversampling; downsample to psf_npixels x psf_npixels
        # so that the field pixel scale matches the lanternfiber mode fields.
        if field.shape[0] != psf_npixels:
            os = field.shape[0] // psf_npixels
            field = field.reshape(psf_npixels, os, psf_npixels, os).mean(axis=(1, 3))

        ### Pupil wavefront phase
        pupil_wf = zpsf.pupil_wf.phase

        ### Decompose focal-plane field into LP modes
        total_coupling, lp_powers, lp_coeffs = lf.calc_injection_multi(
            input_field=field,
            mode_field_numbers=mode_numbers,
            show_plots=False,
            return_abspower=True,
            complex=True,
        )

        ## Calculate PL outputs if desired
        pl_outputs = transfer_matrix.T @ lp_coeffs
        pl_powers = np.abs(pl_outputs) ** 2

        ## Store results
        all_zernikes.append(np.array(zernike_coef_array[i, :]))
        all_total_coupling.append(np.array(total_coupling))
        all_lp_powers.append(np.array(lp_powers))
        all_lp_coeffs.append(np.array(lp_coeffs))
        all_fields.append(np.array(field))
        all_pupil_wf.append(np.array(pupil_wf))
        all_pl_outputs.append(np.array(pl_outputs))
        all_pl_powers.append(np.array(pl_powers))

    ## Combine outputs
    results = {
        "zernike_coeffs": np.array(all_zernikes),
        "total_coupling": np.array(all_total_coupling),
        "lp_powers": np.array(all_lp_powers),
        "lp_coeffs": np.array(all_lp_coeffs),
        "modelabels": np.array(lf.modelabels, dtype=object),
        "nmodes": lf.nmodes,
        "microns_per_pixel": lf.microns_per_pixel,
        "psf_fields": np.array(all_fields),
        "pupil_wf": np.array(all_pupil_wf),
        "pl_outputs": np.array(all_pl_outputs),
        "pl_powers": np.array(all_pl_powers),
    }

    return lf, results




#%%###########################################################################
def source2pl_kolmogorov_temporal(
    phase_screens,
    wavel=1.55,
    f_number=4.5,
    pupil_diameter=1.8,
    n_core=1.4440,
    n_cladding=1.4400,
    core_diameter=18.0,
    max_r=3.0,
    wf_npixels=None,
    psf_npixels=256,
    f_pl_path=None,
    f_pl_name=None,
    r_core=None,
    ds=None,
    dz=None,
    rv=None,
    xywidth=None,
    z_len=None,
    tr=None,
):
    """
    Generate PSFs from preloaded Kolmogorov phase screens via Fraunhofer
    propagation (poppy), then decompose the focal-plane field into LP modes
    using lanternfiber.

    Parameters
    ----------
    phase_screens : ndarray, shape (n_sims, N, N)
        Preloaded Kolmogorov phase screens in radians. Each N x N screen is
        assumed to span the full pupil aperture diameter across N pixels.
    wf_npixels : int
        Pupil-plane array size passed to poppy's OpticalSystem.

    Returns
    -------
    lf : lanternfiber
        Configured lanternfiber object with computed LP mode fields.
    results : dict
        Dataset containing PSF fields, pupil wavefronts, LP powers, LP complex
        coefficients, and PL outputs.
    """

    n_sims = phase_screens.shape[0]

    ##########################################################################
    ### Build lanternfiber and zernikePSF objects

    lf = lanternfiber.lanternfiber(
        n_core=n_core,
        n_cladding=n_cladding,
        core_radius=core_diameter / 2,
        wavelength=wavel,
    )
    lf.find_fiber_modes()
    lf.make_fiber_modes(npix=psf_npixels // 2, show_plots=False, max_r=max_r)

    # Physical pixel scale at focal plane [um/pix] must match lanternfiber mode
    # fields: microns_per_pixel = max_r * core_diameter / psf_npixels.
    # Convert to arcsec/pix for poppy: pixscale = (scale_um / f_um) * (648000/pi)
    focal_length = f_number * pupil_diameter  # [um]
    target_pixel_scale = max_r * core_diameter / psf_npixels  # [um/pix]
    pixscale_arcsec = target_pixel_scale / focal_length * (648000 / np.pi)

    zpsf = zernikePSF.zernikePSF(
        radius=pupil_diameter / 2 * 1e-6,  # aperture radius [m]
        wavelength=wavel * 1e-6,            # wavelength [m]
        pixscale=pixscale_arcsec,
        FOV_pixels=psf_npixels,
        wf_pixels=wf_npixels,
    )

    ##########################################################################
    ### Setup

    mode_numbers = list(range(len(lf.allmodefields_rsoftorder)))

    ## Initialize lists to store results
    all_total_coupling = []
    all_lp_powers = []
    all_lp_coeffs = []
    all_fields = []
    all_pupil_wf = []
    all_pl_outputs = []
    all_pl_powers = []

    ## Load LB transfer matrix for PL propagation
    transfer_matrix = load_lb_transfer_matrix(f_pl_path, f_pl_name,
                                              wavel, r_core, ds,
                                              dz, rv, xywidth,
                                              z_len, tr)

    ##########################################################################
    ### Main simulation loop
    for i in range(n_sims):

        ### Propagate point source through Kolmogorov phase screen aperture
        zpsf.makePhaseScreenPSF(phase_screens[i])

        ### Focal-plane complex field E(x,y) = A(x,y) exp(i phi(x,y))
        field = zpsf.wf.amplitude * np.exp(1j * zpsf.wf.phase)

        # Poppy uses internal oversampling; downsample to psf_npixels x psf_npixels
        # so that the field pixel scale matches the lanternfiber mode fields.
        if field.shape[0] != psf_npixels:
            os = field.shape[0] // psf_npixels
            field = field.reshape(psf_npixels, os, psf_npixels, os).mean(axis=(1, 3))

        ### Pupil wavefront phase
        pupil_wf = zpsf.pupil_wf.phase

        ### Decompose focal-plane field into LP modes
        total_coupling, lp_powers, lp_coeffs = lf.calc_injection_multi(
            input_field=field,
            mode_field_numbers=mode_numbers,
            show_plots=False,
            return_abspower=True,
            complex=True,
        )

        ## Calculate PL outputs
        pl_outputs = transfer_matrix.T @ lp_coeffs
        pl_powers = np.abs(pl_outputs) ** 2

        ## Store results
        all_total_coupling.append(np.array(total_coupling))
        all_lp_powers.append(np.array(lp_powers))
        all_lp_coeffs.append(np.array(lp_coeffs))
        all_fields.append(np.array(field))
        all_pupil_wf.append(np.array(pupil_wf))
        all_pl_outputs.append(np.array(pl_outputs))
        all_pl_powers.append(np.array(pl_powers))

    ## Combine outputs
    results = {
        "total_coupling": np.array(all_total_coupling),
        "lp_powers": np.array(all_lp_powers),
        "lp_coeffs": np.array(all_lp_coeffs),
        "modelabels": np.array(lf.modelabels, dtype=object),
        "nmodes": lf.nmodes,
        "microns_per_pixel": lf.microns_per_pixel,
        "psf_fields": np.array(all_fields),
        "pupil_wf": np.array(all_pupil_wf),
        "pl_outputs": np.array(all_pl_outputs),
        "pl_powers": np.array(all_pl_powers),
    }

    return lf, results



# ##########################################################################
# def wf_psf_pl_prop_temporal_ss(
#     n_sims=100,
#     wavel=1.55,
#     f_number=4.5,
#     pupil_diameter=1.8,
#     n_core=1.4440,
#     n_cladding=1.4400,
#     core_diameter=18.0,
#     max_r=3.0,
#     wf_npixels=512,
#     psf_npixels=256,
#     n_zernikes=30,
#     max_rms_perterm=0.4,
#     min_rms_perterm=0.05,
#     smooth_amt=7,
#     return_wf=True,
#     calc_pl_outputs=True,
#     f_pl_path=None,
#     f_pl_name=None,
#     r_core=None, 
#     ds=None,
#     dz=None, 
#     rv=None, 
#     xywidth=None,
#     z_len=None, 
#     tr=None,
# ):
#     """
#     Generate point-source PSFs using SeidrSim, distorted by random Zernike
#     wavefront aberrations, then decompose the focal-plane field into LP modes.

#     Returns
#     -------
#     sim : SeidrSim
#         Configured simulation object.
#     results : dict
#         Dataset containing PSFs, Zernikes, LP powers, and LP complex coefficients.
#     """

#     ##########################################################################
#     ### Build SeidrSim object
#     sim = SeidrSim(
#         wavel=wavel,
#         n_core=n_core,
#         n_cladding=n_cladding,
#         core_diameter=core_diameter,
#         max_r=max_r,
#         wf_npixels=wf_npixels,
#         psf_npixels=psf_npixels,
#         n_zernikes=n_zernikes,
#         f_number=f_number,
#         pupil_diameter=pupil_diameter,
#     )

#     ##########################################################################
#     ### Setup

#     # LP modes created internally by sim.lf
#     mode_numbers = list(range(len(sim.lf.allmodefields_rsoftorder)))

#     # ## Set up random-number generator
#     # key = jr.PRNGKey(seed)

#     ## Initialize lists to store results
#     all_zernikes = []
#     all_total_coupling = []
#     all_lp_powers = []
#     all_lp_coeffs = []
#     all_fields = []
#     all_pupil_wf = [] if return_wf else None
#     all_pl_outputs = [] if calc_pl_outputs else None

#     ## Generate Zernike coefficient wavefront error RMS values 
#     # using Gaussian kernel smoothing
#     rms_per_mode = zernike_rms_per_mode(max_rms_perterm, min_rms_perterm, 
#                                          n_zernikes)

#     zernike_coef_array = make_smoothrand_multi(n_sims, n_zernikes, 
#                                                 finalsds=rms_per_mode, 
#                                                 smthamt=smooth_amt)
    
#     ## Load LB transfer matrix for PL propagation
#     if calc_pl_outputs:
#         transfer_matrix = load_lb_transfer_matrix(f_pl_path, f_pl_name,
#                                                     wavel, r_core, ds, 
#                                                     dz, rv, xywidth, 
#                                                     z_len, tr)


#     ##########################################################################
#     ### Main simulation loop
#     for i in range(n_sims):

#         # split into three random numbers
#         # key, key_tiptilt, key_ho = jr.split(key, 3)

#         ### Zernike coefficient vector -> ADD RANDOM WALK FROM BN
#         # z[0]      = piston, set to zero
#         # z[1:3]    = tip/tilt
#         # z[3:]     = higher-order aberrations
#         # zernike_coeffs = jnp.concatenate([
#         #     jnp.zeros((1,)),
#         #     tiptilt_rms * jr.normal(key_tiptilt, (2,)),
#         #     ho_rms * jr.normal(key_ho, (n_zernikes - 3,)),
#         # ])

#         ### Apply Zernike aberration to pupil-plane wavefront
#         sim.optics = sim.optics.set(
#             "aperture.coefficients",
#             zernike_coef_array[i,:]
#         )

#         pupil_wf = sim.make_pupil_wavefront(
#             np.array(zernike_coef_array[i,:]),
#             return_phase=True,
#         )

#         ### Propagate point source through optics to focal plane
#         # Returns the complex focal-plane field:
#         # E(x, y) = A(x, y) exp(i phi(x, y))
#         field = sim.propagate_wf()

#         ### Decompose focal-plane field into LP modes
#         total_coupling, lp_powers, lp_coeffs = sim.lf.calc_injection_multi(
#             input_field=field,
#             mode_field_numbers=mode_numbers,
#             show_plots=False,
#             return_abspower=True,
#             complex=True,
#         )

#         ## Calculate PL outputs if desired
#         if calc_pl_outputs:
#             pl_outputs = transfer_matrix.T @ lp_coeffs

#         ## Convert to NumPy arrays and store result
#         all_zernikes.append(np.array(zernike_coef_array[i,:]))
#         all_total_coupling.append(np.array(total_coupling))
#         all_lp_powers.append(np.array(lp_powers))
#         all_lp_coeffs.append(np.array(lp_coeffs))
#         all_fields.append(np.array(field))

#         if return_wf:
#             all_pupil_wf.append(np.array(pupil_wf))
        
#         if calc_pl_outputs:
#             all_pl_outputs.append(np.array(pl_outputs))

#     ## Reset to base, unaberrated state
#     sim.remove_aberrations()

#     ## Combine outputs
#     results = {
#         "zernike_coeffs": np.array(all_zernikes),
#         "total_coupling": np.array(all_total_coupling),
#         "lp_powers": np.array(all_lp_powers),
#         "lp_coeffs": np.array(all_lp_coeffs),
#         "modelabels": np.array(sim.lf.modelabels, dtype=object),
#         "nmodes": sim.lf.nmodes,
#         "microns_per_pixel": sim.lf.microns_per_pixel,
#         "psf_fields": np.array(all_fields),
#     }
    
#     if return_wf:
#         results["pupil_wf"] = np.array(all_pupil_wf)

#     if calc_pl_outputs:
#         results["pl_outputs"] = np.array(all_pl_outputs)

#     return sim, results


