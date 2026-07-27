#%%###########################################################################
"""
Overall plan:
- program inputs: companion properties
- fixed constants: detector, VLTI properties
- outputs: null depth distribution and hence SNR as ratio of companion light 
  to starlight
- define inputs from baldr as a distribution of zernikies
- apply arbitrary correction from PL loop
- assume first n LP modes are injected (for n=1,3)
- Correction due to kernel nuller chip
- look at overall null depth
"""

#%%###########################################################################
### Import Libraries and Modules

import numpy as np
import matplotlib.pyplot as plt


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
### Define Simulation Parameters ###

dir_plot = '/suphys/nlon0790/Documents/python_code/seidr2.0/figures/'

wavelength = 1.55e-6  # meters

## Simulation parameters
n_beams = 4  # number of beams
n_runs = 1000

## Noise properties
sigma_opd = 25e-9  # rms OPD error [m]
sigma_phi = 2 * np.pi * sigma_opd / wavelength  # rms phase error [rad]

## sigma_I = 0.10 is the literature estimate for current XAO performance at 1.55 um:
## "practical RMS coupling efficiency variations with an extreme adaptive optics
## system can be of order 10% at 1.55 um" -- Martinache & Ireland (2018), Sec 3.4,
## citing Jovanovic et al. (2017). sigma_I is applied to intensity directly (not
## amplitude) below, matching M&I2018's definition of sigma_I as "the intensity
## fluctuation on each telescope".
## The smaller values are illustrative only -- they are NOT reported in the literature --
## and represent how the kernel-output noise COULD shrink if HMS-PL-based wavefront
## correction pushes sigma_I down towards the idealized lab ExAO regime (Jovanovic et al.
## 2017 report 67 +/- 2% coupling efficiency at 90% Strehl, i.e. sigma_I ~ 0.02-0.03).
sigma_I = 0.10  # rms intensity error

sigma_I_array = np.array([0.10, 0.05, 0.02])  # current estimate, then illustrative HMS-PL improvement


#%%###########################################################################
### Setup Matrices

## Nuller response matrix
M_matrix = 0.25 * np.array(
    [
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j],
        [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j],
        [1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j],
        [1 + 1j, -1 - 1j, -1 + 1j, 1 - 1j],
    ],
    dtype=np.complex64,
)

## Kernel operator matrix (used to erase second order phase errors)
K_matrix = np.array(
    [
        [1, -1, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 1, -1],
    ],
    dtype=np.float32,
)


#%%###########################################################################
### Kernel Nulling Example Based on Input Noise 

## Generate Random Noisy Input Fields
## sigma_I perturbs intensity directly; amplitude = sqrt(intensity)
input_beam_intensity = np.random.default_rng(0).normal(size=(n_beams, n_runs)) * sigma_I + 1
input_beam_amplitude = np.sqrt(input_beam_intensity)

input_beam_phase = np.random.default_rng(1).normal(size=(n_beams, n_runs)) * sigma_phi


input_beam_field = input_beam_amplitude * np.exp(1j * input_beam_phase)

## Calculate Nuller Outputs
detector_outputs = np.abs(M_matrix @ input_beam_field) ** 2

## Calculate Kernel Nuller Outputs
kernel_outputs = K_matrix @ detector_outputs

print(np.std(kernel_outputs, axis=1))

## Plot Results
plt.figure()
plt.hist(kernel_outputs[0], 
         bins=50, alpha=0.5, 
         label="kernel 1")
plt.hist(kernel_outputs[1], 
         bins=50, alpha=0.5, 
         label="kernel 2")
plt.hist(kernel_outputs[2], 
         bins=50, alpha=0.5, 
         label="kernel 3")

plt.legend()
plt.grid(linestyle=':', linewidth=0.5)
plt.xlabel("Kernel Output")
plt.ylabel("Count")
plt.title("Kernel Output Distribution with Noise")
plt.show()


#%%###########################################################################
### Kernel Nulling Example Based on Looped Intensity Noise and Fringe-Tracking OPD

## OPD levels to compare:
## 50 nm -> Heimdallr's projected fringe-tracking residual (Taras et al. 2024)
## 25 nm -> illustrative scenario assuming a further on-chip fringe-tracking
##          improvement; NOT a literature-cited value.
sigma_opd_array = np.array([50e-9, 25e-9])  # rms OPD error [m]

## Initialise arrays
kernel_outputs_array = np.zeros((3, n_runs, len(sigma_I_array), len(sigma_opd_array)))

for j in range(len(sigma_opd_array)):
    sigma_phi_j = 2 * np.pi * sigma_opd_array[j] / wavelength

    ## Input piston errors (same realisation reused across sigma_I for this OPD)
    input_beam_phase = np.random.default_rng(1).normal(size=(n_beams, n_runs)) * sigma_phi_j

    for i in range(len(sigma_I_array)):
        sigma_I = sigma_I_array[i]

        ## Generate Random Noisy Input Fields
        ## sigma_I perturbs intensity directly; amplitude = sqrt(intensity)
        input_beam_intensity = np.random.default_rng(0).normal(size=(n_beams, n_runs)) * sigma_I + 1
        input_beam_amplitude = np.sqrt(input_beam_intensity)

        input_beam_field = input_beam_amplitude * np.exp(1j * input_beam_phase)

        ## Calculate Nuller Outputs
        detector_outputs = np.abs(M_matrix @ input_beam_field) ** 2

        ## Calculate Kernel Nuller Outputs
        kernel_outputs_array[:, :, i, j] = K_matrix @ detector_outputs


#%% Plot Results for Different Intensity Noise Levels - Subplots, one figure per OPD

for j in range(len(sigma_opd_array)):
    fig, axes = plt.subplots(1, len(sigma_I_array),
                             figsize=(5 * len(sigma_I_array), 4),
                             tight_layout=True)

    for i, ax in enumerate(axes):
        for k in range(3):
            ax.hist(kernel_outputs_array[k, :, i, j],
                    bins=50, alpha=0.5,
                    label=f"kernel {k + 1}")

        ax.grid(linestyle=':', linewidth=0.5)
        ax.set_xlabel("Kernel Output")
        ax.set_title(f"sigma_I = {sigma_I_array[i]}")

    axes[0].set_ylabel("Count")
    axes[0].legend()
    fig.suptitle(f"Kernel Output Distribution vs Intensity Noise Level (OPD = {sigma_opd_array[j]*1e9:.0f} nm)")
    plt.show()

#%% Plot Results for Different Intensity Noise Levels - Overlaid, 50 nm vs 25 nm OPD side by side

n_kernels = kernel_outputs_array.shape[0]
colors = ['tab:blue', 'tab:orange', 'tab:green']  # one per sigma_I
linestyles = ['-', '--', ':']                     # one per kernel

fig, axes = plt.subplots(1, len(sigma_opd_array), figsize=(9, 4),
                         sharey=True, sharex=True)

for j, ax in enumerate(axes):
    for i in range(len(sigma_I_array)):
        for k in range(n_kernels):
            ax.hist(kernel_outputs_array[k, :, i, j],
                    bins=50, histtype='step',
                    color=colors[i], linestyle=linestyles[k], linewidth=1)

    ax.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Kernel Output")
    ax.set_title(f"$\sigma_{{\phi}}$ = {sigma_opd_array[j]*1e9:.0f} nm")

axes[0].set_ylabel("Count")

## Build two separate legends: colour -> sigma_I, linestyle -> kernel (on the first panel only)
color_handles = [axes[0].plot([], [], color=colors[i], linestyle='-',
                         label=f"$\sigma_I$ = {sigma_I_array[i]}")[0]
                 for i in range(len(sigma_I_array))]
style_handles = [axes[0].plot([], [], color='k', linestyle=linestyles[k],
                         label=f"kernel {k + 1}")[0]
                 for k in range(n_kernels)]

legend1 = axes[1].legend(handles=color_handles, 
                         loc='upper right', 
                         fontsize=10)
axes[1].add_artist(legend1)
axes[0].legend(handles=style_handles, loc='upper left', fontsize=10)

# fig.suptitle("Kernel Output Distribution vs Intensity Noise Level, 50 nm (Heimdallr) vs 25 nm (illustrative on-chip tracking) OPD")
plt.tight_layout()
plt.savefig(dir_plot + "kernel_output_distribution_vs_intensity_noise.svg", dpi=150)
plt.show()
# %%
