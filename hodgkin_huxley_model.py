"""
A script to simulate the action potential of a neuron using the Hodgkin-Huxley model.

author: Fabrizio Musacchio
date: Apr 11, 2024
"""
# %% IMPORTS
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a figure folder if it doesn't exist:
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% FUNCTIONS
# define the I_ext function to handle multiple intervals:
def I_ext(t, I_amp=1.0, intervals=[[5, 6], [10, 17]]):
    """ Return I_amp if t is within any of the specified intervals, else return 0. """
    for (start, end) in intervals:
        if start <= t <= end:
            return I_amp
    return 0

# Hodgkin-Huxley model differential equations:
def hodgkin_huxley(t, y, I_amp, intervals):
    V_m, m, h, n = y
    I_ext_current = I_ext(t, I_amp, intervals)
    dVmdt = (I_ext_current - g_Na * m**3 * h * (V_m - U_Na) - g_K * n**4 * (V_m - U_K) - g_L * (V_m - U_L)) / C_m
    alpha_m = 0.1 * (25 - V_m) / (np.exp((25 - V_m) / 10) - 1)
    beta_m = 4.0 * np.exp(-V_m / 18)
    alpha_h = 0.07 * np.exp(-V_m / 20)
    beta_h = 1 / (np.exp((30 - V_m) / 10) + 1)
    alpha_n = 0.01 * (10 - V_m) / (np.exp((10 - V_m) / 10) - 1)
    beta_n = 0.125 * np.exp(-V_m / 80)
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n
    return [dVmdt, dmdt, dhdt, dndt]
# %% CONSTANTS
# set the model constants:
C_m = 1.0  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conductances, in mS/cm^2
g_K = 36.0
g_L = 0.3
U_Na = 120  # 100 reversal potentials, in mV
U_K = -77.0 # -77.0
U_L = -54.387 # -54.387
# %% ESTIMATE U_REST
def n_inf(U_m):
    alpha_n = 0.01 * (10 - U_m) / (np.exp((10 - U_m) / 10) - 1)
    beta_n = 0.125 * np.exp(-U_m / 80)
    return alpha_n / (alpha_n + beta_n)

""" # constants:
g_K = 36.0  # mS/cm^2
g_L = 0.3   # mS/cm^2
U_K = -77.0 # mV
U_L = -54.387 # mV """

# assume an initial U_rest for calculating n_inf:
U_rest_guess = -65.0 # mV
n4 = n_inf(U_rest_guess)**4

# calculate U_rest:
U_rest = (g_K * n4 * U_K + g_L * U_L) / (g_K * n4 + g_L)
print(f"Estimated resting membrane potential U_rest:{U_rest} mv")
# %% FIND PROPER M0, H0, N0 VALUES
# define the range of membrane potentials:
U_m_range = np.linspace(-100, 100, 200)

# define alpha and beta functions for m, h, and n:
def alpha_m(U_m): return 0.1 * (25 - U_m) / (np.exp((25 - U_m) / 10) - 1)
def beta_m(U_m): return 4.0 * np.exp(-U_m / 18)
def alpha_h(U_m): return 0.07 * np.exp(-U_m / 20)
def beta_h(U_m): return 1 / (np.exp((30 - U_m) / 10) + 1)
def alpha_n(U_m): return 0.01 * (10 - U_m) / (np.exp((10 - U_m) / 10) - 1)
def beta_n(U_m): return 0.125 * np.exp(-U_m / 80)

# calculate the steady-state values for m, h, and n:
m_inf = [alpha_m(V) / (alpha_m(V) + beta_m(V)) for V in U_m_range]
h_inf = [alpha_h(V) / (alpha_h(V) + beta_h(V)) for V in U_m_range]
n_inf = [alpha_n(V) / (alpha_n(V) + beta_n(V)) for V in U_m_range]

# find indices where U_m is closest to U_rest mV:
U_find = U_rest 
index_zero = np.argmin(np.abs(U_m_range - U_find))
print(f"At U_m = {U_find:.2f} mV, m_inf = {m_inf[index_zero]:.4f}, h_inf = {h_inf[index_zero]:.4f}, n_inf = {n_inf[index_zero]:.4f}")

# plotting:
plt.figure(figsize=(5.5, 5))
plt.plot(U_m_range, m_inf, label='$m_\infty(U_m)$', c='r')
plt.plot(U_m_range, h_inf, label='$h_\infty(U_m)$', c='g')
plt.plot(U_m_range, n_inf, label='$n_\infty(U_m)$', c='b')
plt.axvline(x=U_find, color='gray', linestyle='--', label=f'$U_m$={U_find:.2f} mV')
# indicate and annotate the steady-state values at U_m = 0 mV
plt.plot(U_find, m_inf[index_zero], 'ro')
plt.text(U_find, m_inf[index_zero], f' {m_inf[index_zero]:.2f}', verticalalignment='bottom',
         color='red')
plt.plot(U_find, h_inf[index_zero], 'go')
plt.text(U_find, h_inf[index_zero], f' {h_inf[index_zero]:.2f}', verticalalignment='bottom',
         color='green')
plt.plot(U_find, n_inf[index_zero], 'bo')
plt.text(U_find, n_inf[index_zero], f'{n_inf[index_zero]:.2f} ', verticalalignment='bottom', 
         horizontalalignment='right', color='blue')
plt.title('Finding steady-state values of m, h, and n')
plt.xlabel('Membrane potential $U_m$ (mV)')
plt.ylabel('Steady-state value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/hodgkin_huxley_model_steady_state_values.png', dpi=300)
plt.show()
# %% SIMULATE MODEL: HEAVISIDE STEP FUNCTION
# set initial conditions: V_m, m, h, n
m0 = m_inf[index_zero] #0.05 # 0.05
h0 = h_inf[index_zero] #0.99 # 0.6
n0 = n_inf[index_zero] #0.12 # 0.32
y0 = [U_rest, m0, h0, n0]

# parameters for I_ext:
I_amp = 50   # amplitude of external current in uA/cm^2; from nA/cm^2 to uA/cm^2: 0.1, thus I_amp=100 mean 10 nA/cm^2
intervals = [[5, 7]] #[7.5, 9.5] [5, 7], [20, 22]
""" # create an interval staring a t_start for an adjustable duration and an adjustable time between intervals:
t_start = 5 # ms
t_stop = 150
t_duration = 15 # ms  #   4  4  2  15
t_off_time = 10 # ms  # 15 10 10  10
intervals = []
t = t_start
while t < t_stop:
    intervals.append([t, t + t_duration])
    t += t_duration + t_off_time """
    
# time range:
t = np.linspace(0, 200, 5000)  # 50 milliseconds, 5000 points

# solve ODE:
sol = solve_ivp(hodgkin_huxley, [t.min(), t.max()], y0, t_eval=t, args=(I_amp, intervals))

# plot results:
plt.figure(figsize=(7, 11))

# plotting membrane potential:
plt.subplot(4, 1, 1)
plt.plot(sol.t, sol.y[0], 'k', label='$U_m/t)$', lw=1.75)
plt.ylabel('membrane potential\n$U_m$ (mV)')
# if intervals is too long, cut it and add "...":
intervals_str = str(intervals)
if len(intervals_str) > 50:
    intervals_str = intervals_str[:50] + "..."
plt.title(f"Membrane potential, gating variables, external current and\nphase plane plots for " +
          f"$C_m$: {C_m}, $g_{{Na}}$: {g_Na}, $g_K$: {g_K}, $g_L$: {g_L},\n$U_{{eq,Na}}$: {U_Na}, $U_{{eq,K}}$: {U_K}, $U_{{eq,L}}$: {U_L}, " +
          f"$I_{{ext}}$: {I_amp},\nand t: {intervals_str}")
plt.axhline(y=U_rest, color='gray', linestyle='--', label='$U_{rest}$')
plt.legend(loc='upper right')
plt.ylim(-100, 125)
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plotting gating variables:
plt.subplot(4, 1, 2) 
plt.plot(sol.t, sol.y[1], 'r', label='$m$', lw=1.75)
plt.plot(sol.t, sol.y[2], 'g', label='$h$', lw=1.75)
plt.plot(sol.t, sol.y[3], 'b', label='$n$', lw=1.75)
plt.ylabel('gating variables')
plt.legend(loc='upper right')
plt.xlabel('time (ms)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plotting external current:
plt.subplot(4, 1, 3)
plt.plot(sol.t, [I_ext(time, I_amp, intervals) for time in sol.t], label='$I_{ext}(t)$', lw=1.75)
plt.ylabel('external current\n$I_{ext}$ ($\\mu A/cm^2$)')
plt.legend(loc='upper right')
plt.xlabel('time (ms)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plot U_m and m in phase space:
plt.subplot(4, 3, 10)
plt.plot(sol.y[0], sol.y[1], 'r', lw=1.75, label='trajectory')
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylabel('$m$/$h$/$n$')
plt.ylim(-0.1, 1.1)

# plot U_m and h in phase space:
plt.subplot(4, 3, 11)
plt.plot(sol.y[0], sol.y[2], 'g', lw=1.75, label='trajectory')
plt.plot(sol.y[0][0], sol.y[2][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[2][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylim(-0.1, 1.1)
#plt.ylabel('gating variable $m$')

# plot U_m and n in phase space:
plt.subplot(4, 3, 12)
plt.plot(sol.y[0], sol.y[3], 'b', lw=1.75)#label='trajectory'
plt.plot(sol.y[0][0], sol.y[3][0], 'bo', label='start', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[3][-1], 'o', c="yellow", label='end', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylim(-0.1, 1.1)
#plt.ylabel('gating variable $h$')
plt.legend(loc='upper right')

plt.tight_layout()
time_ranges_str = '_'.join([f'{time_range[0]}_{time_range[1]}' for time_range in intervals])
plt.savefig(f'figures/hodgkin_huxley_model_Cm_{C_m}_gNa_{g_Na}_gK_{g_K}_gL_{g_L}_UNa_{U_Na}_UK_{U_K}_UL_{U_L}_Iext_{I_amp}_t_{time_ranges_str}.png', dpi=300)
plt.show()

# count the voltage spikes:
idx_spikes = argrelextrema(sol.y[0], np.greater)[0]
idx_spikes = [idx for idx in idx_spikes if sol.y[0][idx] > 0]
print(f"Number of current pulse:  {len(intervals)}.\nNumber of voltage spikes: {len(idx_spikes)}")
# %% PLOT AP FOR DIFFERENT CURRENTS
# define the a set of external currents:
I_amps = [0, 1, 3, 5, 10, 15, 18, 19]
intervals = [[5, 15]]
# time range:
t = np.linspace(0, 50, 5000)  # 50 milliseconds, 5000 points

# solve ODE for each I_amp:
sols = []
for I_amp in I_amps:
    sol = solve_ivp(hodgkin_huxley, [t.min(), t.max()], y0, t_eval=t, args=(I_amp, intervals))
    sols.append(sol)

# plot results:
#plt.figure(figsize=(7, 4))
plt.figure(figsize=(7, 4.2))
plt.axhline(y=U_rest, color='gray', linestyle='--', label='$U_{rest}$', lw=1.0, zorder=3)
for i, sol in enumerate(sols):
    # for the color, use a scaling using an inverted viridis:
    color = plt.cm.viridis(1 - i / len(I_amps))
    plt.plot(sol.t, sol.y[0], label=f'$I_{{ext}} = {I_amps[i]}$', lw=1.75, c=color)
    #plt.plot(sol.t, sol.y[0], label=f'$I_{{ext}} = {I_amps[i]}$', lw=1.75)
plt.title(f"Membrane potential $U_m$ for different external currents $I_{{ext}}$\n" +
          f"$C_m$: {C_m}, $g_{{Na}}$: {g_Na}, $g_K$: {g_K}, $g_L$: {g_L},\n$U_{{Na}}$: {U_Na}, $U_K$: {U_K}, $U_L$: {U_L}," +
          f" t: {intervals}")
plt.xlabel('time (ms)')
plt.ylabel('membrane potential $U_m$ (mV)')
plt.legend()
plt.xlim(3, 60)
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/hodgkin_huxley_model_membrane_potential_currents.png', dpi=300)
plt.show()
# %% PLOT AP FOR STUDYING THE BEHAVIOR OF THE GATING VARIABLES
# define the a set of external currents:
I_amp = 19
intervals = [[6, 15]]
# time range:
t = np.linspace(0, 50, 5000)  # 50 milliseconds, 5000 points

# solve ODE for each I_amp:
sol = solve_ivp(hodgkin_huxley, [t.min(), t.max()], y0, t_eval=t, args=(I_amp, intervals))

# plot results:
plt.figure(figsize=(7, 11.0))
plt.subplot(4, 1, 1) 
#plt.axhline(y=U_rest, color='gray', linestyle='--', label='$U_{rest}$', lw=1.0, zorder=3)
plt.plot(sol.t, sol.y[0], label=f'$U_m(t)$', lw=1.75, c="k")

# mark part of the trajectory, where starts to leave U_rest for the first time until global maximum:
idx_ap_onset = np.where(sol.y[0] > np.ceil(U_rest))[0][0]
idx_max = np.argmax(sol.y[0])
#plt.plot(sol.t[0:idx_ap_onset], sol.y[0][0:idx_ap_onset], lw=1.75, c="k")
#plt.plot(sol.t[idx_ap_onset:idx_max], sol.y[0][idx_ap_onset:idx_max], '--k', lw=1.75, label='depolarization')
# shade the area from 0 to idx_ap_onset (x) and -75 to 125 (y):
plt.fill_between(sol.t[0:idx_ap_onset], -75, 135, color='gray', alpha=0.1)
plt.axvline(x=sol.t[idx_ap_onset], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_max], color='gray', linestyle='-', lw=0.75)
plt.fill_between(sol.t[idx_ap_onset:idx_max], -75, 135, color='g', alpha=0.1)
# after idx_max, plot trajectory until next crossing U_rest:
idx_zerocross_next = np.where(np.diff(np.sign(sol.y[0][idx_max:] - U_rest)))[0][0] + idx_max
#plt.plot(sol.t[idx_max:idx_zerocross_next], sol.y[0][idx_max:idx_zerocross_next], ':k', lw=1.75, label='repolarization')
plt.axvline(x=sol.t[idx_zerocross_next], color='gray', linestyle='-', lw=0.75)
plt.fill_between(sol.t[idx_max:idx_zerocross_next], -75, 135, color='orange', alpha=0.1)
# after idx_zerocross_next, find the index of global minimum:
idx_min = np.argmin(sol.y[0][idx_zerocross_next:]) + idx_zerocross_next
# find the index where sol.y[0][idx_min:]>-55:
idx_urest_next = np.where(sol.y[0][idx_min:] > np.floor(U_rest))[0][0] + idx_min
#plt.plot(sol.t[idx_zerocross_next:idx_urest_next], sol.y[0][idx_zerocross_next:idx_urest_next], '-.k', lw=1.75, label='hyperpolarization')
#plt.plot(sol.t[idx_urest_next:], sol.y[0][idx_urest_next:], lw=1.75, c="k", label=f'$U_{{rest}}$')
plt.axvline(x=sol.t[idx_urest_next], color='gray', linestyle='-', lw=0.75)
plt.fill_between(sol.t[idx_zerocross_next:idx_urest_next], -75, 135, color='purple', alpha=0.1)
plt.fill_between(sol.t[idx_urest_next:], -75, 135, color='gray', alpha=0.1)

# for each shaded area, create a dummy plot for legend:
resting           = mpatches.Patch(color='gray', alpha=0.3, label='resting potential phase')
depolarization    = mpatches.Patch(color='g', alpha=0.3, label='depolarization phase')
repolarization    = mpatches.Patch(color='orange', alpha=0.3, label='repolarization phase')
hyperpolarization = mpatches.Patch(color='purple', alpha=0.3, label='hyperpolarization phase')
line              = mlines.Line2D([], [], color='k', marker='_', markersize=20, label='$U_m(t)$')
plt.legend(handles=[line, resting, depolarization, repolarization, hyperpolarization])

plt.title(f"Membrane potential, gating variables, external current and\nphase plane plots for " +
          f"$C_m$: {C_m}, $g_{{Na}}$: {g_Na}, $g_K$: {g_K}, $g_L$: {g_L},\n$U_{{eq,Na}}$: {U_Na}, $U_{{eq,K}}$: {U_K}, $U_{{eq,L}}$: {U_L}, " +
          f"$I_{{ext}}$: {I_amp},\nand t: {intervals}")
plt.xlabel('time (ms)')
plt.ylabel('membrane potential\n$U_m$ (mV)')
plt.xlim(0, 40)
plt.ylim(-80, 130)
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plotting gating variables:
plt.subplot(4, 1, 2)
plt.plot(sol.t, sol.y[1], 'r', label='$m$', lw=1.75)
""" plt.plot(sol.t[:idx_ap_onset], sol.y[1][:idx_ap_onset], 'r', label='$m$')
plt.plot(sol.t[idx_ap_onset:idx_max], sol.y[1][idx_ap_onset:idx_max], 'r--')
plt.plot(sol.t[idx_max:idx_zerocross_next], sol.y[1][idx_max:idx_zerocross_next], 'r:')
plt.plot(sol.t[idx_zerocross_next:idx_urest_next], sol.y[1][idx_zerocross_next:idx_urest_next], 'r-.') """

plt.plot(sol.t, sol.y[2], 'g', label='$h$', lw=1.75)
""" plt.plot(sol.t[:idx_ap_onset], sol.y[2][:idx_ap_onset], 'g', label='$h$')
plt.plot(sol.t[idx_ap_onset:idx_max], sol.y[2][idx_ap_onset:idx_max], 'g--')
plt.plot(sol.t[idx_max:idx_zerocross_next], sol.y[2][idx_max:idx_zerocross_next], 'g:')
plt.plot(sol.t[idx_zerocross_next:idx_urest_next], sol.y[2][idx_zerocross_next:idx_urest_next], 'g-.') """

plt.plot(sol.t, sol.y[3], 'b', label='$n$', lw=1.75)
""" plt.plot(sol.t[:idx_ap_onset], sol.y[3][:idx_ap_onset], 'b', label='$n$')
plt.plot(sol.t[idx_ap_onset:idx_max], sol.y[3][idx_ap_onset:idx_max], 'b--')
plt.plot(sol.t[idx_max:idx_zerocross_next], sol.y[3][idx_max:idx_zerocross_next], 'b:')
plt.plot(sol.t[idx_zerocross_next:idx_urest_next], sol.y[3][idx_zerocross_next:idx_urest_next], 'b-.') """

plt.fill_between(sol.t[0:idx_ap_onset], -0.01, 1.01, color='gray', alpha=0.1)
plt.fill_between(sol.t[idx_ap_onset:idx_max], -0.01, 1.01, color='g', alpha=0.1)
plt.fill_between(sol.t[idx_max:idx_zerocross_next], -0.01, 1.01, color='orange', alpha=0.1)
plt.fill_between(sol.t[idx_zerocross_next:idx_urest_next], -0.01, 1.01, color='purple', alpha=0.1)
plt.fill_between(sol.t[idx_urest_next:], -0.01, 1.01, color='gray', alpha=0.1)
plt.axvline(x=sol.t[idx_ap_onset], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_max], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_zerocross_next], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_urest_next], color='gray', linestyle='-', lw=0.75)

plt.ylabel('gating variables')
plt.legend(loc='upper right')
plt.xlabel('time (ms)')
#plt.title(f"Gating variables")
plt.xlim(0, 40)
plt.ylim(-0.01, 1.01)
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plotting external current:
plt.subplot(4, 1, 3)
I_ext_array = [I_ext(time, I_amp, intervals) for time in sol.t]
plt.plot(sol.t, I_ext_array, label='$I_{ext}(t)$', lw=1.75)
plt.ylabel('external current\n$I_{ext}$ ($\\mu A/cm^2$)')
plt.legend(loc='upper right')
plt.xlabel('time (ms)')

plt.fill_between(sol.t[0:idx_ap_onset], -1, max(I_ext_array)+3, color='gray', alpha=0.1)
plt.fill_between(sol.t[idx_ap_onset:idx_max], -1, max(I_ext_array)+3, color='g', alpha=0.1)
plt.fill_between(sol.t[idx_max:idx_zerocross_next], -1, max(I_ext_array)+3, color='orange', alpha=0.1)
plt.fill_between(sol.t[idx_zerocross_next:idx_urest_next], -1, max(I_ext_array)+3, color='purple', alpha=0.1)
plt.fill_between(sol.t[idx_urest_next:], -1, max(I_ext_array)+3, color='gray', alpha=0.1)
plt.axvline(x=sol.t[idx_ap_onset], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_max], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_zerocross_next], color='gray', linestyle='-', lw=0.75)
plt.axvline(x=sol.t[idx_urest_next], color='gray', linestyle='-', lw=0.75)

plt.xlim(0, 40)
plt.ylim(-1, max(I_ext_array)+3)
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plot U_m and m in phase space:
plt.subplot(4, 3, 10)
plt.plot(sol.y[0], sol.y[1], 'r', label='trajectory', lw=1.75)
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylabel('$m$/$h$/$n$')
plt.ylim(-0.1, 1.1)

# plot U_m and h in phase space:
plt.subplot(4, 3, 11)
plt.plot(sol.y[0], sol.y[2], 'g',  label='trajectory', lw=1.75)
plt.plot(sol.y[0][0], sol.y[2][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[2][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylim(-0.1, 1.1)
#plt.ylabel('gating variable $m$')

# plot U_m and n in phase space:
plt.subplot(4, 3, 12)
plt.plot(sol.y[0], sol.y[3], 'b', lw=1.75)#label='trajectory'
plt.plot(sol.y[0][0], sol.y[3][0], 'bo', label='start', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[3][-1], 'o', c="yellow", label='end', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylim(-0.1, 1.1)
#plt.ylabel('gating variable $h$')
plt.legend(loc='upper right')

plt.tight_layout()
#plt.savefig('figures/hodgkin_huxley_model_membrane_potential_gating_variables.png', dpi=300)
time_ranges_str = '_'.join([f'{time_range[0]}_{time_range[1]}' for time_range in intervals])
plt.savefig(f'figures/hodgkin_huxley_model_APphase_Cm_{C_m}_gNa_{g_Na}_gK_{g_K}_gL_{g_L}_UNa_{U_Na}_UK_{U_K}_UL_{U_L}_Iext_{I_amp}_t_{time_ranges_str}.png', dpi=300)

plt.show() 
# %% CALCULATE THE FREQUENCY OF THE ACTION POTENTIAL DURING A CONSTANT CURRENT PULSE
# run the simulation for a constant current pulse, that increases linearly from 0 to I_amp_end uA/cm^2:
I_amp_end = 200
intervals = [[5, 495]]
# time range:
t = np.linspace(0, 500, 5000)  # 50 milliseconds, 5000 points

# simulate the model for different external currents:
I_amps =[]
spike_counts = []
for I_amp in range(0, I_amp_end, 10):
    sol = solve_ivp(hodgkin_huxley, [t.min(), t.max()], y0, t_eval=t, args=(I_amp, intervals))
    idx_spikes = argrelextrema(sol.y[0], np.greater)[0]
    idx_spikes = [idx for idx in idx_spikes if sol.y[0][idx] > 0]
    I_amps.append(I_amp)
    spike_counts.append(len(idx_spikes))
    
# from I_amp=30 on, we fit a sigmoid function to the data:
def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))
popt, pcov = curve_fit(sigmoid, I_amps[3:], spike_counts[3:], bounds=(0, [100., 0.1, 200]))
# print the parameters:
L, k, x0 = popt
print(f"Estimated L: {L}, k: {k}, x0: {x0}")

# generate enough x values to create a smooth plot:
x_values = np.linspace(0, 190, 400)
fitted_values = sigmoid(x_values, *popt)
    
# plot the frequency of the action potential as a function of the external current:
plt.figure(figsize=(5, 4))
plt.plot(I_amps, spike_counts, 'ko', lw=1.75, label='data points')
plt.plot(x_values, fitted_values, label='fitted sigmoid curve', color='red', lw=1.75)
plt.title(f"Firing rate vs. external current")
plt.xlabel('external current $I_{ext}$ ($\\mu A/cm^2$)')
plt.ylabel('number of spikes')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig('figures/hodgkin_huxley_model_AP_frequency.png', dpi=300)
plt.show()
# %% SIMULATE MODEL: RAMPED CURRENT INJECTION
# redefine the external current function in such a way that it ramps up linearly from 0 to I_amp 
# between t_ramp_start and t_ramp_end, remains constant at I_amp until t_off, and then drops to 0:
def I_ext_ramped(t, I_amp=10.0, t_ramp_start=5, t_ramp_end=10, t_off=20):
    """
    Calculates the external current I_ext at time t.
    I_ext ramps up linearly from 0 to I_amp between t_ramp_start and t_ramp_end,
    remains constant at I_amp until t_off, and then drops to 0.
    """
    if t < t_ramp_start or t > t_off:
        return 0
    elif t_ramp_start <= t <= t_ramp_end:
        return I_amp * (t - t_ramp_start) / (t_ramp_end - t_ramp_start)
    else:  # t_ramp_end < t <= t_off
        return I_amp

# redefine the Hodgkin-Huxley model to include the ramped external current:
def hodgkin_huxley_ramped(t, y, I_amp, t_ramp_start, t_ramp_end, t_off):
    V_m, m, h, n = y
    # continue with the rest of the model as before:
    I_ext_current = I_ext_ramped(t, I_amp, t_ramp_start, t_ramp_end, t_off)
    dVmdt = (I_ext_current - g_Na * m**3 * h * (V_m - U_Na) - g_K * n**4 * (V_m - U_K) - g_L * (V_m - U_L)) / C_m
    # continue with the rest of the model as before:
    alpha_m = 0.1 * (25 - V_m) / (np.exp((25 - V_m) / 10) - 1)
    beta_m = 4.0 * np.exp(-V_m / 18)
    alpha_h = 0.07 * np.exp(-V_m / 20)
    beta_h = 1 / (np.exp((30 - V_m) / 10) + 1)
    alpha_n = 0.01 * (10 - V_m) / (np.exp((10 - V_m) / 10) - 1)
    beta_n = 0.125 * np.exp(-V_m / 80)
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n
    return [dVmdt, dmdt, dhdt, dndt]

# parameters for I_ext (ramped):
I_amp = 19 # amplitude of external current in uA/cm^2
t_ramp_start=5 # time when the current starts to ramp up, in ms
t_ramp_end=40 # time when the current reaches its maximum, in ms
t_off=100    # time when the current drops to 0, in ms

# time range:
t = np.linspace(0, 200, 5000)  # 50 milliseconds, 5000 points

# solve ODE:
sol = solve_ivp(hodgkin_huxley_ramped, [t.min(), t.max()], y0, t_eval=t, args=(I_amp, t_ramp_start, t_ramp_end, t_off))

# plot results:
plt.figure(figsize=(7, 11))

# plotting membrane potential:
plt.subplot(4, 1, 1)
plt.plot(sol.t, sol.y[0], 'k', label='$U_m/t)$', lw=1.75)
plt.ylabel('membrane potential\n$U_m$ (mV)')
# if intervals is too long, cut it and add "...":
intervals_str = str(intervals)
if len(intervals_str) > 50:
    intervals_str = intervals_str[:50] + "..."
plt.title(f"Membrane potential, gating variables, external current and\nphase plane plots for " +
          f"$C_m$: {C_m}, $g_{{Na}}$: {g_Na}, $g_K$: {g_K}, $g_L$: {g_L},\n$U_{{eq,Na}}$: {U_Na}, $U_{{eq,K}}$: {U_K}, $U_{{eq,L}}$: {U_L}, " +
          f"$I_{{ext}}$: {I_amp},\nt$_{{ramp_start}}$: {t_ramp_start}, t$_{{ramp_end}}$: {t_ramp_end}, t$_{{off}}$: {t_off}")
plt.axhline(y=U_rest, color='gray', linestyle='--', label='$U_{rest}$')
plt.legend(loc='upper right')
plt.ylim(-100, 125)
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plotting gating variables:
plt.subplot(4, 1, 2) 
plt.plot(sol.t, sol.y[1], 'r', label='$m$', lw=1.75)
plt.plot(sol.t, sol.y[2], 'g', label='$h$', lw=1.75)
plt.plot(sol.t, sol.y[3], 'b', label='$n$', lw=1.75)
plt.ylabel('gating variables')
plt.legend(loc='upper right')
plt.xlabel('time (ms)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plotting external current:
plt.subplot(4, 1, 3)
plt.plot(sol.t, [I_ext_ramped(time, I_amp, t_ramp_start, t_ramp_end, t_off) for time in sol.t], label='$I_{ext}(t)$', lw=1.75)
plt.ylabel('external current\n$I_{ext}$ ($\\mu A/cm^2$)')
plt.legend(loc='upper right')
plt.xlabel('time (ms)')
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# plot U_m and m in phase space:
plt.subplot(4, 3, 10)
plt.plot(sol.y[0], sol.y[1], 'r', lw=1.75, label='trajectory')
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylabel('$m$/$h$/$n$')
plt.ylim(-0.1, 1.1)

# plot U_m and h in phase space:
plt.subplot(4, 3, 11)
plt.plot(sol.y[0], sol.y[2], 'g', lw=1.75, label='trajectory')
plt.plot(sol.y[0][0], sol.y[2][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[2][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylim(-0.1, 1.1)
#plt.ylabel('gating variable $m$')

# plot U_m and n in phase space:
plt.subplot(4, 3, 12)
plt.plot(sol.y[0], sol.y[3], 'b', lw=1.75)#label='trajectory'
plt.plot(sol.y[0][0], sol.y[3][0], 'bo', label='start', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[3][-1], 'o', c="yellow", label='end', alpha=0.75, markersize=7)
plt.xlabel('$U_m$ (mV)')
plt.ylim(-0.1, 1.1)
#plt.ylabel('gating variable $h$')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'figures/hodgkin_huxley_model_ramped_Cm_{C_m}_gNa_{g_Na}_gK_{g_K}_gL_{g_L}_UNa_{U_Na}_UK_{U_K}_UL_{U_L}_Iext_{I_amp}_t_{t_ramp_start}_{t_ramp_end}_{t_off}.png', dpi=300)
plt.show()

# count the voltage spikes:
idx_spikes = argrelextrema(sol.y[0], np.greater)[0]
idx_spikes = [idx for idx in idx_spikes if sol.y[0][idx] > 0]
print(f"Number of current pulse:  {len(intervals)}.\nNumber of voltage spikes: {len(idx_spikes)}")
# %% END
