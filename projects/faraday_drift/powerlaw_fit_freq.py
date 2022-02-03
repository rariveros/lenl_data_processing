from procesos import *

if __name__ == '__main__':
    import scipy.optimize as so
    import numpy as np
    from lmfit import Model, Parameters


    def f_freq(x, A, c):
        return A * (x - c) ** 0.5
    def f(x, A, c, noise):
        return A * (((x - c) + ((x - c) ** 2 + 2 * noise) ** 0.5) / 2) ** 0.5

    gammas_np = genfromtxt('E:/mnustes_science/experimental_data/faraday_drift_03/dvelocities_info/velocity_processed/gammas.csv', delimiter=',')
    velocities_np = genfromtxt('E:/mnustes_science/experimental_data/faraday_drift_03/dvelocities_info/velocity_processed/velocities.csv', delimiter=',')
    velocities_error_np = genfromtxt('E:/mnustes_science/experimental_data/faraday_drift_03/dvelocities_info/velocity_processed/velocities_err.csv',delimiter=',')
    freq_np = genfromtxt(
        'E:/mnustes_science/experimental_data/faraday_drift_03/dvelocities_info/freq_processed/freq.csv',
        delimiter=',')
    freq_error_np = genfromtxt(
        'E:/mnustes_science/experimental_data/faraday_drift_03/dvelocities_info/freq_processed/freq_err.csv',
        delimiter=',')


    popt, pcov = curve_fit(f, gammas_np, velocities_np, bounds=[(0, 0.6, 0), (10, 0.683, 10)])
    A = popt[0]
    c = popt[1]
    noise = popt[2]

    x_grid = np.arange(0, 1, 0.001)
    velocity_noisy_fitted = []
    for i in range(len(x_grid)):
        epsilon_i = x_grid[i] - c
        velocity_noisy_fitted_i = A * ((epsilon_i + (epsilon_i ** 2 + 2 * noise) ** 0.5) / 2) ** 0.5
        velocity_noisy_fitted.append(velocity_noisy_fitted_i)
    velocity_noisy_fitted_np = np.array(velocity_noisy_fitted)

    x_grid_antierror = np.arange(c, 1, 0.001)
    velocity_fitted = []
    for i in range(len(x_grid_antierror)):
        epsilon_i = x_grid_antierror[i] - c
        velocity_fitted_i = A * (epsilon_i) ** 0.5
        velocity_fitted.append(velocity_fitted_i)
    velocity_fitted_np = np.array(velocity_fitted)

    popt, pcov = curve_fit(f_freq, gammas_np, freq_np, bounds=[(0, 0), (100, 0.61)])
    A_freq = popt[0]
    c_freq = popt[1]

    x_grid_antierror_freq = np.arange(c_freq, 1, 0.001)
    freq_fitted = []
    for i in range(len(x_grid_antierror_freq)):
        epsilon_i = x_grid_antierror_freq[i] - c_freq
        freq_fitted_i = A_freq * (epsilon_i) ** 0.5
        freq_fitted.append(freq_fitted_i)
    freq_fitted_np = np.array(freq_fitted)

    px = 1/plt.rcParams['figure.dpi']
    fig, ax1 = plt.subplots(figsize=(875*px, 750*px))
    ax1.set_xlabel('$\Gamma_0$', fontsize=20)
    ax1.set_ylabel('$\langle v \\rangle$ (mm/s)', fontsize=20)
    ax1.errorbar(gammas_np, velocities_np, yerr=velocities_error_np, marker='o', ls='', capsize=5, capthick=1,
                 ecolor='k', color='k', zorder=3)
    ax1.plot(x_grid_antierror, velocity_fitted_np, '--', linewidth='2', c='r', label='Noise included', zorder=1)
    ax1.plot(x_grid, velocity_noisy_fitted_np, '-', linewidth='2', c='r', label='No noise', zorder=2)
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0.61, 0.785])
    plt.setp(ax1.get_xticklabels(), fontsize=12)
    plt.setp(ax1.get_yticklabels(), fontsize=12)
    ax1.grid(True)

    left, bottom, width, height = [0.22, 0.5, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(x_grid_antierror_freq, freq_fitted_np, '-', linewidth='2', c='r', label='No noise', zorder=2)
    ax2.errorbar(gammas_np, freq_np, yerr=freq_error_np, marker='o', ls='', lw=1, ms=4, capsize=2, capthick=1,
                 ecolor='k', color='k', zorder=3)
    ax2.set_xlim([0.61, 0.785])
    ax2.set_ylim([0, 0.35])
    ax2.grid(True)
    ax2.set_xlabel('$\Gamma_0$', fontsize=15)
    ax2.set_ylabel('$\langle f \\rangle$', fontsize=15)
    plt.savefig('E:/mnustes_science/experimental_data/faraday_drift_03/dvelocities_info/velocity_processed/fit')
    plt.show()
    plt.close()