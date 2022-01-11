import numpy as np
from scipy.interpolate import interp1d
from procesos import *
from visualizacion import *

if __name__ == '__main__':

    def strob(T_sec, Z_sec, fps, force_freq, strobo, time, window_width, window_left, window_right, Bs_max):
        freq_frames = (force_freq / 2) * (1 / fps)
        period_frames = int(1 / freq_frames)
        if window_left == 'init' and window_right == 'init':
            A = Z_sec[0:period_frames, :]
            max_i, max_j = np.unravel_index(A.argmax(), A.shape)
            window_left = max_j - int(window_width / 2)
            window_right = max_j + int(window_width / 2)
            B_max_j = int(window_width / 2)
            B = Z_sec[:, window_left:window_right]
        else:
            B = Z_sec[:, window_left:window_right]
            B_max_i, B_max_j = np.unravel_index(B.argmax(), B.shape)
            max_j = window_left + B_max_j
            window_left = max_j - int(window_width / 2)
            window_right = max_j + int(window_width / 2)
            B = Z_sec[:, window_left:window_right]
        for k in range(2, len(T_sec) - 2):
            for m in range(2, window_width - 2):
                if np.sign(B[k, m] - B[k - 1, m]) == 1 and np.sign(
                        B[k - 1, m] - B[k - 2, m]) == 1 and np.sign(
                        B[k + 1, m] - B[k, m]) == -1 and np.sign(B[k + 2, m] - B[k + 1, m]) == -1\
                        and np.sign(B[k, m] - B[k, m - 1]) == 1 and np.sign(
                        B[k, m - 1] - B[k, m - 2]) == 1 and np.sign(
                        B[k, m + 1] - B[k, m]) == -1 and np.sign(B[k,m + 2] - B[k, m + 1]) == -1:
                    Bs_max.append(max_j)
                    strobo.append(Z_sec[k, :])
                    time.append(T_sec[k])
        return strobo, time, Bs_max, window_left, window_right

    datos_path = 'D:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    [X, T, Z] = cargar_txt(carpeta, '', X='X', T='T', Z='Z')
    fps = 100
    force_freq = 15.22
    strobo = []
    time = []
    Bs_max = []
    window_left = 'init'
    window_right = 'init'
    for i in range(int(len(T)/fps) - 2):
        init = fps * i
        fin = fps * (i + 1)
        strobo, time, Bs_max, window_left, window_right = strob(T[init:fin], Z[init:fin, :], fps, force_freq, strobo, time, 10, window_left, window_right, Bs_max)
    strobo = np.array(strobo)
    time = np.array(time)
    print(len(Bs_max), len(time))
    #strobo = filtro_superficie(strobo, 4, 'Y')
    visualizacion(X, np.arange(len(time)), strobo, tipo='colormap', guardar='si', path=carpeta,
                  file='', nombre='strobo_plot', cmap='seismic', vmin=-20, vzero=0, vmax=20)
    plt.scatter(Bs_max, np.arange(len(time)), s=0.5, c='black')
    plt.show()