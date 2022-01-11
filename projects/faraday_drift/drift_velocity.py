from procesos import *
from visualizacion import *
from numpy import genfromtxt
from scipy.fft import fft, fftfreq, fftshift

if __name__ == '__main__':
    ###   OPEN FILES   ###
    datos_path = 'E:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    T_per = genfromtxt(carpeta + '\\' + 'T_per.csv', delimiter=',')
    X_mm = genfromtxt(carpeta + '\\' + 'X_mm.csv', delimiter=',')
    Z_mm = genfromtxt(carpeta + '\\' + 'Z_mm.csv', delimiter=',')

    ###   DEFINIENDO COSAS, VENTANA INICIAL E INTERVALO TEMPORAL A ANALIZAR  ###
    T = np.arange(len(T_per))
    X = np.arange(len(X_mm))

    window_l = 524
    window_u = 576
    t_inicial = 10000
    t_final = 30000
    L_wind = window_u - window_l

    ###   ENCONTRANDO MAXIMOS   ###
    i_array = []
    j_array = []
    tx_array = []
    x_array = []
    for i in range(t_inicial, t_final):
        j = window_l + np.argmax(Z_mm[i, window_l:window_u])
        tx_array.append(i)
        x_array.append(X_mm[j])
        window_l = int(j - L_wind / 2)
        window_u = int(j + L_wind / 2)
    tx_np = np.array(tx_array)
    x_np = np.array(x_array)

    ###   REGRESIÓN LINEAL   ###
    linear_fit = linregress(tx_array, x_array)
    x_fit = linear_fit.slope * tx_np + linear_fit.intercept
    movingframe_oscilation = x_np - x_fit

    ###  FREQUENCY STRANGE INSTABILITY###

    N = int(len(tx_np))
    T = 1/400
    yf = fft(movingframe_oscilation)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    yplot = fftshift(yf)

    xf_positive = xf[int(len(xf)/2):-1]
    yplot_positive = yplot[int(len(yplot) / 2):-1]
    max_freq_index = np.argmax(yplot_positive)
    max_freq = xf_positive[max_freq_index + 1]

    print('Frequency = ' + str(max_freq) + ' Hz\nMax_intensity = ' + str(1.0/N * np.abs(yplot_positive[max_freq_index + 1])))
    textstr = '$\omega_{max}$ = ' + str(max_freq) + ' Hz'
    plt.text(1, 0.8, textstr, fontsize=15)
    plt.plot(xf_positive, 1.0 / N * np.abs(yplot_positive))
    plt.xlim([0, 2])
    plt.ylim([0, 1.0/N * np.abs(yplot_positive[max_freq_index + 1]) + 0.5])
    plt.title('FFT', size=15)
    plt.xlabel('$\omega$ (Hz)', size=12)
    plt.ylabel('$\\tilde{x}(\omega)$ (mm)', size=12)
    plt.grid()
    #plt.show()


    ###   PLOTS   ###
    #plt.plot(tx_array, movingframe_oscilation, linewidth=0.5, c='r')
    #plt.xlim([tx_array[0],tx_array[-1]])
    #plt.title('Oscillations on Drift moving frame', size=15)
    #plt.xlabel('$t$ (s)', size=12)
    #plt.ylabel('$x$ (mm)', size=12)
    #plt.grid()
    #plot = plt.pcolormesh(X_mm, T_per, Z_mm, cmap='Reds', shading='auto')
    #cbar = plt.colorbar(plot, shrink=1)
    plt.show()