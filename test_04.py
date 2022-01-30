from directorios import *
from procesos import *

if __name__ == '__main__':
    datos_path = 'F:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    [X, T, STROBO] = cargar_txt(carpeta, '', X='X', T='T', STROBO_MEAN='STROBO_MEAN')
    print('Datos cargados')
    analytic_signal = hilbert(STROBO[0, :])
    amplitude_envelope = np.abs(analytic_signal)
    X = X - X[-1]/2
    def func(x, a, b, c, d):
        return a * np.exp(-b * (x - d) ** 2) + c

    popt, pcov = curve_fit(func, X, amplitude_envelope)
    plt.plot(X, func(X, *popt), 'g--')
    plt.show()