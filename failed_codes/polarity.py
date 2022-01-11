import numpy as np
from scipy.interpolate import interp1d
import random
from procesos import *
from visualizacion import *

if __name__ == '__main__':
    datos_path = 'D:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    #Z = zero_fix(carpeta, 20, 'mean')
    [X, T, Z] = cargar_txt(carpeta, '', X='X', T='T', Z='Z')
    Z_polar = np.ones((len(Z[:, 0]), len(Z[0, :])))
    for i in range(len(T)):
        for j in range(len(X)):
            Z_polar[i, j] = np.sign(Z[i, j])
            #if abs(Z[i, j]) < 1:
            #    Z_polar[i, j] = 0
            #else:
            #    Z_polar[i, j] = np.sign(Z[i, j])
    polarity = Z_polar
    #visualizacion(X, T, Z_polar, tipo='colormap', guardar='no', path=carpeta,
    #              file='', nombre='strobo_plot_2_filt2', cmap='seismic')
    #plt.show()
    l = random.sample(range(300, 700), 30)
    for k in l:
        for i in range(len(T)):
            polarity[i, :] = Z_polar[i, :] * Z_polar[i, k]
    POL = filtro_superficie(polarity, 50, 'Y')
    POLAR = np.ones((len(Z[:, 0]), len(Z[0, :])))
    for i in range(len(T)):
        for j in range(len(X)):
            POLAR[i, j] = np.sign(POL[i, j])
    guardar_txt(carpeta, '', X=X, polarity=POLAR)
    visualizacion(X, T, Z_polar, tipo='colormap', guardar='si', path=carpeta,
                  file='', nombre='polarity_filt', cmap='seismic')
    plt.close()
    visualizacion(X, T, polarity, tipo='colormap', guardar='si', path=carpeta,
                  file='', nombre='polarity_filt', cmap='seismic')
    plt.close()
    visualizacion(X, T, POL, tipo='colormap', guardar='si', path=carpeta,
                  file='', nombre='polarity_filt', cmap='seismic')
    plt.close()
    visualizacion(X, T, POLAR, tipo='colormap', guardar='si', path=carpeta,
                  file='', nombre='polarity_final', cmap='seismic')
    plt.close()




    #ALT2
    #test_01 = np.ones(len(Z[0, :]))
    #test_02 = np.ones(len(Z[0, :]))
    #for j in range(len(X)):
    #    test_01[j] = Z_polar[0, j] * Z_polar[3, j]
    #    test_02[j] = Z_polar[4, j] * Z_polar[3, j]
    #suma1 = 0
    #for j in range(len(X)):
    #    suma1 = suma1 + test_01[j]
    #suma2 = 0
    #for j in range(len(X)):
    #    suma2 = suma2 + test_02[j]
    #print(np.sign(suma1))
    #print(np.sign(suma2))
    #plt.plot(test_01[0, :])
    #plt.plot(test_02[0, :])
    #plt.show()


    #ALT2
    #test_03 = np.ones((len(Z[0, :])))
    #polarity = np.ones((len(Z[:, 0]), len(Z[0, :])))
    #polarity[0, :] = Z_polar[0, :]
    #plt.plot(polarity[0, :])
    #plt.plot(Z_polar[1, :])
    #plt.show()
    #for i in range(0, len(T) - 1):
    #    current_line = polarity[i, :]
    #    print(i)
    #    n = 0
    #    for j in range(len(X)):
    #        if current_line[j] == Z_polar[i + 1, j]:
    #            n = n + 1
    #    if n > 800: #son iguales
    #        polarity[i + 1, :] = Z_polar[i + 1, :]
    #    else:
    #        polarity[i + 1, :] = -Z_polar[i + 1, :]
    #visualizacion(X, T, polarity, tipo='colormap', guardar='no', path=carpeta,
    #              file='', nombre='strobo_plot_2_filt2', cmap='seismic')
    #plt.show()

