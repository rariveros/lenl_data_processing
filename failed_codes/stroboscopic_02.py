import numpy as np
from scipy.interpolate import interp1d
from procesos import *

if __name__ == '__main__':
    datos_path = 'D:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    [X, T, Z] = cargar_txt(carpeta, '', X='X', T='T', Z='Z')
    Z=np.abs(Z)
    print(len(T))
    #Z = ajuste_altura(X, T, Z, 4, 0.5)
    #Z = zero_fix(carpeta, 20, 'zero')
    I = []
    J = []
    A = []
    for i in range(2, len(T) - 2):
        for j in range(40, len(X) - 40):
            if np.sign(Z[i, j] - Z[i - 1, j]) == 1 and np.sign(Z[i + 1, j] - Z[i, j]) == -1\
                and np.sign(Z[i, j] - Z[i, j - 20]) == 1 and np.sign(Z[i, j] - Z[i, j + 20]) == 1:
                I.append(i)
                J.append(j)
                A.append([i, j])
    def takeFirst(elem):
        return elem[0]
    #A.sort(key=takeFirst)
    #visualizacion(X, T, Z, tipo='colormap', guardar='si', path=carpeta,
    #             file='', nombre='strobo_plot_2_filt2', cmap='seismic', vmin=-20, vzero=0, vmax=20)
    #plt.plot(J, I, 'o', color='black', markersize=1)
    #plt.show()
    n = 25
    XD = []
    max = [0,]
    for k in range(int(len(T) / n)):
        print(str(k) + ' de ' + str(int(len(T) / n)))
        init = n * k
        fin = n * (k + 1)
        strob = []
        for i in range(len(A)):
            if init < A[i][0] < fin:
                strob.append(Z[A[i][0], :])
        strob = np.array(strob)
        max_i, max_j = np.unravel_index(strob.argmax(), strob.shape)
        D = []
        for i in range(len(strob)):
            if 0.8 * strob[max_i, max_j] < strob[i, max_j]:
                D.append(strob[i, :])
        D = np.array(D)
        D_mean = []
        for i in range(len(X)):
            suma = 0
            for j in range(len(D[:, 0])):
                suma = suma + D[j, i]
            promedio = suma / len(D[:, 0])
            D_mean.append(promedio)
        D_mean = np.array(D_mean)
        if k != 0 and k != 1:
            XD_np = np.array(XD)
            MAX_j = np.argmax(XD_np[-1, :])
            if np.sign(D_mean[MAX_j] * XD_np[-1, MAX_j]) == -1:
                XD.append(- D_mean)
                max.append(MAX_j)
            elif np.sign(D_mean[MAX_j] * XD_np[-1, MAX_j]) == 0:
                print('hola ctm')
            else:
                XD.append(D_mean)
                max.append(MAX_j)
        else:
            XD.append(D_mean)
    XD = np.array(XD)
    #XD = filtro_superficie(XD, 2, 'Y')
    print(len(max))
    print(len(XD))
    pcm = plt.pcolormesh(X, np.arange(len(XD)), XD, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    #visualizacion(X, np.arange(len(D)), D, tipo='3D', guardar='no', path=carpeta,
    #              file='', nombre='strobo_plot_3D', cmap='seismic')
    #plt.scatter(max, np.arange(len(XD)), s=0.5, c='black')
    guardar_txt(carpeta, '', strobo=XD)
    plt.show()
