from procesos import *

if __name__ == '__main__':

    def buscar_extremos(X, T, Z, min_max, mode):
        elementos_extremos = []
        I = []
        J = []
        m = 20
        def takeFirst(elem):
            return elem[0]
        if min_max == 'max' and mode == 'diferencia':
            for i in range(2, len(T) - 2):
                for j in range(m, len(X) - m):
                    if np.sign(Z[i, j] - Z[i - 1, j]) == 1 and np.sign(Z[i - 1, j] - Z[i - 2, j]) == 1\
                            and np.sign(Z[i, j] - Z[i + 1, j]) == 1 and np.sign(Z[i + 1, j] - Z[i + 2, j]) == 1\
                            and np.sign(Z[i, j] - Z[i, j - m]) == 1 and np.sign(Z[i, j] - Z[i, j + m]) == 1:
                        I.append(i)
                        J.append(j)
                        elementos_extremos.append([i, j])
        elif min_max == 'min' and mode == 'diferencia':
            for i in range(2, len(T) - 2):
                for j in range(m, len(X) - m):
                    if np.sign(Z[i, j] - Z[i - 1, j]) == -1 and np.sign(Z[i - 1, j] - Z[i - 2, j]) == -1\
                            and np.sign(Z[i, j] - Z[i + 1, j]) == -1 and np.sign(Z[i + 1, j] - Z[i + 2, j]) == -1\
                            and np.sign(Z[i, j] - Z[i, j - m]) == -1 and np.sign(Z[i, j] - Z[i, j + m]) == -1:
                        I.append(i)
                        J.append(j)
                        elementos_extremos.append([i, j])
        elementos_extremos.sort(key=takeFirst)
        return elementos_extremos, I, J

    def stroboscopic(X, T, Z, A, n, mode, window):
        lower_window = window[0]
        upper_window = window[1]
        st_dynamic = []
        ext = [0, ]
        for k in range(int(len(T) / n)):
            print(str(k) + ' de ' + str(int(len(T) / n)))
            init = n * k
            fin = n * (k + 1)
            strob = []
            for i in range(len(A)):
                if init < A[i][0] < fin and lower_window < A[i][1] < upper_window:
                    strob.append(Z[A[i][0], :])
                lower_window = A[i][1] - int((upper_window - lower_window) / 2)
                upper_window = A[i][1] + int((upper_window - lower_window) / 2)
            strob = np.array(strob)
            if mode == 'max':
                ext_i, ext_j = np.unravel_index(strob.argmax(), strob.shape)
            elif mode == 'min':
                ext_i, ext_j = np.unravel_index(strob.argmin(), strob.shape)
            D = []
            for i in range(len(strob)):
                if 0.8 * abs(strob[ext_i, ext_j]) < abs(strob[i, ext_j]):
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
                XD_np = np.array(st_dynamic)
                MAX_j = np.argmax(XD_np[-1, :])
                if np.sign(D_mean[MAX_j] * XD_np[-1, MAX_j]) == -1:
                    st_dynamic.append(- D_mean)
                    ext.append(MAX_j)
                elif np.sign(D_mean[MAX_j] * XD_np[-1, MAX_j]) == 0:
                    print('hola')
                else:
                    st_dynamic.append(D_mean)
                    ext.append(MAX_j)
            else:
                st_dynamic.append(D_mean)
        st_dynamic_np = np.array(st_dynamic)
        return st_dynamic_np

    datos_path = 'F:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    #Z = zero_fix(carpeta, 20, 'zero')

    [X, T, Z] = cargar_txt(carpeta, '', X='X', T='T', Z='Z')

    min_elements, I, J = buscar_extremos(X, T, Z, 'max', 'diferencia')
    #visualizacion(X, T, Z, tipo='colormap', guardar='si', path=carpeta,
    #              file='', nombre='strobo_plot_2_filt2', cmap='seismic', vmin=-20, vzero=0, vmax=20)
    #plt.plot(J, I, 'o', color='black', markersize=1)
    #plt.show()

    upper_window = 700
    lower_window = 850
    STROBO = stroboscopic(X, T, Z, min_elements, 80, 'min', [lower_window, upper_window])
    pcm = plt.pcolormesh(X, np.arange(len(STROBO[:, 0])), STROBO, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
