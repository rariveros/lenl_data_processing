from directorios import *
from procesos import *

if __name__ == '__main__':
    datos_path = 'F:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    [X, T, Z] = cargar_txt(carpeta, '', X='X', T='T', Z='Z')
    print('Datos cargados')
    Z_abs = np.abs(Z)
    SUMAS = []
    STROBO = []
    n = 0
    for i in range(len(T)):
        SUMAS_i = np.sum(Z_abs[i, :])
        SUMAS.append(SUMAS_i)
        if i > 3 and SUMAS[-1] < SUMAS[-2] and SUMAS[-3] < SUMAS[-2]:
            strobo_i = Z[i - 1, :] * (-1) ** n
            STROBO.append(strobo_i)
            n = n + 1
    STROBO = np.array(STROBO)
    T_STROBO = np.arange(len(STROBO[:, 0]))
    STROBO_MEAN = []
    for i in range(1, len(T_STROBO), 2):
        STROBO_MEAN_i = (STROBO[i, :] + STROBO[i - 1, :]) / 2
        STROBO_MEAN.append(STROBO_MEAN_i)
    STROBO_MEAN = np.array(STROBO_MEAN)
    T_STROBO_MEAN = np.arange(len(STROBO_MEAN[:, 0]))
    guardar_txt(carpeta, 'no', STROBO_MEAN=STROBO_MEAN)
    guardar_txt(carpeta, 'no', T_STROBO_MEAN=T_STROBO_MEAN)

    #PLOT INTEGRAL ESPACIAL
    plt.plot(T, SUMAS)
    plt.xlim([T[0], T[-1]])
    plt.xlabel('$t$', size='20')
    plt.ylabel('$\int \psi(x)$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(carpeta + '/int_space.png')
    plt.show()
    plt.close()

    # PLOT STROBO NO MEAN
    vmin = -np.amax(STROBO)
    vmax = np.amax(STROBO)
    pcm = plt.pcolormesh(X, T_STROBO, STROBO, cmap='seismic', shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(carpeta + '/strobo_maxmin.png')
    plt.show()
    plt.close()

    # PLOT STROBO MEAN
    vmin = -np.amax(STROBO_MEAN)
    vmax = np.amax(STROBO_MEAN)
    pcm = plt.pcolormesh(X, T_STROBO_MEAN, STROBO_MEAN, cmap='seismic', shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(carpeta + '/strobo_mean.png')
    plt.show()
