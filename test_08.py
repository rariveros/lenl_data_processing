from directorios import *
from procesos import *


if __name__ == "__main__":
    a = 'F:/mnustes_science/experimental_data/ayudantia_2022/4_triangulos/bifurcacion_secador/a=6.3_f=14.9'
    X = np.loadtxt(a + '/X.txt', delimiter=',')
    X_mm = np.loadtxt(a + '/X_mm.txt', delimiter=',')
    T_s = np.loadtxt(a + '/T.txt', delimiter=',')
    Z_mm = np.loadtxt(a + '/Z_mm.txt', delimiter=',')
    norm = TwoSlopeNorm(vmin=np.amin(Z_mm), vcenter=0, vmax=np.amax(Z_mm))
    pcm = plt.pcolormesh(X_mm, T_s, Z_mm, norm=norm, cmap='seismic', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X_mm[0], X_mm[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    print(T_s)
