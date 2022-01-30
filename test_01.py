from directorios import *
from procesos import *


if __name__ == '__main__':
    #PARAMETROS INICIALES
    zero = 'no'
    sigma = 'fixed'
    project_file = 'ayudantia_2022/4_triangulos/bifurcacion_secador'

    #DETECCION DE CONTORNOS
    X, T, PHI, carpeta = deteccion_contornos_new('F', zero, sigma, file_name=project_file)

    #NIVELACION DEL CERO
    Z = nivel_mean(PHI, X, T)
    Z = np.array(Z)
    guardar_txt(carpeta, 'no', Z=Z)

    #APLICANDO FILTRO (SUAVE)
    Z = filtro_superficie(Z, 10, 'X')
    Z = filtro_superficie(Z, 2, 'Y')
    
    #GENERANDO IMAGEN DE REFERENCIA
    vmin = -np.amax(Z)
    vmax = np.amax(Z)
    pcm = plt.pcolormesh(X, T, Z, cmap='seismic', shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(carpeta + '/module_spacetime.png')
    plt.close()