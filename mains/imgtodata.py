from directorios import *
from procesos import *
from visualizacion import *


if __name__ == '__main__':
    sigma = 'fixed'
    project_file = 'ayudantia_2022'
    X, T, PHI, carpeta = deteccion_contornos('single_file', sigma, 'jpg', file_name=project_file)
    Z = nivel_mean(PHI, X, T)
    Z = np.array(Z)
    visualizacion(X, T, Z, tipo='colormap', guardar='si', path=carpeta,
                  file='', nombre='faraday', cmap='seismic', vmin=-20, vzero=0, vmax=20)
    guardar_txt(carpeta, 'no', Z=Z)