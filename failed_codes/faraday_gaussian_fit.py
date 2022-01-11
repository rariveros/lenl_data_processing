
from procesos import *
from visualizacion import *

if __name__ == '__main__':
    [X, T, PHI] = cargar_txt(experimental_data_file, '\\07-05-2021\\f=15.22_a=6.0', X='X', T='T', PHI='PHI')
    img_file = 'E:\mnustes_science\images\img_lab\\07-05-2021\\f=15.22_a=6.0'

    ########## PROCESOS Y OTROS #########
    X = np.array([i - X[-1] / 2 for i in X])
    PHI_filtrado = filtro_superficie(PHI, 10, 'YX')
    mean, std = proyeccion_desvesta(PHI_filtrado)
    PHI_filtrado = nivel(PHI_filtrado, mean)
    PHI_proyectado, frec, power_density = proyeccion_maximos(PHI_filtrado)
    PHI_std = (PHI_proyectado[np.argmax(PHI_proyectado)] / std[np.argmax(std)]) * std

    #################### CONVERSION A CM ##################
    [X, PHI_proyectado, PHI_std] = resize_arrays_referenced(4, [X, PHI_proyectado, PHI_std], img_file)
    sinegauss_fit, parametros = fit_gauss_sin(X, PHI_std)
    gaussian_fit = gauss(X, parametros[0], parametros[1])
    visualizacion(X, [PHI_std, sinegauss_fit, gaussian_fit], ['r', 'grey', 'black'], ['-', ':', '--'],
                  tipo="2D_multiple", guardar='si',
                  path=experimental_data_file, file='\\07-05-2021\\f=15.22_a=6.0\\datos_procesados',
                  nombre='plot')
    guardar_txt(experimental_data_file, '\\07-05-2021\\f=15.22_a=6.0\\datos_procesados', X=X, parametros=parametros, PHI_std=PHI_std, sinegauss_fit=sinegauss_fit, gaussian_fit=gaussian_fit)
