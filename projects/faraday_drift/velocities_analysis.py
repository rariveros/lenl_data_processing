from procesos import *
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':

    ###   OPEN FILES   ###
    print('Abriendo archivos...')
    datos_path = 'E:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    x = []
    y = []
    y_err = []

    onlyfiles = [f for f in listdir(carpeta) if isfile(join(carpeta, f))]
    for i in range(len(onlyfiles)):
        current_csv = carpeta + '/' + onlyfiles[i]
        info_dvelocity = genfromtxt(current_csv, delimiter=',')
        ampd = info_dvelocity[0, 0]
        Gamma = (((14.8 / 2) / (2 * np.pi)) ** 2) * (ampd / 12) * 0.5
        mean = np.mean(info_dvelocity[:, 2])
        std = np.std(info_dvelocity[:, 2])
        x.append(Gamma)
        y.append(-mean)
        y_err.append(std)
    x_np = np.array(x)
    y_np = np.array(y)
    yerr_np = np.array(y_err)
    np.savetxt(carpeta + '/velocity_processed/gammas.csv', x_np, delimiter='.')
    np.savetxt(carpeta + '/velocity_processed/velocities.csv', y_np, delimiter='.')
    np.savetxt(carpeta + '/velocity_processed/velocities_err.csv', yerr_np, delimiter='.')
    plt.grid()
    plt.xlim([x[0] - (x[0] - x[1]) / 2, x[-1] + (x[0] - x[1]) / 2])
    plt.xlabel('$\Gamma_0$', size=15)
    plt.ylabel('$\langle v \\rangle$',size=15)
    plt.errorbar(x, y, yerr=y_err, marker='o', ls='', capsize=5, capthick=1, ecolor='black', color='red')
    plt.savefig(carpeta + '/velocity_processed/plot')
    plt.show()
    plt.close()