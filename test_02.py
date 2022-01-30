from directorios import *
from procesos import *


if __name__ == '__main__':
    disco = 'F'
    datos_path = str(disco) + ':\mnustes_science\experimental_data'

    root = tk.Tk()
    root.withdraw()
    processing_file = filedialog.askdirectory(parent=root,
                                             initialdir=str(disco) + ":\mnustes_science\experimental_data",
                                             title='Seleccione la carpeta para proceso')
    [X, T, strobo] = cargar_txt(processing_file, '', X='X', T='T', strobo='STROBO')

    #plt.plot(Z_abs[:, 250])

    amplitude = []
    for i in range(len(X)):
        amp_i = np.mean(strobo[:, i])
        amplitude.append(amp_i)
    amplitude_np = np.array(amplitude)
    plt.plot(amplitude_np)
    plt.xlim([X[0], X[-1]])
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(processing_file + '/amplitude.png')
    plt.show()
    plt.close()
    #GENERANDO IMAGEN DE REFERENCIA
    vmin = 0
    vmax = np.amax(strobo)
    pcm = plt.pcolormesh(X, np.arange(len(strobo[:, 0])), strobo, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(processing_file + '/STROBO.png')
    plt.show()
