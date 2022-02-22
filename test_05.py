from directorios import *
from procesos import *

if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle('Sharing both axes')
    axs[0].plot(x, y ** 2)
    axs[1].plot(x, 0.3 * y, 'o')
    axs[2].plot(x, y, '+')

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
    plt.show()
    plt.close()