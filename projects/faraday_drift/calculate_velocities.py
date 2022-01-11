from procesos import *
from visualizacion import *

if __name__ == '__main__':
    ampd = 13.5

    ###   OPEN FILES   ###
    print('Abriendo archivos...')
    datos_path = 'E:\mnustes_science\experimental_data'
    carpeta = select_directory(datos_path)
    T_per = genfromtxt(carpeta + '\\' + 'T_per.csv', delimiter=',')
    X_mm = genfromtxt(carpeta + '\\' + 'X_mm.csv', delimiter=',')
    Z_mm = genfromtxt(carpeta + '\\' + 'Z_mm.csv', delimiter=',')
    print('Archivos Cargados (=^ ◡ ^=)')

    T_0 = 17000
    T_f = 28000
    #dvelocity_input = [[215, 277, 300, 37000], [350, 398, 300, 37000], [425, 464, 300, 37000], [556, 600, 300, 37000],
    #                   [628, 692, 300, 37000], [738, 796, 300, 37000]]
    #dvelocity_input = [[655, 690, T_0, T_f], [332, 387, T_0, T_f], [433, 472, T_0, T_f], [558, 596, T_0, T_f], [766, 812, T_0, T_f]]
    #####################dvelocity_input = [[342, 398, T_0, T_f], [424, 472, T_0, T_f], [554, 610, T_0, T_f], [654, 706, T_0, T_f],
    #####################                   [760, 828, T_0, T_f]]
    #dvelocity_input = [[261, 301, T_0, T_f], [397, 429, T_0, T_f], [476, 507, T_0, T_f], [605, 639, T_0, T_f],
    #                   [688, 727, T_0, T_f]]
    #####################dvelocity_input = [[234, 290, T_0, T_f], [340, 420, T_0, T_f], [434, 480, T_0, T_f], [560, 612, T_0, T_f],
    #####################                   [660, 716, T_0, T_f]]
    #dvelocity_input = [[355, 391, T_0, T_f], [446, 490, T_0, T_f], [563, 600, T_0, T_f], [663, 697, T_0, T_f]]
    #####################dvelocity_input = [[250, 312, T_0, T_f], [340, 378, T_0, T_f], [468, 510, T_0, T_f], [544, 592, T_0, T_f],
    #####################                   [660, 706, T_0, T_f]]
    #dvelocity_input = [[262, 298, T_0, T_f], [343, 387, T_0, T_f], [481, 513, T_0, T_f], [557, 596, T_0, T_f],
    #                   [691, 732, T_0, T_f]]
    #####################dvelocity_input = [[244, 292, T_0, T_f], [370, 426, T_0, T_f], [445, 502, T_0, T_f], [580, 620, T_0, T_f],
    #####################                   [668, 710, T_0, T_f]]
    #dvelocity_input = [[257, 302, T_0, T_f], [354, 406, T_0, T_f], [463, 501, T_0, T_f], [564, 606, T_0, T_f],
    #                   [667, 708, T_0, T_f]]
    #####################dvelocity_input = [[240, 306, T_0, T_f], [370, 424, T_0, T_f], [454, 512, T_0, T_f], [569, 630, T_0, T_f],
    #####################                   [656, 720, T_0, T_f]]
    #dvelocity_input = [[282, 315, T_0, T_f], [358, 394, T_0, T_f], [491, 526, T_0, T_f], [571, 607, T_0, T_f],
    #                   [697, 741, T_0, T_f]]
    #####################dvelocity_input = [[234, 276, T_0, T_f], [344, 394, T_0, T_f], [428, 490, T_0, T_f], [546, 590, T_0, T_f],
    #####################                   [650, 688, T_0, T_f]]
    #dvelocity_input = [[347, 396, T_0, T_f], [470, 517, T_0, T_f], [553, 595, T_0, T_f], [671, 718, T_0, T_f]]
    #####################dvelocity_input = [[254, 300, T_0, T_f], [336, 402, T_0, T_f], [448, 504, T_0, T_f], [552, 588, T_0, T_f],
    #####################                   [650, 714, T_0, T_f]]
    #dvelocity_input = [[266, 301, T_0, T_f], [333, 373, T_0, T_f], [479, 520, T_0, T_f], [579, 589, T_0, T_f],
    #                   [688, 723, T_0, T_f]]
    #####################dvelocity_input = [[264, 320, T_0, T_f], [344, 398, T_0, T_f], [484, 544, T_0, T_f], [564, 610, T_0, T_f],
    #####################                   [700, 744, T_0, T_f]]
    #dvelocity_input = [[271, 308, T_0, T_f], [360, 393, T_0, T_f], [492, 529, T_0, T_f], [581, 613, T_0, T_f],
    #                   [708, 746, T_0, T_f]]
    #####################dvelocity_input = [[234, 290, T_0, T_f], [370, 422, T_0, T_f], [460, 500, T_0, T_f], [584, 632, T_0, T_f],
    #####################                   [676, 720, T_0, T_f]]
    #dvelocity_input = [[264, 306, T_0, T_f], [348, 386, T_0, T_f], [487, 526, T_0, T_f], [567, 604, T_0, T_f],
    #                   [701, 739, T_0, T_f]]
    #####################dvelocity_input = [[256, 304, T_0, T_f], [344, 398, T_0, T_f], [472, 522, T_0, T_f], [556, 602, T_0, T_f],
    #####################                   [686, 750, T_0, T_f]]
    #dvelocity_input = [[281, 315, T_0, T_f], [359, 393, T_0, T_f], [502, 532, T_0, T_f], [577, 608, T_0, T_f],
    #                   [709, 745, T_0, T_f]]
    #####################dvelocity_input = [[258, 330, T_0, T_f], [354, 402, T_0, T_f], [480, 546, T_0, T_f], [580, 630, T_0, T_f],
    #####################                   [710, 760, T_0, T_f]]
    #dvelocity_input = [[271, 315, T_0, T_f], [365, 408, T_0, T_f], [492, 531, T_0, T_f], [591, 624, T_0, T_f],
    #                   [714, 754, T_0, T_f]]
    #####################dvelocity_input = [[250, 300, T_0, T_f], [378, 424, T_0, T_f], [460, 508, T_0, T_f], [584, 628, T_0, T_f],
    #####################                   [670, 726, T_0, T_f]]
    #dvelocity_input = [[276, 304, T_0, T_f], [360, 394, T_0, T_f], [475, 517, T_0, T_f], [567, 596, T_0, T_f],
    #                   [688, 725, T_0, T_f]]
    #####################dvelocity_input = [[252, 310, T_0, T_f], [340, 408, T_0, T_f], [480, 530, T_0, T_f], [570, 616, T_0, T_f],
    #####################                   [698, 746, T_0, T_f]]
    #dvelocity_input = [[289, 322, T_0, T_f], [382, 419, T_0, T_f], [501, 530, T_0, T_f], [589, 616, T_0, T_f],
    #                   [698, 735, T_0, T_f]]
    #####################dvelocity_input = [[256, 312, T_0, T_f], [380, 426, T_0, T_f], [470, 526, T_0, T_f], [580, 634, T_0, T_f],
    #####################                   [680, 736, T_0, T_f]]
    dvelocity_input = [[301, 335, T_0, T_f], [376, 421, T_0, T_f], [516, 553, T_0, T_f], [592, 627, T_0, T_f],
                       [720, 760, T_0, T_f]]
    print('Comenzando iteración')
    info_drift = [] #
    fits = []
    for i in range(len(dvelocity_input)):
        t_np, x_np, x_fit, linear_fit, moving_frame = drift_velocity(T_per, X_mm, Z_mm, dvelocity_input[i][0], dvelocity_input[i][1], dvelocity_input[i][2], dvelocity_input[i][3])
        freq, intensity = drift_frequency(moving_frame, t_np)
        fits_i = [t_np, x_fit]
        info_drift_i = [ampd, i, linear_fit.slope, linear_fit.intercept, freq, intensity]
        info_drift.append(info_drift_i)
        fits.append(fits_i)
        print('Drift velocity ' + str(i + 1) + '/' + str(len(dvelocity_input)))
    dvelocity_input_np = np.array(dvelocity_input)
    info_drift_np = np.array(info_drift)
    os.chdir(carpeta)
    os.chdir('../')
    cwd = os.getcwd()
    dvelocities_file = cwd + '/dvelocities_info'
    inputs_file = dvelocities_file + '/inputs'
    if not os.path.exists(dvelocities_file):
        os.makedirs(dvelocities_file)
    if not os.path.exists(inputs_file):
        os.makedirs(inputs_file)
    np.savetxt(inputs_file + '\\' + 'dvelocity_input_' + str(ampd) + '.csv', info_drift, delimiter=',')
    np.savetxt(dvelocities_file + '\\' + 'info_dvelocity_' + str(ampd) + '.csv', info_drift, delimiter=',')

    plt.plot(fits[0][1], fits[0][0], linewidth=3, c='g')
    plt.plot(fits[1][1], fits[1][0], linewidth=3, c='g')
    plt.plot(fits[2][1], fits[2][0], linewidth=3, c='g')
    plt.plot(fits[3][1], fits[3][0], linewidth=3, c='g')
    plt.plot(fits[4][1], fits[4][0], linewidth=3, c='g')
    T_np = np.arange(len(T_per))
    T_sec = T_np / 400
    plot = plt.pcolormesh(X_mm, T_sec, Z_mm, cmap='Reds', shading='auto')
    cbar = plt.colorbar(plot, shrink=1)
    plt.savefig(carpeta + '\\' + 'drift_velocities')
    plt.show()