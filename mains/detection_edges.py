from procesos import *

if __name__ == '__main__':
    sigma = 'fixed'
    root = tk.Tk()
    root.withdraw()
    detection_path = 'C:/Users/mnustes_science/PT_fluids/faraday_drift/figures/fordetection'
    canned_path = 'C:/Users/mnustes_science/PT_fluids/faraday_drift/figures/canned'
    img_path = 'C:/Users/mnustes_science/PT_fluids/faraday_drift/figures/img'
    datos_path = 'C:/Users/mnustes_science/PT_fluids/faraday_drift/figures/data'
    plot_path = 'C:/Users/mnustes_science/PT_fluids/faraday_drift/figures/plots'
    reference_image = filedialog.askopenfilename(parent=root, initialdir=detection_path)
    corte_coords = ROI_select(reference_image)
    rec_list = list(corte_coords)
    IMGs = os.listdir(detection_path)
    for i in range(len(IMGs)):
        im = cv2.imread(detection_path + '/' + IMGs[i])
        imCrop = im[rec_list[1]:(rec_list[1] + rec_list[3]), rec_list[0]:(rec_list[0] + rec_list[2])]
        imBlur = cv2.GaussianBlur(imCrop, (3, 3), 0)
        edges = auto_canny(imBlur, sigma)
        cv2.imwrite(img_path + '/' + str(IMGs[i]), imCrop)
        cv2.imwrite(canned_path + '/' + str(IMGs[i]), edges)
    X, T, Z = datos_3d(IMGs, canned_path)
    np.savetxt(datos_path + '/Z.txt', Z, delimiter=',')
    np.savetxt(datos_path + '/X.txt', X, delimiter=',')
    np.savetxt(datos_path + '/T.txt', T, delimiter=',')
    print('guardado listo!')
    for i in range(len(IMGs) - 1):
        img = cv2.imread(img_path + '/' + IMGs[0], 0)
        rows, cols = img.shape
        fig, ax = plt.subplots()
        x = range(len(Z[0, :]))
        im = cv2.imread(img_path + '/' + IMGs[i])
        ax.imshow(im, extent=[0, cols, 0, rows])
        ax.plot(x, Z[i, :], linewidth=1.5, color='firebrick')
        #plt.xlabel('x (pixels)')
        #plt.ylabel('$\psi(x)$ (pixels)')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
        plt.savefig(plot_path + '/' + IMGs[i], dpi=500)
        plt.close()
