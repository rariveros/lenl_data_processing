from directorios import *
from procesos import *


if __name__ == '__main__':
    ### Definiendo parametros y eligiendo carpeta a detectar ###
    file = 'C:/Users/rariv/Desktop/test'
    IMG_names = os.listdir(file)
    N_img = len(IMG_names)
    resize_scale = 0.5
    thresh = 48

    ### Abriendo imagen de referencia para determinar la región de interes (ROI) y la conversion a mm ###
    root = tk.Tk()
    root.withdraw()
    reference_image = filedialog.askopenfilename(parent=root, initialdir=file, title='Reference Selection')
    img_reference = cv2.imread(str(reference_image))

    ### Resize image for ROI selection ###
    h, w, c = img_reference.shape
    h_resized, w_resized = h * resize_scale,  w * resize_scale
    resized_img = cv2.resize(img_reference, (int(w_resized), int(h_resized)))
    cut_coords = cv2.selectROI(resized_img)
    cv2.destroyAllWindows()

    FC_mm = pix_to_mm(resized_img, resize_scale)

    ### Cut image with resized scale ###
    cut_coords_list = list(cut_coords)
    x_1 = int(cut_coords_list[0] / resize_scale)
    x_2 = int(cut_coords_list[2] / resize_scale)
    y_1 = int(cut_coords_list[1] / resize_scale)
    y_2 = int(cut_coords_list[3] / resize_scale)
    img_crop = img_reference[y_1:(y_1 + y_2), x_1:(x_1 + x_2)]
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    Ny, Nx = img_gray.shape

    ### Binarize images with 0 and 1 ###
    img_binarized = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img_binary = img_binarized / 255

    ### Se genera un operador similar a Dx sparse y un vector contador ###
    D = sparse_D(Ny)
    enumerate_array = np.arange(Ny)[::-1]

    ### Iteración de detección ###
    Z = []
    for i in range(N_img):
        img_i = cv2.imread(file + '/' + IMG_names[i])
        img_crop = img_i[y_1:(y_1 + y_2), x_1:(x_1 + x_2)]
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        img_binarized = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        img_binary = img_binarized / 255
        Z_i = []
        for j in range(Nx):
            Dy = D * img_binary[:, j]
            position = np.dot(enumerate_array, Dy)
            Z_i.append(position)
        Z.append(Z_i)

    ### Definiendo espacio-temporal en numpy en pixeles y mm ##
    X = np.arange(Nx)
    X_mm = FC_mm * X
    T = np.arange(N_img)
    Z = np.array(Z)
    Z_mm = FC_mm * Z


    ### Visualizacion del diagrama espacio-temporal  ###
    pcm = plt.pcolormesh(X_mm, T, Z_mm, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\eta(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X_mm[0], X_mm[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()


    


