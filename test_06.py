from directorios import *
from procesos import *

if __name__ == '__main__':
    file = 'C:/Users/rariv/Desktop/test'
    resize_scale = 0.5
    thresh = 50
    
    root = tk.Tk()
    root.withdraw()
    reference_image = filedialog.askopenfilename(parent=root, initialdir=file, title='Reference Selection')
    img_reference = cv2.imread(str(reference_image))

    ### Resize image for ROI selection ###
    h, w, c = img_reference.shape
    print(c)
    h_resized, w_resized = h * resize_scale,  w * resize_scale
    resized_img = cv2.resize(img_reference, (int(w_resized), int(h_resized)))
    cut_coords = cv2.selectROI(resized_img)
    cv2.destroyAllWindows()

    ### Cut image with resized scale ###
    cut_coords_list = list(cut_coords)
    x_1 = int(cut_coords_list[0] / resize_scale)
    x_2 = int(cut_coords_list[2] / resize_scale)
    y_1 = int(cut_coords_list[1] / resize_scale)
    y_2 = int(cut_coords_list[3] / resize_scale)
    img_crop = img_reference[y_1:(y_1 + y_2), x_1:(x_1 + x_2)]
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    Nx, Ny = img_gray.shape

    img_binarized = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    print(img_gray[:, 0])
    print(img_binarized[:, 0])
    #cv2.imshow('Grey scale', img_gray)
    #cv2.imshow('Binarized', img_binarized)
    #cv2.waitKey(0)
    img_binary = img_binarized / 255


    def sparse_D(Nx):
        data = np.ones((2, Nx))
        data[1] = -data[1]
        diags = [1, 0]
        D2 = sparse.spdiags(data, diags, Nx, Nx)
        D2 = sparse.lil_matrix(D2)
        D2[-1, -1] = 0
        return D2

    D = sparse_D(Nx)
    print(len(D.A[:, 0]))
    print(len(img_binary[:, 0]))
    Dy = D * img_binary[:, 0]
    print(Dy)

    


