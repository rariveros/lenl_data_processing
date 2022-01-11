import glob
from PIL import Image

if __name__ == '__main__':
    # filepaths
    fp_in = 'C:/Users/mnustes_science/PT_fluids/faraday_drift/figures/plots/*.jpg'
    fp_out = "image.gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=40, loop=0)