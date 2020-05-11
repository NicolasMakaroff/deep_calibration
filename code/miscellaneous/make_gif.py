import glob
from PIL import Image

# filepaths
fp_in = "figures/data_20/*"
fp_out = "figures/data_20/StyblinskyTang_20pts.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f).resize((600, 600), Image.ANTIALIAS) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500,loop=0)
