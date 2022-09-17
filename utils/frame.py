import cv2
from PIL import Image
import os
directory_path = os.path.dirname(__file__)
def add_frame(path):
    # reading the image
    virat_img = cv2.imread(path)
    
    # making border around image using copyMakeBorder
    borderoutput = cv2.copyMakeBorder(
        virat_img, 100, 0, 50, 50, cv2.BORDER_CONSTANT, value=[255,255,255])

    img = Image.fromarray(cv2.cvtColor(borderoutput,cv2.COLOR_BGR2RGB),mode='RGB')
    # image watermark
    size = (500, 500)

    crop_image = Image.open(os.path.join(directory_path,"logo.jpg"))
    # to keep the aspect ration in intact
    crop_image.thumbnail(size)

    # add watermark
    copied_image = img.copy()
    # base image
    copied_image.paste(crop_image, (0, 0))
    # pasted the crop image onto the base image
    return copied_image