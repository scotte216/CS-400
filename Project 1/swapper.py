# USAGE
# python swapper.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg

# import the necessary packages
from Transforms import l_alpha_beta_color_transfer
from Transforms import l_a_b_color_transfer
from Transforms import x_y_z_color_transfer
import argparse
import cv2

def show_image(title, image, width = 600):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    # estate
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # display image
    cv2.imshow(title, resized)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required = True,
    help = "Path to the color source image")
ap.add_argument("-t", "--target", required = True,
    help = "Path to the target original image")
args = vars(ap.parse_args())

# load the images
source = cv2.imread(args["source"])
target = cv2.imread(args["target"])

# transfer the color distribution from the source image
# to the target image
result = l_alpha_beta_color_transfer(source, target)
result2 = l_a_b_color_transfer(source, target)
result3 = x_y_z_color_transfer(source, target)

# show the images and wait for a key press
show_image("Source", source)
cv2.waitKey(0);
show_image("Target", target)
cv2.waitKey(0);
show_image("L Alpha Beta", result)
cv2.waitKey(0);
show_image("L A B", result2)
cv2.waitKey(0)
show_image("X Y Z", result3)
cv2.waitKey(0)
