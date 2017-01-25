import numpy as np
import cv2

def transformRGB2L_Alpha_Beta(input):
    RGB2LMS = [[0.3811, 0.5783, 0.0402],
               [0.1967, 0.7244, 0.0782],
               [0.0241, 0.1288, 0.8444]]

    #lms = [[[0.0, 0.0, 0.0] for x in range(input.shape[1])] for x in range(input.shape[0])]

    lms = np.tensordot(input, RGB2LMS, ([2], [1]))

    #Ensure no values are too small (to avoid divide by zero)
    lms = np.clip(lms,0.0000000001,255)
    lms = np.log10(lms)

    a = [[1/np.sqrt(3), 0, 0],
         [0, 1/np.sqrt(6), 0],
         [0, 0, 1/np.sqrt(2)]]

    b = [[1, 1, 1],
         [1, 1, -2],
         [1, -1, 0]]

    ab = np.dot(a, b)

    #lab = [[[0.0, 0.0, 0.0] for x in range(input.shape[1])] for x in range(input.shape[0])]

    lab = np.tensordot(lms, ab, ([2], [1]))

    lab = np.clip(lab, 0, 255)
    return lab


def transformL_Alpha_Beta2RGB(input):

    a = [[1, 1, 1],
         [1, 1, -1],
         [1, -2, 0]]

    b = [[np.sqrt(3)/3, 0, 0],
         [0, np.sqrt(6)/6, 0],
         [0, 0, np.sqrt(2)/2]]

    ab = np.dot(a,b)

    #lms = [[[0.0, 0.0, 0.0] for x in range(input.shape[1])] for x in range(input.shape[0])]
    #rgb = [[[0.0, 0.0, 0.0] for x in range(input.shape[1])] for x in range(input.shape[0])]

    lms = np.tensordot(input, ab, ([2], [1]))

    lms = np.power(10,lms)

    LMS2RGB = [[4.4679, -3.5873, 0.1193],
               [-1.2186, 2.3809, -0.1624],
               [0.0497, -0.2439, 1.2045]]

    rgb = np.tensordot(lms , LMS2RGB, ([2], [1]))

    rgb = np.clip(rgb, 0, 255)
    return rgb


def l_a_b_color_transfer(source,target):
    # Convert the images from BGR to LAB colorspace
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Use the 3-channel color swapper function
    result = transformation(source, target)

    # convert the image from LAB to GBR color space and return
    return cv2.cvtColor(result.astype("uint8"), cv2.COLOR_LAB2BGR)


def x_y_z_color_transfer(source, target):
    # Convert the images from BGR to XYZ colorspace
    source = cv2.cvtColor(source, cv2.COLOR_BGR2XYZ).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2XYZ).astype("float32")

    # Use the 3-channel color swapper function
    result = transformation(source, target)

    # convert the image from XYZ to GBR color space and return
    return cv2.cvtColor(result.astype("uint8"), cv2.COLOR_XYZ2BGR)


def l_alpha_beta_color_transfer(source, target):
    """

    """
    # convert the source file from BGR to RGB color, and convert to float32
    # Then run the custom RGB 2 L Alpha Beta color space
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).astype("float32")
    source = transformRGB2L_Alpha_Beta(source)

    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype("float32")
    target = transformRGB2L_Alpha_Beta(target)

    #Next, take the L Alpha Beta color and run the calculations to switch colors
    result = transformation(source, target)

    #finally transform the result from L Alpha Beta to RGB, then RGB to BGR
    result = transformL_Alpha_Beta2RGB(result)
    transfer = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_RGB2BGR)

    # return the color transferred image
    return transfer

def transformation(source, target):
    # Calculate mean and standard deviation for all 3 channels.
    ((chan1MeanSrc, chan2MeanSrc, chan3MeanSrc), (chan1StdSrc, chan2StdSrc, chan3StdSrc)) = cv2.meanStdDev(source)
    ((chan1MeanTar, chan2MeanTar, chan3MeanTar), (chan1StdTar, chan2StdTar, chan3StdTar)) = cv2.meanStdDev(target)

    # subtract the target means from each channel
    (chan1, chan2, chan3) = cv2.split(target)

    chan1 -= chan1MeanTar
    chan2 -= chan2MeanTar
    chan3 -= chan3MeanTar

    #normalize each channel by the ratio of the standard deviations of the source and target
    chan1 = (chan1StdSrc / chan1StdTar) * chan1
    chan2 = (chan2StdSrc / chan2StdTar) * chan2
    chan3 = (chan3StdSrc / chan3StdTar) * chan3

    # add the source means to each channel
    chan1 += chan1MeanSrc
    chan2 += chan2MeanSrc
    chan3 += chan3MeanSrc

    # Ensure all values fall in acceptable ranges
    chan1 = np.clip(chan1, 0, 255)
    chan2 = np.clip(chan2, 0, 255)
    chan3 = np.clip(chan3, 0, 255)

    # merge all channels together and return
    return cv2.merge([chan1, chan2, chan3])

