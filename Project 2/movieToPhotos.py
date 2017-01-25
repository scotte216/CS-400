import cv2
import Transformations
import numpy as np
import os


def videoProcess(interpolator, source, filename, dir, scaleFactor):

    cam = cv2.VideoCapture("{}".format(source))

    ret, first = cam.read()

    height, width, junk = np.shape(first)
    height = int(height * scaleFactor)
    width = int(width * scaleFactor)

    scale = (width, height)

    first = cv2.resize(first, scale, interpolation=cv2.cv.CV_INTER_CUBIC)

    i = 0
    cv2.imwrite("{}/{}{:0>3d}.jpg".format(dir, filename, i), first)
    ret, second = cam.read()
    second = cv2.resize(second, scale, interpolation=cv2.cv.CV_INTER_CUBIC)

    bw_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    bw_second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    forward_flow = cv2.calcOpticalFlowFarneback(bw_first, bw_second, .5, 3, 15, 5, 7, 1.5, 1)
    back_flow = cv2.calcOpticalFlowFarneback(bw_second, bw_first, .5, 3, 15, 5, 7, 1.5, 1)

    if interpolator is not 'fast':
        int_flow = Transformations.intermediate_flow(forward_flow, first, second)
        mask1, mask2 = Transformations.occlusion(forward_flow, back_flow)

    while ret:
        if interpolator == 'fast':
            inbetween = Transformations.fastInterpolation(first, second, forward_flow, back_flow)
        else:
            inbetween = Transformations.interpolate(first, second, int_flow, mask1, mask2)

        i += 1
        cv2.imwrite("{}/{}{:0>3d}.jpg".format(dir, filename, i), inbetween)
        print "Frame {} done.".format(i)
        i += 1
        cv2.imwrite("{}/{}{:0>3d}.jpg".format(dir, filename, i), second)
        first = second
        bw_first = bw_second

        ret, second = cam.read()

        if ret:
            second = cv2.resize(second, scale, interpolation=cv2.cv.CV_INTER_CUBIC)
            bw_second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
            forward_flow = cv2.calcOpticalFlowFarneback(bw_first, bw_second, .5, 3, 15, 5, 7, 1.5, 1)
            back_flow = cv2.calcOpticalFlowFarneback(bw_second, bw_first, .5, 3, 15, 5, 7, 1.5, 1)
            if interpolator is not 'fast':
                int_flow = Transformations.intermediate_flow(forward_flow, first, second)
                mask1, mask2 = Transformations.occlusion(forward_flow, back_flow)
    print "Done!"
    os.system("ffmpeg -pattern_type glob -framerate 60 -i '{}/{}*.jpg' -c:v libx264 -r 60 -pix_fmt yuv420p {}/{}.mp4".format(dir, filename, dir, filename))
    print "Movie created: {}/{}.mp4".format(dir, filename)
    os.system("open {}/{}.mp4".format(dir, filename))
