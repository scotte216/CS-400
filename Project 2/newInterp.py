import cv2
import Transformations
import numpy as np


def execute(interpolator, source1, source2, output, scaleFactor):

    first = cv2.imread(source1)
    second = cv2.imread(source2)

    height, width, junk = np.shape(first)
    height = int(height * scaleFactor)
    width = int(width * scaleFactor)

    scale = (width, height)

    first = cv2.resize(first, scale, interpolation=cv2.cv.CV_INTER_CUBIC)
    second = cv2.resize(second, scale, interpolation=cv2.cv.CV_INTER_CUBIC)

    bw_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    bw_second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    forward_flow = cv2.calcOpticalFlowFarneback(bw_first, bw_second, .5, 3, 15, 5, 7, 1.5, 1)
    back_flow = cv2.calcOpticalFlowFarneback(bw_second, bw_first, .5, 3, 15, 5, 7, 1.5, 1)

    if interpolator is 'fast':
        result = Transformations.fastInterpolation(first, second, forward_flow, back_flow)
    else:
        int_flow = Transformations.intermediate_flow(forward_flow, first, second)
        mask1, mask2 = Transformations.occlusion(forward_flow, back_flow)
        result = Transformations.interpolate(first, second, int_flow, mask1, mask2)

    cv2.imwrite("{}.jpg".format(output), result)
    while True:
        cv2.imshow("first", first)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ch == 27:
            break
        cv2.imshow("middle", result)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ch == 27:
            break
        cv2.imshow("third", second)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ch == 27:
            break
