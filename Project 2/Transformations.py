import numpy as np


def intermediate_flow(flow, img0, img1):
    h, w, junk = np.shape(flow)
    y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)
    result = np.empty((h, w, 2))
    dx, dy = .5 * flow[y, x].T

    # dest_coords is (w*h) elements, each element holds [x, y]
    dest_coords = np.vstack([x + dx, y + dy]).T
    dest_coords = np.round(dest_coords, 2)

    dest_coords[:, 0] = np.clip(dest_coords[:, 0], 0, w - 1)
    dest_coords[:, 1] = np.clip(dest_coords[:, 1], 0, h - 1)

    for index in range(0, h * w - 1):
        each = dest_coords[index]
        lower = np.floor(each).astype(int)
        upper = np.ceil(each).astype(int)
        best = each
        previousDiff = np.inf
        for i in range(lower[0], upper[0]):
            for j in range(lower[1], upper[1]):
                try:
                    diff = np.average(np.abs(np.subtract(img0[j, i, :].astype(int), img1[j, i, :].astype(int))))
                    best = [i, j] if diff < previousDiff else best
                except IndexError:
                    continue
        dest_coords[index, :] = best

    nx, ny = dest_coords.T.astype(int)
    result[ny, nx, :] = flow[ny, nx]

    return result


def occlusion(forward_flow, back_flow):
    h, w, junk = np.shape(forward_flow)
    img0_mask = np.zeros((h, w))
    img1_mask = np.ones((h, w))

    y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)
    dx, dy = forward_flow[y, x].T
    forward = np.vstack([x + dx, y + dy]).T
    fx = np.clip(forward[:, 0], 0, w - 1).astype(int)
    fy = np.clip(forward[:, 1], 0, h - 1).astype(int)

    img1_mask[fy, fx] = 0

    for i in range(0, h * w - 1):
        if abs(np.linalg.norm(forward_flow[y[i], x[i]] - back_flow[fy[i], fx[i]])) > .5:
            img0_mask[y[i], x[i]] = 1

    return img0_mask, img1_mask


def fastInterpolation(first, second, forward_flow, back_flow):
    h, w = first.shape[:2]
    y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)

    # creates a true/false array for each element if it's close to the
    # same element in the other image
    same = np.isclose(first[y, x, :], second[y, x, :], .01, 10)
    # or
    # same = np.isclose(np.vstack(first), np.vstack(second), .01, 5)

    dfx, dfy = .5 * forward_flow[y, x].T
    dbx, dby = .5 * back_flow[y, x].T

    forward = np.vstack([x + dfx, y + dfy]).T.astype(int)
    backward = np.vstack([x + dbx, y + dby]).T.astype(int)

    fx = np.clip(forward[:, 0], 0, w - 1)
    fy = np.clip(forward[:, 1], 0, h - 1)

    bx = np.clip(backward[:, 0], 0, w - 1)
    by = np.clip(backward[:, 1], 0, h - 1)

    temp1 = ~same * .5 * first[np.round(by), np.round(bx), :]

    temp2 = ~same * .5 * second[np.round(fy), np.round(fx), :]

    result = temp1 + temp2 + same * np.vstack(second)

    result = result.astype("uint8").reshape(h, w, 3)

    return result


def interpolate(first, second, int_flow, first_occ, second_occ):
    h, w, junk = np.shape(first)
    result = np.empty((h, w, 3))
    for i in range(h):
        for j in range(w):
            x_first = round(j - .5 * int_flow[i, j, 0])
            y_first = round(i - .5 * int_flow[i, j, 1])
            if x_first < 0:
                x_first = 0
            if x_first >= w:
                x_first = w - 1
            if y_first < 0:
                y_first = 0
            if y_first >= h:
                y_first = h - 1

            x_second = round(j + .5 * int_flow[i, j, 0])
            y_second = round(i + .5 * int_flow[i, j, 1])
            if x_second < 0:
                x_second = 0
            if x_second >= w:
                x_second = w - 1
            if y_second < 0:
                y_second = 0
            if y_second >= h:
                y_second = h - 1

            if first_occ[y_first, x_first] == 0 and second_occ[y_second, x_second] == 0:
                result[i, j, :] = .5 * (first[y_first, x_first, :]) + .5 * (second[y_second, x_second, :])
            elif first_occ[y_first, x_first] == 1:
                result[i, j, :] = second[y_second, x_second, :]
            elif second_occ[y_second, x_second] == 1:
                result[i, j, :] = first[y_first, x_first, :]

    return result.astype("uint8")
