import numpy as np
import cv2
import argparse
from numba import jit
from scipy import ndimage as ndi


def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis


def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def backward_energy(im):
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag


@jit
def forward_energy(im):
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8),
                      cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    return energy


@jit
def add_seam(im, im_seam, seam_idx, mask):
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                output[row, col, ch] = im[row, col, ch]
                output[row, col, ch] = im_seam[row, col, ch]
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = im_seam[row, col, ch]
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output


@jit
def add_seam_grayscale(im, seam_idx):

    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = 150
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = 150
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]

    return output


@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


@jit
def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))


@jit
def get_minimum_seam(im, mask=None, remove_mask=None):

    h, w = im.shape[:2]
    energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
    M = energyfn(im)

    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    # give removal mask priority over protective mask by using larger negative value
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask


def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, remove_mask=mask)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask


def seams_insertion(im, im_seam, num_add, mask=None, vis=False, rot=False):
    seams_record = []
    temp_im_record = []
    temp_im = im_seam.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, remove_mask=temp_mask)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im_record.append(temp_im)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    # seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        imseam = temp_im_record.pop()
        im = add_seam(im, imseam, seam, mask)
        if vis:
            visualize(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        # for remaining_seam in seams_record:
        #     remaining_seam[np.where(remaining_seam >= seam)] += 1

    return im, mask


def seam_carve(im, im_seam, dy, dx, mask=None, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = seams_removal(output, -dx, mask, vis)

    elif dx > 0:
        output, mask = seams_insertion(output, im_seam, dx, mask, vis)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        im_seam = rotate_image(im_seam, True)

        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(
            output, im_seam, dy, mask, vis, rot=True)
        output = rotate_image(output, False)
        im_seam = rotate_image(im_seam, False)

    return output, mask


def object_removal(im, rmask, mask=None, vis=False, horizontal_removal=False):
    im = im.astype(np.float64)
    rmask = rmask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im

    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, mask, rmask)
        if vis:
            visualize(output, boolmask, rotate=horizontal_removal)
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)

    num_add = (h if horizontal_removal else w) - output.shape[1]
    output, mask = seams_insertion(
        output, output, num_add, mask, vis, rot=horizontal_removal)
    if horizontal_removal:
        output = rotate_image(output, False)

    return output


SEAM_COLOR = np.array([255, 200, 200])    # seam visualization color (BGR)
SHOULD_DOWNSIZE = True
DOWNSIZE_WIDTH = 500
ENERGY_MASK_CONST = 100000.0
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
USE_FORWARD_ENERGY = True                 # if True, use forward energy algorithm

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-resize", action='store_true')
    group.add_argument("-remove", action='store_true')

    ap.add_argument("-im", help="Path to image", required=True)
    ap.add_argument("-im_seam", help="Path to image", required=True)
    ap.add_argument("-out", help="Output file name", required=True)
    ap.add_argument("-maskout", help="Output file name")
    ap.add_argument("-mask", help="Path to (protective) mask")
    ap.add_argument("-rmask", help="Path to removal mask")
    ap.add_argument(
        "-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
    ap.add_argument(
        "-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)
    ap.add_argument(
        "-vis", help="Visualize the seam removal process", action='store_true')
    ap.add_argument(
        "-hremove", help="Remove horizontal seams for object removal", action='store_true')
    ap.add_argument("-backward_energy",
                    help="Use backward energy map (default is forward)", action='store_true')
    args = vars(ap.parse_args())

    IM_PATH, IM_SEAM, MASK_PATH, OUTPUT_NAME, MASK_OUT_NAME, R_MASK_PATH = args[
        "im"], args["im_seam"], args["mask"], args["out"], args["maskout"], args["rmask"]

    im = cv2.imread(IM_PATH)
    im_seam = cv2.imread(IM_SEAM)
    assert im is not None
    mask = cv2.imread(MASK_PATH, 0) if MASK_PATH else None
    rmask = cv2.imread(R_MASK_PATH, 0) if R_MASK_PATH else None

    USE_FORWARD_ENERGY = not args["backward_energy"]

    # downsize image for faster processing
    h, w = im.shape[:2]
    if SHOULD_DOWNSIZE and w > DOWNSIZE_WIDTH:
        im = resize(im, width=DOWNSIZE_WIDTH)
        if mask is not None:
            mask = resize(mask, width=DOWNSIZE_WIDTH)
        if rmask is not None:
            rmask = resize(rmask, width=DOWNSIZE_WIDTH)

    # image resize mode
    if args["resize"]:
        dy, dx = args["dy"], args["dx"]
        assert dy is not None and dx is not None
        output, mask_out = seam_carve(im, im_seam, dy, dx, mask, args["vis"])
        cv2.imwrite(OUTPUT_NAME, output)
        cv2.imwrite(MASK_OUT_NAME, mask_out)

    # object removal mode
    elif args["remove"]:
        assert rmask is not None
        output = object_removal(im, rmask, mask,
                                args["vis"], args["hremove"])
        cv2.imwrite(OUTPUT_NAME, output)
