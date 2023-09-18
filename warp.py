import math
import numpy as np
from PIL import Image
import cv2


# Compute homogeneous matrix
def compute_h(p1, p2, rows, cols, flipped):
    x_diff = math.radians(p1[0] - p2[0])
    y_diff = math.radians(p1[1] - p2[1])
    z_diff = math.radians(p1[2] - p2[2])
    x_flip = np.eye(3)
    y_flip = np.eye(3)
    if flipped:
        x_flip = np.mat([(-1, 0, cols-1),
                 (0, 1, 0),
                 (0, 0, 1)])
#    print(x_flip)
#    print(y_flip)
    srcPoint = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]], dtype=np.float32)
    if x_diff > 0:
        dstPoint = np.array([[(cols-1) * -math.tan(x_diff)/10, 0], [(cols-1) * math.tan(x_diff)/10, rows-1], [(cols-1) * (1-math.tan(x_diff)/10), rows-1], [(cols-1) * (1+math.tan(x_diff)/10), 0]], dtype=np.float32)
    else:
        x_diff = -x_diff
        dstPoint = np.array([[(cols-1) * math.tan(x_diff)/10, 0], [(cols-1) * -math.tan(x_diff)/10, rows-1], [(cols-1) * (1+math.tan(x_diff)/10), rows-1], [(cols-1) * (1-math.tan(x_diff)/10), 0]], dtype=np.float32)
    matrix_x = cv2.getPerspectiveTransform(srcPoint, dstPoint)
#    print(matrix_x)

    if y_diff < 0:
        y_diff = -y_diff
        dstPoint = np.array([[0, (rows-1) * -math.tan(y_diff)/10], [0, (rows-1) * (1+math.tan(y_diff)/10)], [cols-1, (rows-1) * (1-math.tan(y_diff)/10)], [cols-1, (rows-1) * math.tan(y_diff)/10]], dtype=np.float32)
    else:
        dstPoint = np.array([[0, (rows-1) * math.tan(y_diff)/10], [0, (rows-1) * (1-math.tan(y_diff)/10)], [(cols-1), (rows-1) * (1+math.tan(y_diff)/10)], [(cols-1), (rows-1) * -math.tan(y_diff)/10]], dtype=np.float32)

#    print(dstPoint)
    matrix_y = cv2.getPerspectiveTransform(srcPoint, dstPoint)
#    print(matrix_y)

    Rx = np.mat([(1, 0, 0),
                 (0, 1, rows/2 * (1 - math.cos(x_diff))),
                 (0, 0, 1)]) * \
         np.mat([(1, 0, 0),
                 (0, math.cos(x_diff), 0),
                 (0, 0, 1)]) * matrix_x
    Ry = np.mat([(1, 0, cols/2 * (1 - math.cos(y_diff))),
                 (0, 1, 0),
                 (0, 0, 1)]) * \
         np.mat([(math.cos(y_diff), 0, 0),
                 (0, 1, 0),
                 (0, 0, 1)]) * matrix_y
    Rz = np.mat([(1, 0, cols/2 * (1 - math.cos(z_diff)) + rows/2 * math.sin(z_diff)),
                 (0, 1, rows/2 * (1 - math.cos(z_diff)) - cols/2 * math.sin(z_diff)),
                 (0, 0, 1)]) * \
         np.mat([(math.cos(z_diff), -math.sin(z_diff), 0),
                 (math.sin(z_diff), math.cos(z_diff), 0),
                 (0, 0, 1)])
    R = Rz * Ry * Rx * x_flip
    return R


# warp image with homogeneous matrix
def warp_image(igs_in, R): # igs_ref, H):
    rows, cols, depth = igs_in.shape
    # we will warp image in reverse way; calculate matrix inversion
    H_inv = np.linalg.inv(R.reshape(3, 3))

    # helper functions for r, g, b. calculate matrix multiplication product to find out coordinate.
    def warp_point_0(j, i):
        product = [(H_inv[0,0] * i + H_inv[0,1] * j + H_inv[0,2] * 1) / (H_inv[2,0] * i + H_inv[2,1] * j + H_inv[2,2] * 1),
                   (H_inv[1,0] * i + H_inv[1,1] * j + H_inv[1,2] * 1) / (H_inv[2,0] * i + H_inv[2,1] * j + H_inv[2,2] * 1)]
        if 0 < product[1] < rows and 0 < product[0] < cols:
            return igs_in[int(product[1]), int(product[0]), 0]
        else:
            # if there is no corresponding coordinate, just return 0
            return 0

    def warp_point_1(j, i):
        product = [(H_inv[0,0] * i + H_inv[0,1] * j + H_inv[0,3] * 1) / (H_inv[2,0] * i + H_inv[2,1] * j + H_inv[2,2] * 1),
                   (H_inv[1,0] * i + H_inv[1,1] * j + H_inv[1,3] * 1) / (H_inv[2,0] * i + H_inv[2,1] * j + H_inv[2,2] * 1)]
        if 0 < product[1] < rows and 0 < product[0] < cols:
            return igs_in[int(product[1]), int(product[0]), 1]
        else:
            return 0

    def warp_point_2(j, i):
        product = [(H_inv[0,0] * i + H_inv[0,1] * j + H_inv[0,3] * 1) / (H_inv[2,0] * i + H_inv[2,1] * j + H_inv[2,2] * 1),
                   (H_inv[1,0] * i + H_inv[1,1] * j + H_inv[1,3] * 1) / (H_inv[2,0] * i + H_inv[2,1] * j + H_inv[2,2] * 1)]
        if 0 < product[1] < rows and 0 < product[0] < cols:
            return igs_in[int(product[1]), int(product[0]), 2]
        else:
            return 0

    r = np.fromfunction(np.vectorize(warp_point_0), (rows, cols)).reshape(rows, cols, 1)
    g = np.fromfunction(np.vectorize(warp_point_1), (rows, cols)).reshape(rows, cols, 1)
    b = np.fromfunction(np.vectorize(warp_point_2), (rows, cols)).reshape(rows, cols, 1)

    # concatenate and make warp image.
    igs_warp = np.concatenate((r, g, b), axis=-1).astype(np.uint8)

    return igs_warp


def main():
    print("Start warping.")
    ##############
    # step 1: mosaicing
    ##############
    # twice_matrix = np.loadtxt('woman.txt', usecols=range(4))
    target_matrix = np.loadtxt('target_angle.txt', usecols=range(4))
    target_line, temp_cols = target_matrix.shape
    textbook_matrix = np.loadtxt('textbook_angle.txt', usecols=range(4))
    textbook_line, temp_cols = textbook_matrix.shape
    result_angle = []
    # read images
    for i in range(0, textbook_line):
        textbook_number = int(textbook_matrix[i][0])
        textbook_angle = textbook_matrix[i][1:4]
#        print(textbook_number)
#        print(textbook_angle)
        diff = 50000
        face_number = -1
        face_angle = [0, 0, 0]
        flipped = False
        # for j in range(0, 104):
        for j in range(0, target_line):
            twice_angle = target_matrix[j][1:4]
            flipped_angle = target_matrix[j][1:4] * np.array([-1, 1, -1])
            if diff > np.sum(np.abs(textbook_angle - twice_angle)):
                face_angle = twice_angle
                face_number = int(target_matrix[j][0])
                diff = np.sum(np.abs(textbook_angle - twice_angle))
                flipped = False
            if diff > np.sum(np.abs(textbook_angle - flipped_angle)):
                face_angle = flipped_angle
                face_number = int(target_matrix[j][0])
                diff = np.sum(np.abs(textbook_angle - flipped_angle))
                flipped = True
#        print(face_angle)
#        print(face_number)
        diff_angle = textbook_angle - face_angle
        diff_angle = np.insert(diff_angle, 0, textbook_number)
        result_angle.append(diff_angle)
        # img_in = cv2.imread('woman_crop/' + str(face_number) + '.png')
        img_in = cv2.imread('target_crop/' + str(face_number) + '.png')
        rows, cols, ch = img_in.shape
        R = compute_h(textbook_angle, face_angle, rows, cols, flipped)
        img_warp = cv2.warpPerspective(img_in, R, (rows, cols))
        cv2.imwrite('warped/' + str(textbook_number) + '.png', img_warp)


if __name__ == '__main__':
    main()
