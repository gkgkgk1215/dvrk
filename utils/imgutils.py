import cv2
import numpy as np

def img_transform(img, angle_deg, tx, ty):
    M_rot = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle_deg, 1)
    M_tran = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_rot, (img.shape[0], img.shape[1]))
    rotated = cv2.warpAffine(img, M_tran, (img.shape[0], img.shape[1]))
    return rotated


def pnt_transform(pnts, angle_deg, tx, ty):
    R = cv2.getRotationMatrix2D((0, 0), -angle_deg, 1)[:, :2]
    T = np.array([tx, ty])
    return np.array([np.array(np.matmul(R, p) + T) for p in pnts])


def cnt_transform(cnt, angle_deg, tx, ty):
    coords = np.array([[p[0][0], p[0][1]] for p in cnt])
    coords_transformed = pnt_transform(coords, angle_deg, tx, ty)
    coords_transformed = coords_transformed.astype(int)
    return np.reshape(coords_transformed, (coords_transformed.shape[0], 1, 2))

def saveimg2npy(filename_input, filename_output):
    mask = cv2.imread(filename_input)
    mask = mask[:,:,0]
    np.save(filename_output, mask)

if __name__ == '__main__':
    saveimg2npy('../img/block_sample.png', 'mask.npy')