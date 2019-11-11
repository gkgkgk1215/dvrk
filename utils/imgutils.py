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
    img = cv2.imread(filename_input)
    ret, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = mask_inv[:,:,0]
    np.save(filename_output, mask_inv)
    cv2.imshow('mask', mask_inv)
    cv2.waitKey(0)

def contour2npy(filename_input, filename_output):
    img = cv2.imread(filename_input, cv2.IMREAD_GRAYSCALE)
    ret, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    edge = cv2.Canny(img, mask.shape[0], mask.shape[1])
    _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background = np.zeros_like(img)
    cv2.drawContours(background, contours, -1, (255,255,255), 2)
    cv2.imshow("contour", background)
    cv2.waitKey(0)
    np.save(filename_output, background)

if __name__ == '__main__':
    saveimg2npy('../img/block_sample_drawing.png', 'mask.npy')
    # contour2npy('../img/block_sample_drawing.png', 'contour.npy')