# based on Python 3.7 / OpenCV

import cv2
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np

"""
Image showing
"""
def showImage():
    imgfile = 'img/left_image_raw.png'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('First OpenCV', img)

    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.title('model')
    plt.show()

    # key = cv2.waitKey(0) & 0xFF
    #
    # if key == 27:
    #     cv2.destroyAllWindows()
    # elif key == ord('c'):
    #     cv2.imwrite('img/left_image_raw_copy.png', img)
    #     cv2.destroyAllWindows()
    # cv2.namedWindow('Second OpenCV', cv2.WINDOW_NORMAL)

"""
Video showing
"""
def showVideo():
    try:
        print ("camera ON")
        cap = cv2.VideoCapture(0)
    except:
        print ("camera failed")
        return

    cap.set(3,1920)  # 3 means width
    cap.set(4,1080)  # 4 means height

    while True:
        ret, frame = cap.read()
        if not ret:
            print ("video reading error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('video', gray)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break;

    cap.release()
    cv2.destroyAllWindows()

"""
Video writing
"""
def writeVideo():
    try:
        print ("camera ON")
        cap = cv2.VideoCapture(0)
    except:
        print ("camera failed")
        return

    fps = 20.0
    width = int(cap.get(3))
    height = int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc('D','I','V','X')

    out = cv2.VideoWriter('my_first_video_record.avi', fcc, fps, (width,height))
    print ("recording started")

    while True:
        ret, frame = cap.read()
        if not ret:
            print ("video reading error")
            break

        # frame = cv2.flip(frame,0)
        cv2.imshow('video', frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            print ("finishing recording")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

"""
Drawing shapes
"""
def drawing():
    img = np.zeros((512,512,3), np.uint8)

    # drawing lines with various color and thickness
    # BGR: Blue-Green-Red order
    cv2.line(img, (0,0), (400,100), (255,0,0), 5)
    cv2.rectangle(img, (384,0), (510,128), (0,255,0), 3)
    cv2.circle(img, (447,63), 63, (0,0,255), -1)
    cv2.ellipse(img, (256,256), (100,50), 20,45,270, (255,0,0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Tuesday', (10,500), font, 4, (255,255,255), 2)

    cv2.imshow('drawing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Mouse event
"""
b = [i for i in range(256)]
g = [i for i in range(256)]
r = [i for i in range(256)]

def onMouse(event, x,y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        shuffle(b), shuffle(g), shuffle(r)
        cv2.circle(param, (x,y), 50, (b[0],g[0],r[0]), -1)

def mouseBrush():
    img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('mouseEvent')
    cv2.setMouseCallback('mouseEvent',onMouse,param=img)

    while True:
        cv2.imshow('mouseEvent', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

"""
Trackbar function
"""
def onChange(x):
    pass

def trackbar():
    img = np.zeros((200,512,3), np.uint8)
    cv2.namedWindow('color_palette')

    cv2.createTrackbar('B', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('G', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('R', 'color_palette', 0, 255, onChange)

    while True:
        cv2.imshow('color_palette', img)
        key = cv2.waitKey(1) & 0xFF
        if key==27:
            break

        b = cv2.getTrackbarPos('B', 'color_palette')
        g = cv2.getTrackbarPos('G', 'color_palette')
        r = cv2.getTrackbarPos('R', 'color_palette')
        img[:] = [b,g,r]
    cv2.destroyAllWindows()

"""
manipulation of image pixesl & ROI (Region on Image)
"""

def pixelExtract():
    img = cv2.imread('img/right_image_raw.png')
    print ('shape:',img.shape)
    print ('size:', img.size, '(byte)')
    print ('data type:', img.dtype)

    px = img[200,300]
    print (px)

    B = img.item(200, 300, 0)
    G = img.item(200, 300, 1)
    R = img.item(200, 300, 2)
    BGR = [B,G,R]
    print (BGR)

    # img.itemset((200,300,0), 100)   # change blue=100 at pixel (200,300)

    cv2.imshow('original', img)
    subimg = img[50:100, 100:200]   # img cutting
    cv2.imshow('cutting', subimg)
    img[50:100, 0:100] = subimg
    cv2.imshow('modified', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pixelSplit():
    img = cv2.imread('img/right_image_raw.png')

    # b,g,r = cv2.split(img)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    print (img[100,100])
    print (b[100,100], g[100,100], r[100,100])

    cv2.imshow('blue_channel', b)
    cv2.imshow('green_channel', g)
    cv2.imshow('red_channel', r)

    merged_img = cv2.merge((b,g,r))
    cv2.imshow('merged', merged_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
image operation (calculation)
"""

def addImage(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.imshow('left_img', img1)
    cv2.imshow('right_img', img2)

    add_img1 = img1 + img2
    add_img2 = cv2.add(img1,img2)

    cv2.imshow('img1+img2', add_img1)
    cv2.imshow('add(img1,img2)', add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def OnMouse(x):
    pass

def imageBlending(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('MIXING', 'ImgPane', 0,100, OnMouse)
    mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        cv2.imshow('ImgPane', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        mix = cv2.getTrackbarPos('MIXING', 'ImgPane')
    cv2.destroyAllWindows()

def bitOperation(hpos,vpos):
    img1 = cv2.imread('img/IMAG0019_l.jpg')
    img2 = cv2.imread('img/berkeley_logo.png')    # logo

    # area selection to position the logo
    rows, cols, channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]

    # creating mask and mask_inverse
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # make logo black and abstract logo
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # make logo background transparent and add logo image
    dst = cv2.add(img1_bg, img2_fg)
    img1[vpos:vpos + rows, hpos:hpos + cols] = dst

    cv2.imshow('result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
color space handling
"""

def hsv():
    blue = np.uint8([[[255,0,0]]])
    green = np.uint8([[[0, 255, 0]]])
    red = np.uint8([[[0, 0, 255]]])

    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    print ('HSV for BLUE:', hsv_blue)
    print('HSV for GREEN:', hsv_green)
    print('HSV for RED:', hsv_red)

"""
color tracking
"""

def color_tracking():
    try:
        print ("camera ON")
        cap = cv2.VideoCapture(0)
    except:
        print ("camera ON failed")
        return

    while True:
        ret, frame = cap.read()

        # BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of HSV & set threshold
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])

        lower_red = np.array([-20, 100, 100])
        upper_red = np.array([20, 255, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # masking
        res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
        res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

        cv2.imshow('original', frame)
        cv2.imshow('BLUE', res1)
        cv2.imshow('GREEN', res2)
        cv2.imshow('RED', res3)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

"""
Thresholding
"""

def thresholding():
    img = cv2.imread('img/left_image_raw.png', cv2.IMREAD_GRAYSCALE)
    threshold =127
    ret, thr1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    ret, thr2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    ret, thr3 = cv2.threshold(img, threshold, 255, cv2.THRESH_TRUNC)
    ret, thr4 = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
    ret, thr5 = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO_INV)

    # Adaptive thresholding
    thr6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thr7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'ADAPTIVE_MEAN', 'ADAPTIVE_GAUSSIAN']
    images = [img, thr1, thr2, thr3, thr4, thr5, thr6, thr7]

    for i in range(len(titles)):
        cv2.imshow(titles[i], images[i])


    # Otsu binarization
    ret, thr8 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu binarization after Gaussian blurring
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thr9 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['original_noisy', 'Histogram', 'G-Thresholding',
              'original_noisy', 'Histogram', 'Otsu Thresholding',
              'Gaussian-filtered', 'Histogram', 'Otsu Thresholding']
    images = [img, 0, thr1, img, 0, thr8, blur, 0, thr9]

    for i in range(3):
        plt.subplot(3,3,i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Image blur
"""

# 2D convolution (image filtering)
def bluring_filtering():
    img = cv2.imread('img/left_image_raw.png')

    kernel = np.ones((5,5), np.float32)/25
    blur = cv2.filter2D(img, -1, kernel)

    cv2.imshow('original', img)
    cv2.imshow('blur', blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def onMouse(x):
    pass

def bluring():
    img = cv2.imread('img/left_image_raw.png')

    cv2.namedWindow('BlurPane')
    cv2.createTrackbar('BLUR_MODE', 'BlurPane', 0, 2, OnMouse)
    cv2.createTrackbar('BLUR', 'BlurPane', 0, 5, OnMouse)

    mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPane')
    val = cv2.getTrackbarPos('BLUR', 'BlurPane')

    while True:
        val = val*2 + 1
        try:
            if mode==0:
                blur = cv2.blur(img, (val, val))
            elif mode==1:
                blur = cv2.GaussianBlur(img, (val,val), 0)
            elif mode==2:
                blur = cv2.medianBlur(img, val)
            else:
                break
            cv2.imshow('BlurPane', blur)
        except:
            break

        key = cv2.waitKey(1) & 0xFF
        if key==27:
            break

        mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPane')
        val = cv2.getTrackbarPos('BLUR', 'BlurPane')

    cv2.destroyAllWindows()

"""
image contour
"""
def moment():
    img = cv2.imread('img/berkeley_logo.png')
    img1 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold = 127
    ret, thr = cv2.threshold(imgray, threshold, 255, 0)
    ret, thr1 = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY)
    ret, thr2 = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY_INV)
    ret, thr3 = cv2.threshold(imgray, threshold, 255, cv2.THRESH_TRUNC)
    ret, thr4 = cv2.threshold(imgray, threshold, 255, cv2.THRESH_TOZERO)
    ret, thr5 = cv2.threshold(imgray, threshold, 255, cv2.THRESH_TOZERO_INV)

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[40]
    mmt = cv2.moments(cnt)
    for key, val in mmt.items():
        print ('%s:\t%.5f' %(key,val))

    # center of moment
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])
    print(cx, cy)

    # area & perimeter
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print ('Area =', area)
    print ('Perimeter = ', perimeter)

    for i in range(len(contours)):
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx_PD = cv2.approxPolyDP(contours[i], epsilon, True)
        cv2.drawContours(img, [approx_PD], 0, (255,255,0), 1)
        cv2.imshow('contour', img)

        check = cv2.isContourConvex(contours[i])
        if not check:
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(img1, [hull], 0, (255,255,0), 1)
        cv2.imshow('convex_hull', img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convex():
    img = cv2.imread('img/hand.jpg')
    img1 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[1]
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    cv2.imshow('contour', img)

    check = cv2.isContourConvex(cnt)
    if not check:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img1, [hull], 0, (0, 255, 0), 3)
        cv2.imshow('convex_hull', img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour():
    img = cv2.imread('img/crash.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape[:2]

    ret, thr = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contour hierarchy

    cnt = contours[5]

    # bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 3)

    # minimum area rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print (rect)
    print (box)

    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

    # minimum enclosing circle
    (x,y), r = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(r)
    cv2.circle(img, center, radius, (255, 0, 0), 3)

    # fitting ellipse
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img, ellipse, (0,255,0), 3)

    # fitting line
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    ly = int((-x*vy/vx)+y)
    ry = int(((cols-x)*vy/vx)+y)
    cv2.line(img, (cols-1, ry), (0, ly), (0, 0, 255), 3)
    cv2.imshow('contour', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour_adv():
    img = cv2.imread('img/crash.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[5]

    hull = cv2.convexHull(cnt)
    cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)

    # defects calculation
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        sp, ep, fp, dist = defects[i, 0]
        start = tuple(cnt[sp][0])
        end = tuple(cnt[ep][0])
        farthest = tuple(cnt[fp][0])
        cv2.circle(img, farthest, 5, (0, 255, 0), -1)


    # distance to contour
    outside = (55,150)
    inside = (140, 100)

    dist1 = cv2.pointPolygonTest(cnt, outside, True)
    dist2 = cv2.pointPolygonTest(cnt, inside, True)

    print ('Distance from contour to point (%d, %d) = %.3f' %(outside[0], outside[1], dist1))
    print ('Distance from contour to point (%d, %d) = %.3f' %(inside[0], inside[1], dist2))

    cv2.circle(img, outside, 3, (0, 255, 0), -1)
    cv2.circle(img, inside, 3, (255, 0, 255), -1)

    cv2.imshow('defects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.matchShapes()

def histogram():
    pass

# image histrogram equalization
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def equalization():
    pass

# 2D image histogram
def hist2D():
    pass

# image histrogram background projection
def backProjection():
    pass

# fourier transform
def fourier():
    pass

# template matching
def tmpMatching():
    pass

# line/circle detection using the Hough transformation
def hough():
    pass

# image segmentation using watershed
def watershed():
    pass

# foreground abstraction using grabcut algorithm
def grabcut():
    pass


# object tracking
col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
roi_hist = None

def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputmode
    global rectangle, roi_hist, trackWindow

    if inputmode:
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col,row), (x,y), (0,255,0), 2)
                cv2.imshow('frame', frame)

        elif event == cv2.EVENT_LBUTTONUP:
            inputmode = False
            rectangle = False
            cv2.rectangle(frame, (col,row), (x,y), (0,255,0), 2)
            height, width = abs(row-y), abs(col-x)
            trackWindow = (col, row, width, height)
            roi = frame[row:row+height, col:col+width]
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0,180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return

def camShift():
    global frame, frame2, inputmode, trackWindow, roi_hist, out
    try:
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        cap.set(4, 320)
    except:
        print('camera operation failed')
        return

    ret, frame = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break

        if k == ord('i'):
            print ('Select a tracking area and press any key')
            inputmode = True
            frame2 = frame.copy()

            while inputmode:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    showImage()
    # showVideo()
    # writeVideo()
    # drawing()
    # mouseBrush()
    # trackbar()
    # pixelExtract()
    # pixelSplit()
    # addImage('img/left_image_raw.png', 'img/right_image_raw.png')
    # imageBlending('img/left_image_raw.png', 'img/right_image_raw.png')
    # bitOperation(800,10)
    # hsv()
    # color_tracking()
    # thresholding()
    # bluring_filtering()
    # bluring()
    # moment()
    # convex()
    # contour()
    # contour_adv()
    # histogram()
    # equalization()
    # hist2D()
    # backProjection()
    # fourier()
    # tmpMatching()
    # hough()
    # watershed()
    # camShift()