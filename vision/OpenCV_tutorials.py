# based on Python 3.7 / OpenCV
import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
from scipy import signal
from scipy import misc
import imutils

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
        # rotated = imutils.rotate_bound(gray, 20)
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

    # # video recording
    # self.__video_recording = video_recording
    # if self.__video_recording is True:
    #     fps = 1000.0 / self.__interval_ms
    #     fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    #     self.__img_width = 640
    #     self.__img_height = 360
    #     self.out = cv2.VideoWriter('video_recorded.avi', fcc, fps, (self.__img_width, self.__img_height))
    #     print ("start recording")


"""
Drawing shapes
"""
def drawing():
    img = np.zeros((512,512,3), np.uint8)

    # drawing lines with various color and thickness
    # BGR: Blue-Green-Red order
    cv2.line(img, (0,0), (400,100), (255,0,0), 5)
    # cv2.triangle()
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
    if event == cv2.EVENT_LBUTTOnNDBLCLK:
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
    img1 = cv2.imread('../img/IMAG0019_l.jpg')
    img2 = cv2.imread('../img/berkeley_logo.png')    # logo

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

def correlated():
    face = misc.face(gray=True) - misc.face(gray=True).mean()
    template = np.copy(face[300:365, 670:750])  # right eye
    template -= template.mean()
    face = face + np.random.randn(*face.shape) * 50  # add noise
    corr = signal.correlate2d(face, template, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

def rotate_image():
    mask = cv2.imread('../img/block_sample.png')
    print (mask.shape)
    rotated = imutils.rotate_bound(mask, 40)
    print (rotated.shape)
    cv2.imshow("aa", rotated)
    cv2.waitKey(0)

def scaling_image():
    mask = cv2.imread('../img/block_sample.png')
    height, width = mask.shape[:2]
    print (height, width, (int)(1.0*width), (int)(1.0*height))
    downsized = cv2.resize(mask, ((int)(0.5*width), (int)(0.5*height)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("downsizing", downsized)
    cv2.waitKey(0)

def checkerboard_calibration(row, col):
    imgfile = '../img/img_color_checkerboard.png'
    img = cv2.imread(imgfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (row, col), None)
    if ret == True:
        # If found, add object points, image points (after refining them)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        detected = cv2.drawChessboardCorners(img, (row, col), corners2, ret)

    cv2.imshow('Checkerboard', detected)
    cv2.waitKey(0)

def transform_pixel_to_checkerboard(row, col):
    row=3
    col=4
    # import camera intrinsics
    filename = '../calibration_files/calib_laptop.npz'
    with np.load(filename) as X:
        _, mtx, dist, _, t = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

    # from calibration software
    fx = 550.541
    fy = 554.281
    cx = 371.127
    cy = 272.827
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([0.088292, -0.274401, 0.013027, 0.024800])

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(row,col,0)
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)

    # unit cell size
    # length = 13.59  # mm
    length = 30  # mm
    # length = 25.4

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        # imgfile = '../img/p2c_test.png'
        # img = cv2.imread(imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        key = cv2.waitKey(1) & 0xFF
        # if key == ord('\r'):  # ENTER
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
        if ret == True:
            # If found, add object points, image points (after refining them)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (col, row), corners_refined, ret)
            # corners_refined = np.squeeze(corners_refined)

            # one time capturing camera intrinsic parameters is unreliable
            # ret, mtx, dist, rvecs, tvecs1 =\
            #     cv2.calibrateCamera([objp], [corners_refined], gray.shape[::-1], None, None)

            _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners_refined, mtx, dist)

            R = cv2.Rodrigues(rvecs)[0]
            print ()
            print (np.array(tvecs)*length)
            print()
        cv2.imshow("Image", img)
        cv2.waitKey(1)


def homography(row, col):
    imgfile = '../img/checker_board_inclined.png'

    # Read source image.
    img = cv2.imread(imgfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
    if ret == True:
        # If found, add object points, image points (after refining them)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        detected = cv2.drawChessboardCorners(img, (row, col), corners_refined, ret)
        number_of_corners = corners_refined.shape[0]
        corners_refined = corners_refined.ravel().reshape(number_of_corners, 2)
        np.save("checker_board_inclined", corners_refined)

    # Four corners of the checkerboard in source image
    pts_src = np.array([corners_refined[0], corners_refined[2], corners_refined[9], corners_refined[11]])

    # Four corners of the box in destination image.
    pts_dst = np.array([[0, 0], [400, 0], [0, 400], [400, 400]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    img_out = cv2.warpPerspective(img, h, (600, 600))

    # choose center point as an input
    input = np.array([[580], [284], [1]])
    cv2.circle(img, (input[0],input[1]), 5, (255,255,255), -1)
    out = np.matrix(h)*input
    out = out/out[2]
    cv2.circle(img_out, (out[0], out[1]), 10, (255,255,255), -1)

    # Display images
    cv2.imshow("Source Image", img)
    cv2.imshow("Warped Source Image", img_out)
    cv2.waitKey(0)

def resize_img():
    filename = "../img/block_sample_drawing3.png"
    img = cv2.imread(filename)

    # resize image
    scale_percent = 80  # percent of original size
    dx = int(img.shape[0] * scale_percent / 100)
    dy = int(img.shape[1] * scale_percent / 100)
    dim = (dx, dy)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    cv2.imshow("original image", img)
    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vconcat_resize_min(self, im_list):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=cv2.INTER_CUBIC)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(self, im_list):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=cv2.INTER_CUBIC)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    elif style=='dashed':
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def downsample_naive(img, downsample_factor):
    """
    Naively downsamples image without LPF.
    """
    new_img = img.copy()
    new_img = new_img[::downsample_factor]
    new_img = new_img[:, ::downsample_factor]
    return new_img

def extract_extreme_points():
    # leftmost, rightmost, topmost, bottommost
    pl = np.array(contour[contour[:,:,0].argmin()][0])
    pr = np.array(contour[contour[:,:,0].argmax()][0])
    pt = np.array(contour[contour[:,:,1].argmin()][0])
    pb = np.array(contour[contour[:,:,1].argmax()][0])
    print (pl, pr, pt, pb)

def image_moment():
    mmt = cv2.moments(contour)
    for key,val in mmt.items():
        print(key, val)

    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])

    # mu20_ = mmt['m20']/mmt['m00'] - cx*cx
    # mu02_ = mmt['m02']/mmt['m00'] - cy*cy
    # mu11_ = mmt['m11']/mmt['m00'] - cx*cy
    # theta = 0.5*np.arctan2(2*mu11_, mu20_-mu02_)

def load_intrinsics(self, filename):
    # load calibration data
    with np.load(filename) as X:
        _, mtx, dist, _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
    return mtx, dist

if __name__ == "__main__":
    # showImage()
    # showVideo()
    # writeVideo()
    # drawing()
    # mouseBrush()
    # trackbar()
    # pixelExtract()
    # pixelSplit()
    # addImage('../img/img_color_checkerboard.png', '../img/img_color_pegboard.png')
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
    # correlated()
    # rotate_image()
    # scaling_image()
    # checkerboard_calibration(8,6)
    transform_pixel_to_checkerboard(5,7)
    # homography(4,3)
    # resize_img()
    # image stack (concatenate)
    # images = self.hconcat_resize_min([img_color, depth_masked, cv2.cvtColor(red_masked, cv2.COLOR_GRAY2BGR)])
    # img = np.zeros((500, 500, 3), np.uint8)
    # drawline(img, [0, 0], [400, 400], (255, 0, 0), 1, 'dashed', gap=10)