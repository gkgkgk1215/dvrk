"""Shared methods, to be loaded in other code.
"""
import sys
import numpy as np
from os import path


ESC_KEYS = [27, 1048603]
MILLION = float(10**6)

def rad_to_deg(rad):
    return np.array(rad) *180./np.pi


def deg_to_rad(deg):
    return np.array(deg) *np.pi/180.

def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def call_wait_key(nothing=None):
    """Call this like: `utils.call_wait_key( cv2.imshow(...) )`.
    """
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program...")
        sys.exit()

def load_mapping_table(row_board, column_board, file_name, cloth_height=0.005):
    """

    :param row_board: number of rows.
    :param column_board: number of columns.
    :param file_name: name of the calibration file
    :param cloth_height: height offset
    :return: data from calibration
    """
    if path.exists(file_name):
        # import data from file
        data_default = np.loadtxt(file_name, delimiter=',')
    else:
        # if file does not exist, set default
        data_default = np.zeros((row_board * column_board, 5))

    cnt = 0
    for i in range(row_board):
        for j in range(column_board):
            data_default[cnt, 0] = -1 + j * 0.4
            data_default[cnt, 1] = -1 + i * 0.4
            data_default[cnt, 4] = data_default[cnt, 4] + cloth_height
            cnt += 1
    data = data_default
    # print data

    data_square = np.zeros((row_board + 1, column_board + 1, 5))
    for i in range(row_board):
        for j in range(column_board):
            data_square[i, j, :] = data[column_board * j + i, 0:5]

    for i in range(row_board):
        data_square[i, column_board, :] = data_square[i, column_board - 1, :]
    for j in range(column_board):
        data_square[row_board, j] = data_square[row_board - 1, j]

    return data_square

def transform_CB2PSM(x, y, row_board, col_board, data_square):
    """Minho's code, for calibation, figure out the PSM coordinates.

    Parameters (x,y) should be in [-1,1] (if not we clip it) and represent
    the coordinate range over the WHITE CLOTH BACKGROUND PLANE (or a
    'checkboard' plane). We then convert to a PSM coordinate.

    :param row_board: number of rows.
    :param col_board: number of columns.
    :param data_square: data from calibration.
    """
    if x>1: x=1.0
    if x<-1: x=-1.0
    if y>1:  y=1.0
    if y<-1: y=-1.0

    for i in range(row_board):
        for j in range(col_board):
            if x == data_square[row_board-1, j, 0] and y == data_square[i, col_board-1, 1]: # corner point (x=1,y=1)
                return data_square[row_board-1,col_board-1,2:5]
            else:
                if x == data_square[row_board-1, j, 0]:  # border line of x-axis
                    if data_square[i, j, 1] <= y and y < data_square[i, j + 1, 1]:
                        y1 = data_square[row_board-1, j, 1]
                        y2 = data_square[row_board-1, j+1, 1]
                        Q11 = data_square[row_board-1, j, 2:5]
                        Q12 = data_square[row_board-1, j+1, 2:5]
                        return (y2-y)/(y2-y1)*Q11 + (y-y1)/(y2-y1)*Q12
                elif y == data_square[i, col_board-1, 1]:  # border line of y-axis
                    if data_square[i, j, 0] <= x and x < data_square[i + 1, j, 0]:
                        x1 = data_square[i, col_board-1, 0]
                        x2 = data_square[i+1, col_board-1, 0]
                        Q11 = data_square[i, col_board-1, 2:5]
                        Q21 = data_square[i+1, col_board-1, 2:5]
                        return (x2-x)/(x2-x1)*Q11 + (x-x1)/(x2-x1)*Q21
                else:
                    if data_square[i,j,0] <= x and x < data_square[i+1,j,0]:
                        if data_square[i,j,1] <= y and y < data_square[i,j+1,1]:
                            x1 = data_square[i, j, 0]
                            x2 = data_square[i+1, j, 0]
                            y1 = data_square[i, j, 1]
                            y2 = data_square[i, j+1, 1]
                            Q11 = data_square[i, j, 2:5]
                            Q12 = data_square[i, j+1, 2:5]
                            Q21 = data_square[i+1, j, 2:5]
                            Q22 = data_square[i+1, j+1, 2:5]
                            if x1==x2 or y1==y2:
                                return []
                            else:
                                return 1/(x2-x1)/(y2-y1)*(Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1))


def move_p_from_net_output(x, y, dx, dy, row_board, col_board, data_square, p):
    """Minho's code, for calibration, processes policy network output.

    :params (x, y, dx, dy): outputs from the neural network.
    :param row_board: number of rows.
    :param col_board: number of columns.
    :param data_square: data from calibration.
    :param p: An instance of `dvrkClothSim`.
    """
    pickup_pos = transform_CB2PSM(x,
                                  y,
                                  row_board,
                                  col_board,
                                  data_square)
    release_pos_temp = transform_CB2PSM(x+dx,
                                        y+dy,
                                        row_board,
                                        col_board,
                                        data_square)
    release_pos = np.array([release_pos_temp[0], release_pos_temp[1]])
    # print pickup_pos, release_pos
    # just checking if the ROS input is fine
    # user_input = raw_input("Are you sure the values to input to the robot arm?(y or n)")
    # if user_input == "y":
    p.move_pose_pickup(pickup_pos, release_pos, 0, 'rad')
