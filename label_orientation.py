import numpy as np
import pandas as pd
import os
import sys
from PyQt5.QtCore import Qt, QPoint, QCoreApplication, QRect, QPointF
from PyQt5.QtWidgets import QProgressDialog, QApplication, QLabel, QMainWindow, QAction, QInputDialog, QFileDialog
from PyQt5.QtGui import QPainter, QPixmap, QImage, QColor, QPen
import cv2
import scipy.io
from functools import partial
import math
from PIL import Image
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--start", type=int, default=1, help="lower limit of the image index (inclusive)")
# parser.add_argument("--end", type=int, default=-1, help="higher limit of the image index (exclusive)")
# opt = parser.parse_args()


class FrameBox(QMainWindow):

    def __init__(self):
        super().__init__()
        self.frame_qpixmap = QPixmap()
        self.initUI()
        self.initParameters()
        self.initAnnotation()
        self.initMenu()
        self.setFrame(self.index)


    def initUI(self):
        # set size of the widget to the size of frames
        # self.setGeometry(100, 100, self.frame_width, self.frame_height)

        # self.info_bar = QLabel()
        self.frame_display = QLabel(self)
        # self.frame_display.resize(self.frame_width, self.frame_height)

        self.frame_display.setMouseTracking(True)
        self.setMouseTracking(True)

        self.show()
        self.statusBar().showMessage('initUI... done')

    # read k, points of neg, nuc, pos, and pos_o
    def initAnnotation(self):
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix('csv')
        file_name = file_dialog.getOpenFileName(self, 'Select Annotation File')
        if file_name[0]:
            self.df = pd.read_csv(file_name[0])
            self.df['deleted'] = 0
            # self.paths = self.df['path'].unique()
            self.hi_idx = len(self.df) - 1
            self.lo_idx = 0
            self.statusBar().showMessage('initAnnotation... done')

    def initParameters(self):
        self.types_of_annotations = ["pts_pos", "pts_neg"]
        self.theta = 0
        self.mode = 0  # 1 is update, 2 is delete, 0 doesn't change the existing annotations
        self.cur_annotation_type = ""
        self.index = 0
        self.x = 0
        self.y = 0
        self.modify = False
        self.scale = 1.0

    # init menu bar
    def initMenu(self):
        menubar = self.menuBar()
        # append = menubar.addMenu("Append")
        update = menubar.addMenu("Update")

        # add actions to append and update
        for i in range(len(self.types_of_annotations)):
            annotation = self.types_of_annotations[i]
            # new_action_append = QAction(annotation, self)
            # new_action_append.setShortcut("Ctrl+" + str(i + 1))
            # new_action_append.setShortcut("" + str(i + 1))
            new_action_update = QAction(annotation, self)
            new_action_update.setShortcut("" + str(i + 1))
            # pass argument using partial
            # new_action_append.triggered.connect(partial(self.appendAnnotations, i))
            new_action_update.triggered.connect(partial(self.updateAnnotations, i))

            # append.addAction(new_action_append)
            update.addAction(new_action_update)

        change_frame_num = menubar.addMenu("Change index")
        change_frame_num_action = QAction("Set Index Number", self)
        change_frame_num_action.setShortcut("Ctrl+f")
        change_frame_num_action.triggered.connect(self.change_frame_number)
        change_frame_num.addAction(change_frame_num_action)

        save_annotation = menubar.addMenu("Save")
        save_annotation_action = QAction("Select Directory", self)
        save_annotation_action.setShortcut("Ctrl+s")
        save_annotation_action.triggered.connect(self.saveAllAnnotationsToFile)
        save_annotation.addAction(save_annotation_action)

    def updateAnnotations(self, index):
        cur = self.types_of_annotations[index]
        self.statusBar().showMessage("Update mode selected: {}".format(cur))
        self.mode = 1
        self.cleanUpCurrentAnnotation()
        self.cur_annotation_type = cur

    # New Feature:
    # search for point most close to the given position, if the distance is within radius, we delete this point
    def deleteAnnotations(self, index):
        self.df = self.df.drop(index)
        self.setFrame(self.frame_num)
        self.update()

    def setFrame(self, index, scale=1):
        # assert(frame_num > 0)
        # assert(frame_num <= self.max_frame_num)

        self.statusBar().showMessage("set index number to {}".format(index))
        self.frame_num = index  # update current frame number

        # get width and height of the current image
        # frame_file_name = self.path_to_images + '/%06d.jpg' % (frame_num + 1)
        frame_file_name = self.df.loc[index].path

        image = cv2.imread(frame_file_name)

        h, w, c = image.shape

        x, y = self.df.loc[index][['x', 'y']].values
        x = int(x * w)
        y = int(y * h)

        window_size = 4 * (int(64 * scale) // 4)

        box = [max(0, x - math.floor(window_size / 2)), max(0, y - math.floor(window_size / 2)),
               min(w, x + math.ceil(window_size / 2)), min(h, y + math.ceil(window_size / 2))]

        window = np.zeros((window_size, window_size, c), np.uint8)
        x_start = math.floor(window_size / 2) - x + box[0]
        y_start = math.floor(window_size / 2) - y + box[1]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        window[y_start:y_start + y_width, x_start:x_start + x_width, :] = image[box[1]:box[3], box[0]:box[2], :]

        self.frame_height, self.frame_width, _ = window.shape

        # adjust window size
        ratio = min(300 / self.frame_width, 300 / self.frame_height)
        # ratio = min(1200 / self.frame_width, 1200 / self.frame_height)
        self.frame_width = int(self.frame_width * ratio)
        self.frame_height = int(self.frame_height * ratio)

        # reset the size of the window
        self.setGeometry(1300, -1000, self.frame_width, self.frame_height)
        # self.setGeometry(0, 0, self.frame_width, self.frame_height)
        self.frame_display.resize(self.frame_width, self.frame_height)

        image_data = np.array(window)

        # add a black section on the bottom of the original image
        image_data = cv2.copyMakeBorder(image_data, 0, int(image_data.shape[0] * 0.1), 0, 0, cv2.BORDER_CONSTANT)

        q_image = QImage(image_data, image_data.shape[1], image_data.shape[0], QImage.Format_RGB888)
        q_image = q_image.scaledToHeight(self.frame_height).scaledToWidth(self.frame_width)

        self.frame_qpixmap = QPixmap.fromImage(q_image)
        self.update()

        cls = self.df.loc[self.index][['class']].values
        theta = self.df.loc[self.index][['theta']].values

        cls = cls.astype(int)
        theta = theta.astype(float)

        if self.df.loc[index][['deleted']].values == 0:
            self.drawAngle(cls, theta)

        del q_image

    def addToCurrentAnnotation(self, x, y):
        if not self.mode == 0:
            self.cur_annotation = np.vstack((self.cur_annotation, [x, y]))
            self.setFrame(self.frame_num)

    def saveCurrentAnnotation(self):
        if self.cur_annotation_type == '' or self.mode == 3:
            return

        # class index, 0 for pos, 1 for neg
        c = ['pts_pos', 'pts_neg'].index(self.cur_annotation_type)

        # update mode
        if self.mode == 1:
            # new_df = pd.DataFrame(self.cur_annotation, columns=['x', 'y'])
            # new_df['path'] = self.paths[self.frame_num - 1]
            # new_df['class'] = c
            #
            # idx_to_drop = self.df[((self.df['path'] == self.paths[self.frame_num - 1]) &
            #                     (self.df['class'] == c))].index
            # self.df = self.df.drop(idx_to_drop)
            # self.df = self.df.append(new_df, ignore_index=True)
            self.df.loc[self.index].theta = self.theta

            self.setFrame(self.index)
            self.statusBar().showMessage("annotations updated")

    def saveAllAnnotationsToFile(self, file_name=None):
        if not file_name:
            file_dialog = QFileDialog()
            file_dialog.setDefaultSuffix('csv')
            file_name = file_dialog.getSaveFileName(self, 'Save File', '{}.csv'.format('Saved_Results/'))[0]
        self.saveToFile(file_name)

    def saveToFile(self, path_to_file):
        if path_to_file:
            # self.df = self.df.sort_values(by=['path'], ignore_index=True)
            self.df.to_csv(path_to_file, sep=',', index=False)
            self.statusBar().showMessage("{} saved!".format(path_to_file))

    def cleanUpCurrentAnnotation(self):
        # self.cur_annotation = np.empty((0, 1), float)
        self.theta = 0

    def drawAngle(self, cls, theta, r=0.25):
        if cls == 0:
            from_pts = np.array([[0.5, 0.5]])
            to_pts = np.array([[
                0.5 + np.sin(theta) * r,
                0.5 + np.cos(theta) * r
            ]])
            self.drawlines(from_pts, to_pts, Qt.blue)

        elif cls == 1:
            from_pts = np.array([
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5]
            ])
            to_pts = np.array([
                [0.5 + np.sin(theta) * r, 0.5 + np.cos(theta) * r],
                [0.5 + np.sin(theta + 2/3 * np.pi) * r, 0.5 + np.cos(theta + 2/3 * np.pi) * r],
                [0.5 + np.sin(theta + 4/3 * np.pi) * r, 0.5 + np.cos(theta + 4/3 * np.pi) * r]
            ])
            self.drawlines(from_pts, to_pts, Qt.red)
        self.update()

    def drawpoints(self, points, color, radius=3):
        painter = QPainter(self.frame_qpixmap)
        painter.setBrush(color)
        for point in points:
            # update: relative position
            center = QPoint(point[0] * self.frame_width, point[1] * self.frame_height)
            painter.drawEllipse(center, radius, radius)
        self.update()

    def drawlines(self, from_pts, to_pts, color):
        assert(len(from_pts) == len(to_pts))
        painter = QPainter(self.frame_qpixmap)
        painter.setPen(QPen(color, 4, Qt.SolidLine))
        # painter.setBrush(color)
        for i, _ in enumerate(from_pts):
            f = from_pts[i]  # from
            t = to_pts[i]    # to
            x1, x2 = f[0] * self.frame_width, t[0] * self.frame_width
            y1, y2 = f[1] * self.frame_height, t[1] * self.frame_height
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        self.update()

    def drawCurrentAnnotation(self):
        if self.cur_annotation_type in self.types_of_annotations:
            index = self.types_of_annotations.index(self.cur_annotation_type)
            colors = [Qt.blue, Qt.red, Qt.green, Qt.yellow]
            if self.cur_annotation.size > 0:
                self.drawpoints(self.cur_annotation, colors[index])


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.frame_qpixmap)

    def mouseMoveEvent(self, e):
        prev_x = self.x
        prev_y = self.y

        self.x = e.x() / self.frame_width
        self.y = e.y() / (self.frame_height / 1.1)

        if self.modify:
            v1 = np.array([prev_x - .5, prev_y - .5])
            v2 = np.array([self.x - .5, self.y - .5])
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            v2 = v2 / (np.linalg.norm(v2) + 1e-16)

            d_theta = np.arccos(np.dot(v1, v2))

            if ((v1[0]-0.5) * (v2[1]-0.5) - (v1[1]-0.5) * (v2[0]-0.5)) < -1e-5:
                d_theta *= -1.0

            self.theta = (self.theta - d_theta) % (2 * np.pi)
            self.df.at[self.index, 'theta'] = self.theta

            self.setFrame(self.index, self.scale)
            self.update()

    # modify position of a given defect (x, y) by off_x and off_y
    def modifyPosition(self, index, off_x=.0, off_y=.0):
        x, y = self.df.loc[index][['x', 'y']].values
        x += off_x
        y += off_y
        self.df.at[index, 'x'] = x
        self.df.at[index, 'y'] = y

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Right or key == Qt.Key_E:
            self.scale = 1.0
            if self.mode == 3:
                self.mode = 0
            if self.index < self.hi_idx - 1:
                self.index += 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.index, self.scale)

        elif key == Qt.Key_Left or key == Qt.Key_Q:
            self.scale = 1.0
            if self.mode == 3:
                self.mode = 0
            if self.frame_num > self.lo_idx:
                self.index -= 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.index, self.scale)

        elif key == Qt.Key_W:
            self.modifyPosition(self.index, off_y=-0.002)
            self.setFrame(self.index, self.scale)
        elif key == Qt.Key_S:
            self.modifyPosition(self.index, off_y=0.002)
            self.setFrame(self.index, self.scale)
        elif key == Qt.Key_A:
            self.modifyPosition(self.index, off_x=-0.002)
            self.setFrame(self.index, self.scale)
        elif key == Qt.Key_D:
            self.modifyPosition(self.index, off_x=0.002)
            self.setFrame(self.index, self.scale)
        # delete / undo
        elif key == Qt.Key_Tab:
            is_deleted = self.df.loc[self.index].deleted
            self.df.at[self.index, 'deleted'] = 1 - is_deleted
            self.setFrame(self.index, self.scale)
        #
        # # reset annotation mode to 0
        # elif key == Qt.Key_C:
        #     self.mode = 0
        #     self.cleanUpCurrentAnnotation()
        #     self.cur_annotation_type = ""
        #     self.setFrame(self.frame_num)
        #     self.statusBar().showMessage("Cancel annotation")
        # save cur_annotation according to self.mode
        elif key == Qt.Key_Return or key == Qt.Key_Space:
            self.saveCurrentAnnotation()

    def change_frame_number(self):
        user_frame_num, ok = QInputDialog.getInt(
            self, "Change Frame Number", "Please enter a number from 0 to {}".format(len(self.df)))
        if ok:
            if user_frame_num < len(self.df) and user_frame_num >= 0:
                self.index = user_frame_num
                self.setFrame(self.index)
                self.cleanUpCurrentAnnotation()
            else:
                self.statusBar().showMessage("The input is not valid, please try again")

    def mousePressEvent(self, e):
        self.modify = True

    def mouseReleaseEvent(self, e):
        self.modify = False

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        new_scale = self.scale - 0.05 * (delta / 60)
        new_scale = 0.05 * (new_scale // 0.05)
        self.scale = np.clip(new_scale, 0.5, 5.0).astype(float)
        self.setFrame(self.index, self.scale)


def run():
    app = QApplication(sys.argv)
    fb = FrameBox()
    try:
        fb.show()
    except KeyboardInterrupt:
        print('Interrupted, saving results to Saved_Results/temp.csv ...')
        fb.saveAllAnnotationsToFile('Saved_Results/temp.csv')
        print('Saved')
    finally:
        sys.exit(app.exec_())


if __name__ == '__main__':
    run()
