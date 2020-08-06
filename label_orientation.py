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

    # def appendAnnotations(self, index):
    #     cur = self.types_of_annotations[index]
    #     self.statusBar().showMessage("Append mode selected: {}".format(cur))
    #     self.mode = 2
    #     self.cleanUpCurrentAnnotation()
    #     self.cur_annotation_type = cur

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
        print(frame_file_name)

        image = cv2.imread(frame_file_name)

        h, w, c = image.shape

        x, y = self.df.loc[index][['x', 'y']].values
        x = int(x * w)
        y = int(y * h)

        window_size = int(64 * scale)

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

        self.drawAngle(cls, theta)

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

        # append mode
        # elif self.mode == 2:
        #     new_df = pd.DataFrame(self.cur_annotation, columns=['x', 'y'])
        #     new_df['path'] = self.paths[self.frame_num - 1]
        #     new_df['class'] = c
        #
        #     self.df = self.df.append(new_df, ignore_index=True)
        #
        #     self.setFrame(self.frame_num)
        #     self.statusBar().showMessage("annotations appended")

    def saveAllAnnotationsToFile(self, file_name=None):
        if not file_name:
            file_dialog = QFileDialog()
            file_dialog.setDefaultSuffix('csv')
            file_name = file_dialog.getSaveFileName(self, 'Save File', '{}.csv'.format('Saved_Results/'))[0]
        self.saveToFile(file_name)

    def saveToFile(self, path_to_file):
        if path_to_file:
            self.df = self.df.sort_values(by=['path'], ignore_index=True)
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

    # def drawPrevAnnotation(self):
    #     if self.frame_num <= 1:
    #         return
    #     cur_path = self.paths[self.frame_num - 1]
    #     prev_path = self.paths[self.frame_num - 2]
    #
    #     cur_df = self.df[self.df['path'] == cur_path]
    #     prev_df = self.df[self.df['path'] == prev_path]
    #
    #     cur_pos = cur_df[cur_df['class'] == 0][['x', 'y']].values
    #     cur_neg = cur_df[cur_df['class'] == 1][['x', 'y']].values
    #
    #     prev_pos = prev_df[prev_df['class'] == 0][['x', 'y']].values
    #     prev_neg = prev_df[prev_df['class'] == 1][['x', 'y']].values
    #
    #     # assign everu prev annotation with one corresponding label in current frame (if applicable)
    #     pos_dis = np.array([[i, j, np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2)] for i, r1 in enumerate(prev_pos)
    #                         for j, r2 in enumerate(cur_pos)])
    #     pos_dis = pos_dis[pos_dis[:, 2] < 0.02, :]
    #
    #     # assign everu prev annotation with one corresponding label in current  frame( if applicable)
    #     neg_dis = np.array([[i, j, np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2)] for i, r1 in enumerate(prev_neg)
    #                         for j, r2 in enumerate(cur_neg)])
    #     neg_dis = neg_dis[neg_dis[:, 2] < 0.02, :]
    #
    #     pos_map = np.empty([0, 2], dtype=int)
    #     for idx in np.argsort(pos_dis[:, 2]):
    #         i, j = int(pos_dis[idx, 0]), int(pos_dis[idx, 1])
    #         if (i not in pos_map[:, 0]) and (j not in pos_map[:, 1]):
    #             pos_map = np.vstack((pos_map, np.array([i, j])))
    #
    #     neg_map = np.empty([0, 2], dtype=int)
    #     for idx in np.argsort(neg_dis[:, 2]):
    #         i, j = int(neg_dis[idx, 0]), int(neg_dis[idx, 1])
    #         if (i not in neg_map[:, 0]) and (j not in neg_map[:, 1]):
    #             neg_map = np.vstack((neg_map, np.array([i, j])))
    #
    #     self.drawlines(prev_pos[pos_map[:, 0], :], cur_pos[pos_map[:, 1], :], QColor(0, 0, 255, 255))
    #     self.drawlines(prev_neg[neg_map[:, 0], :], cur_neg[neg_map[:, 1], :], QColor(255, 0, 0, 255))
        # self.drawpoints(prev_pos[pos_map[:, 0], :], QColor(0, 0, 255, 255), radius=2)
        # self.drawpoints(prev_neg[neg_map[:, 0], :], QColor(255, 0, 0, 255), radius=2)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.frame_qpixmap)

    def mouseMoveEvent(self, e):
        prev_x = self.x
        prev_y = self.y

        self.x = e.x() / self.frame_width
        self.y = e.y() / (self.frame_height / 1.1)

        if self.modify:
            self.theta = (self.theta + 2 * (prev_y - self.y)) % (2 * np.pi)
            self.df.at[self.index, 'theta'] = self.theta

            self.setFrame(self.index)
            self.update()

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_D or key == Qt.Key_Right:
            if self.mode == 3:
                self.mode = 0
            if self.index < self.hi_idx - 1:
                self.index += 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.index)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            if self.mode == 3:
                self.mode = 0
            if self.frame_num > self.lo_idx:
                self.index -= 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.index)
        # new feature:
        elif key == Qt.Key_Tab:
            self.mode = 3

        # reset annotation mode to 0
        elif key == Qt.Key_C:
            self.mode = 0
            self.cleanUpCurrentAnnotation()
            self.cur_annotation_type = ""
            self.setFrame(self.frame_num)
            self.statusBar().showMessage("Cancel annotation")
        # save cur_annotation according to self.mode
        elif key == Qt.Key_Return or key == Qt.Key_Space:
            self.saveCurrentAnnotation()

    def change_frame_number(self):
        user_frame_num, ok = QInputDialog.getInt(
            self, "Change Frame Number", "Please enter a number from 0 to {}".format(self.max_frame_num))
        if ok:
            if user_frame_num <= self.max_frame_num and user_frame_num > 0:
                self.frame_num = user_frame_num
                self.setFrame(self.frame_num)
                self.cleanUpCurrentAnnotation()
            else:
                self.statusBar().showMessage("The input is not valid, please try again")

    def mousePressEvent(self, e):
        self.modify = True
        # if self.mode == 1 or self.mode == 2:
        #     text = "x: {0}, y: {1}".format(self.x, self.y)
        #     self.addToCurrentAnnotation(self.x, self.y)
        #     self.statusBar().showMessage("{} selected".format(text))
        # elif self.mode == 3:
        #     self.deleteAnnotations(self.index)

    def mouseReleaseEvent(self, e):
        self.modify = False


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
