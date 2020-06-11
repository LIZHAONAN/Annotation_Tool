'''
Tool for annotate images
Author: Zhaonan Li zli@brandeis.edu
Created at: 6/20/2019
'''
import numpy as np
import pandas as pd
import os
import sys
from PyQt5.QtCore import Qt, QPoint, QCoreApplication, QRect
from PyQt5.QtWidgets import QProgressDialog, QApplication, QLabel, QMainWindow, QAction, QInputDialog, QFileDialog
from PyQt5.QtGui import QPainter, QPixmap, QImage
import cv2
import scipy.io
from functools import partial
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=1, help="lower limit of the image index (inclusive)")
parser.add_argument("--end", type=int, default=-1, help="higher limit of the image index (exclusive)")
opt = parser.parse_args()


class FrameBox(QMainWindow):

    def __init__(self, lo=1, hi=-1):
        super().__init__()
        self.lo_idx = lo  # lower limit of the image index (inclusive)
        self.hi_idx = hi  # higher limit of the image index (exclusive)
        self.frame_qpixmap = QPixmap()
        self.initUI()
        self.initParameters()
        self.initAnnotation()
        self.initMenu()
        self.setFrame(self.frame_num)


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
            self.paths = self.df['path'].unique()
            self.max_frame_num = len(self.paths)
            self.statusBar().showMessage('initAnnotation... done')

    def initParameters(self):
        self.types_of_annotations = ["pts_pos", "pts_neg", "pts_non"]
        self.x = 0
        self.y = 0
        self.mode = 0  # 1 is update, 2 is append, 3 is delete, 0 doesn't change the existing annotations
        self.cur_annotation = np.empty((0, 2), float)
        self.cur_annotation_type = ""
        if self.hi_idx == -1:
            self.hi_idx = self.max_frame_num + 1
        self.frame_num = self.lo_idx  # this number corresponds to the image_index

    # init menu bar
    def initMenu(self):
        menubar = self.menuBar()
        append = menubar.addMenu("Append")
        update = menubar.addMenu("Update")

        # add actions to append and update
        for i in range(len(self.types_of_annotations)):
            annotation = self.types_of_annotations[i]
            new_action_append = QAction(annotation, self)
            new_action_append.setShortcut("Ctrl+" + str(i + 1))
            new_action_update = QAction(annotation, self)
            new_action_update.setShortcut("Shift+" + str(i + 1))
            # pass argument using partial
            new_action_append.triggered.connect(partial(self.appendAnnotations, i))
            new_action_update.triggered.connect(partial(self.updateAnnotations, i))

            append.addAction(new_action_append)
            update.addAction(new_action_update)

        change_frame_num = menubar.addMenu("Change Frame")
        change_frame_num_action = QAction("Set Frame Number", self)
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

    def appendAnnotations(self, index):
        cur = self.types_of_annotations[index]
        self.statusBar().showMessage("Append mode selected: {}".format(cur))
        self.mode = 2
        self.cleanUpCurrentAnnotation()
        self.cur_annotation_type = cur

    # New Feature:
    # search for point most close to the given position, if the distance is within radius, we delete this point
    def deleteAnnotations(self, x, y, radius=0.02):
        cur_pts = self.df[self.df['path'] == self.paths[self.frame_num - 1]][['x', 'y', 'class']].values
        cur_dis = np.sqrt(np.sum(np.square(cur_pts[:, :2] - [x, y]), axis=1))
        if np.min(cur_dis) < radius:
            cur_pts = np.delete(cur_pts, np.argmin(cur_dis), 0)
            new_df = pd.DataFrame(cur_pts, columns=['x', 'y', 'class'])
            new_df['path'] = self.paths[self.frame_num - 1]

            idx_to_drop = self.df[(self.df['path'] == self.paths[self.frame_num - 1])].index
            self.df = self.df.drop(idx_to_drop)
            self.df = self.df.append(new_df, ignore_index=True)
            self.setFrame(self.frame_num)


    def setFrame(self, frame_num):
        assert(frame_num > 0)
        assert(frame_num <= self.max_frame_num)

        self.statusBar().showMessage("set frame number to {}".format(frame_num))

        self.frame_num = frame_num  # update current frame number

        # get width and height of the current image
        # frame_file_name = self.path_to_images + '/%06d.jpg' % (frame_num + 1)
        frame_file_name = self.paths[frame_num - 1]
        print(frame_file_name)
        image_data = cv2.imread(frame_file_name)
        self.frame_height, self.frame_width, _ = image_data.shape

        # adjust window size
        # ratio = min(1000 / self.frame_width, 1000 / self.frame_height)
        ratio = min(800 / self.frame_width, 800 / self.frame_height)
        self.frame_width = int(self.frame_width * ratio)
        self.frame_height = int(self.frame_height * ratio)

        # reset the size of the window
        # self.setGeometry(1300, -1000, self.frame_width, self.frame_height)
        self.setGeometry(0, 0, self.frame_width, self.frame_height)
        self.frame_display.resize(self.frame_width, self.frame_height)

        image_data = np.array(image_data)
        # add a black section on the bottom of the original image
        image_data = cv2.copyMakeBorder(image_data, 0, int(image_data.shape[0] * 0.1), 0, 0, cv2.BORDER_CONSTANT)

        q_image = QImage(image_data, image_data.shape[1], image_data.shape[0], QImage.Format_RGB888)
        q_image = q_image.scaledToHeight(self.frame_height).scaledToWidth(self.frame_width)

        self.frame_qpixmap = QPixmap.fromImage(q_image)
        self.update()

        cur_path = self.paths[frame_num - 1]
        cur_df = self.df[self.df['path'] == cur_path]

        pts_pos = cur_df[cur_df['class'] == 0][['x', 'y']].values
        pts_neg = cur_df[cur_df['class'] == 1][['x', 'y']].values
        # pts_pos_o = self.pts_pos_o[frame_num][0]
        # pts_nuc = self.pts_nuc[frame_num][0]

        if pts_pos.size > 0:
            self.drawpoints(pts_pos, Qt.blue)
        if pts_neg.size > 0:
            self.drawpoints(pts_neg, Qt.red)
        # if pts_pos_o.size > 0:
        #     self.drawpoints(pts_pos_o, Qt.green)
        # if pts_nuc.size > 0:
        #     self.drawpoints(pts_nuc, Qt.yellow)

        self.drawCurrentAnnotation()

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
            new_df = pd.DataFrame(self.cur_annotation, columns=['x', 'y'])
            new_df['path'] = self.paths[self.frame_num - 1]
            new_df['class'] = c

            idx_to_drop = self.df[((self.df['path'] == self.paths[self.frame_num - 1]) &
                                (self.df['class'] == c))].index
            self.df = self.df.drop(idx_to_drop)
            self.df = self.df.append(new_df, ignore_index=True)

            self.setFrame(self.frame_num)
            self.statusBar().showMessage("annotations updated")

        # append mode
        elif self.mode == 2:
            new_df = pd.DataFrame(self.cur_annotation, columns=['x', 'y'])
            new_df['path'] = self.paths[self.frame_num - 1]
            new_df['class'] = c

            self.df = self.df.append(new_df, ignore_index=True)

            self.setFrame(self.frame_num)
            self.statusBar().showMessage("annotations appended")

    def saveAllAnnotationsToFile(self):
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix('csv')
        saved_file_name = file_dialog.getSaveFileName(self, 'Save File', '{}.csv'.format('Saved_Results/'))[0]
        self.saveToFile(saved_file_name)

    def saveToFile(self, path_to_file):
        if path_to_file:
            self.df = self.df.sort_values(by=['path'], ignore_index=True)
            self.df.to_csv(path_to_file, sep=',', index=False)
            self.statusBar().showMessage("{} saved!".format(path_to_file))

    def cleanUpCurrentAnnotation(self):
        self.cur_annotation = np.empty((0, 2), float)

    def drawpoints(self, points, color):
        painter = QPainter(self.frame_qpixmap)
        painter.setBrush(color)
        for point in points:
            # update: relative position
            center = QPoint(point[0] * self.frame_width, point[1] * self.frame_height)
            painter.drawEllipse(center, 3, 3)
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
        self.x = e.x() / self.frame_width
        self.y = e.y() / (self.frame_height / 1.1)

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_D or key == Qt.Key_Right:
            if self.mode == 3:
                self.mode = 0
            if self.frame_num < self.hi_idx - 1:
                self.frame_num += 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.frame_num)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            if self.mode == 3:
                self.mode = 0
            if self.frame_num > self.lo_idx:
                self.frame_num -= 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.frame_num)
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
                self.cleanUpCurrentAnnotation
            else:
                self.statusBar().showMessage("The input is not valid, please try again")

    def mousePressEvent(self, e):
        if self.mode == 1 or self.mode == 2:
            text = "x: {0}, y: {1}".format(self.x, self.y)
            self.addToCurrentAnnotation(self.x, self.y)
            self.statusBar().showMessage("{} selected".format(text))
        elif self.mode == 3:
            self.deleteAnnotations(self.x, self.y)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    fb = FrameBox(opt.start, opt.end)
    fb.show()
    sys.exit(app.exec_())