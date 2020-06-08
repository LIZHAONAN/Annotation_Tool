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

class FrameBox(QMainWindow):

    def __init__(self):
        self.frame_qpixmap = QPixmap()
        super().__init__()
        self.initParameters()
        self.initUI()
        # self.initFrames()
        self.initMenu()
        self.initAnnotation()
        self.setFrame(self.frame_num)

    def initParameters(self):
        self.types_of_annotations = ["pts_pos", "pts_neg", "pts_non"]
        self.x = 0
        self.y = 0
        self.mode = 0  # 1 is update, 2 is append, 3 is delete, 0 doesn't change the existing annotations
        self.cur_annotation = np.empty((0, 2), float)
        self.cur_annotation_type = ""
        # self.path_to_images = 'Images/images'
        self.path_to_annot = '18000_4c.mat'
        self.frame_num = 0 # this number corresponds to the image_index

    # examine the images folder, obtain following variables:
    # - total frame number
    # def initFrames(self):
        # get total number of frames
        # file_names = [int(x.split('.jpg')[0]) for x in os.listdir(self.path_to_images) if '.jpg' in x]
        # self.max_frame_num = max(file_names)

        # self.sizes_of_images = np.zeros((self.max_frame_num, 2)) # indices in these two array should start with 0
        # self.resize_ratio = np.zeros(self.max_frame_num) # indices in these two array should start with 0
        # # obtain sizes of images
        # for i in range(self.max_frame_num):
        #     file_name = os.path.join(self.path_to_images, '%06d.jpg' % (i + 1))
        #     image = cv2.imread(file_name)

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
            # mat = scipy.io.loadmat(file_name[0])
            self.df = pd.read_csv(file_name[0])
            self.paths = self.df['path'].unique()
            self.max_frame_num = len(self.paths)

            # self.k = mat['k'][0]
            # self.pts_neg = mat['pts_neg'][0]
            # self.pts_nuc = mat['pts_nuc'][0]
            # self.pts_pos = mat['pts_pos'][0]
            # self.pts_pos_o = mat['pts_pos_o'][0]
            self.statusBar().showMessage('initAnnotation... done')
        # else:
        #     # initialize result
        #     self.k = 0
        #     self.pts_neg = [[] for x in range(self.max_frame_num)]
        #     self.pts_nuc = [[] for x in range(self.max_frame_num)]
        #     self.pts_pos = [[] for x in range(self.max_frame_num)]
        #     self.pts_pos_o = [[] for x in range(self.max_frame_num)]
        #     self.statusBar().showMessage('initAnnotation... done')

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
        # if self.frame_num < 11000:
        cur = self.types_of_annotations[index]
        self.statusBar().showMessage("Update mode selected: {}".format(cur))
        self.mode = 1
        self.cleanUpCurrentAnnotation()
        self.cur_annotation_type = cur
        # else:
        #     self.statusBar().showMessage("Update mode selected: {}".format("Update mode is disabled when frame number > 11000!"))

    def appendAnnotations(self, index):
        cur = self.types_of_annotations[index]
        self.statusBar().showMessage("Append mode selected: {}".format(cur))
        self.mode = 2
        self.cleanUpCurrentAnnotation()
        self.cur_annotation_type = cur

    # New Feature:
    # search for point most close to the given position, if the distance is within radius, we delete this point
    def deleteAnnotations(self, x, y, radius=0.02):
        # cur_pts_neg = self.pts_neg[self.frame_num][0]
        # cur_pts_pos = self.pts_pos[self.frame_num][0]
        # cur_pts_nuc = self.pts_nuc[self.frame_num][0]
        # cur_pts_pos_o = self.pts_pos_o[self.frame_num][0]
        # if len(cur_pts_neg) > 0:
        #     dis = np.sqrt(np.sum(np.square(cur_pts_neg - [x, y]), axis=1))
        #     if np.min(dis) < radius:
        #         cur_pts_neg = np.delete(cur_pts_neg, np.argmin(dis), 0)
        #         self.pts_neg[self.frame_num][0] = cur_pts_neg
        #         self.setFrame(self.frame_num)
        #
        # if len(cur_pts_pos) > 0:
        #     dis = np.sqrt(np.sum(np.square(cur_pts_pos - [x, y]), axis=1))
        #     if np.min(dis) < radius:
        #         cur_pts_pos = np.delete(cur_pts_pos, np.argmin(dis), 0)
        #         self.pts_pos[self.frame_num][0] = cur_pts_pos
        #         self.setFrame(self.frame_num)
        #
        # if len(cur_pts_pos_o) > 0:
        #     dis = np.sqrt(np.sum(np.square(cur_pts_pos_o - [x, y]), axis=1))
        #     if np.min(dis) < radius:
        #         cur_pts_pos_o = np.delete(cur_pts_pos_o, np.argmin(dis), 0)
        #         self.pts_pos_o[self.frame_num][0] = cur_pts_pos_o
        #         self.setFrame(self.frame_num)
        #
        # if len(cur_pts_nuc) > 0:
        #     dis = np.sqrt(np.sum(np.square(cur_pts_nuc - [x, y]), axis=1))
        #     if np.min(dis) < radius:
        #         cur_pts_nuc = np.delete(cur_pts_nuc, np.argmin(dis), 0)
        #         self.pts_nuc[self.frame_num][0] = cur_pts_nuc
        #         self.setFrame(self.frame_num)

        cur_pts = self.df[self.df['path'] == self.paths[self.frame_num]][['x', 'y', 'class']].values
        cur_dis = np.sqrt(np.sum(np.square(cur_pts[:, :2] - [x, y]), axis=1))
        if np.min(cur_dis) < radius:
            cur_pts = np.delete(cur_pts, np.argmin(cur_dis), 0)
            new_df = pd.DataFrame(cur_pts, columns=['x', 'y', 'class'])
            new_df['path'] = self.paths[self.frame_num]

            idx_to_drop = self.df[(self.df['path'] == self.paths[self.frame_num])].index
            self.df = self.df.drop(idx_to_drop)
            self.df = self.df.append(new_df, ignore_index=True)
            self.setFrame(self.frame_num)


    def setFrame(self, frame_num):
        assert(frame_num >= 0)
        assert(frame_num < self.max_frame_num)

        self.statusBar().showMessage("set frame number to {}".format(frame_num))

        self.frame_num = frame_num  # update current frame number

        # get width and height of the current image
        # frame_file_name = self.path_to_images + '/%06d.jpg' % (frame_num + 1)
        frame_file_name = self.paths[frame_num]
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

        cur_path = self.paths[frame_num]
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
            # if self.cur_annotation_type == 'pts_neg':
            #     self.pts_neg[self.frame_num][0] = self.cur_annotation
            # elif self.cur_annotation_type == 'pts_pos':
            #     self.pts_pos[self.frame_num][0] = self.cur_annotation
            # elif self.cur_annotation_type == 'pts_pos_o':
            #     self.pts_pos_o[self.frame_num][0] = self.cur_annotation
            # elif self.cur_annotation_type == 'pts_nuc':
            #     self.pts_nuc[self.frame_num][0] = self.cur_annotation

            new_df = pd.DataFrame(self.cur_annotation, columns=['x', 'y'])
            new_df['path'] = self.paths[self.frame_num]
            new_df['class'] = c

            idx_to_drop = self.df[((self.df['path'] == self.paths[self.frame_num]) &
                                (self.df['class'] == c))].index
            self.df = self.df.drop(idx_to_drop)
            self.df = self.df.append(new_df, ignore_index=True)

            self.setFrame(self.frame_num)
            self.statusBar().showMessage("annotations updated")

        # append mode
        elif self.mode == 2:
            # annotation_selected = np.empty((0, 2), float)
            # if self.cur_annotation_type == 'pts_neg':
            #     annotation_selected = self.pts_neg[self.frame_num][0]
            # elif self.cur_annotation_type == 'pts_pos':
            #     annotation_selected = self.pts_pos[self.frame_num][0]
            # elif self.cur_annotation_type == 'pts_pos_o':
            #     annotation_selected = self.pts_pos_o[self.frame_num][0]
            # elif self.cur_annotation_type == 'pts_nuc':
            #     annotation_selected = self.pts_nuc[self.frame_num][0]

            # initialize with empty np array if annotation needed to be changed is an empty python list
            # (which cannot be stacked)
            # if len(annotation_selected) == 0:
            #     annotation_selected = np.empty((0, 2), float)

            # annotation_selected = np.vstack((annotation_selected, self.cur_annotation))

            # if self.cur_annotation_type == 'pts_neg':
            #     self.pts_neg[self.frame_num][0] = annotation_selected
            # elif self.cur_annotation_type == 'pts_pos':
            #     self.pts_pos[self.frame_num][0] = annotation_selected
            # elif self.cur_annotation_type == 'pts_pos_o':
            #     self.pts_pos_o[self.frame_num][0] = annotation_selected
            # elif self.cur_annotation_type == 'pts_nuc':
            #     self.pts_nuc[self.frame_num][0] = annotation_selected

            new_df = pd.DataFrame(self.cur_annotation, columns=['x', 'y'])
            new_df['path'] = self.paths[self.frame_num]
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
            # new_mat = {'k': self.k,
            #            'pts_neg': self.pts_neg,
            #            'pts_nuc': self.pts_nuc,
            #            'pts_pos': self.pts_pos,
            #            'pts_pos_o': self.pts_pos_o}
            # scipy.io.savemat(path_to_file, new_mat)

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
            if self.frame_num < self.max_frame_num - 1:
                self.frame_num += 1
                self.cleanUpCurrentAnnotation()
                self.setFrame(self.frame_num)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            if self.mode == 3:
                self.mode = 0
            if self.frame_num > 0:
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
            if user_frame_num < self.max_frame_num and user_frame_num >= 0:
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
    fb = FrameBox()
    fb.show()
    sys.exit(app.exec_())