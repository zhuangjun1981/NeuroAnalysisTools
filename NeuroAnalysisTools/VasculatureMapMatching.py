__author__ = 'junz'
'''
written by Jun Zhuang
11/21/2014
'''

# version 3

import sys, os, random
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ni
from PyQt5 import QtCore, QtGui, QtWidgets
import json

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from core import FileTools as ft
from core import ImageAnalysis as ia
from core import PlottingTools as pt

try:
    import tifffile as tf
except ImportError:
    import skimage.external.tifffile as tf

try: 
    import cv2
    from core.ImageAnalysis import rigid_transform_cv2 as rigid_transform
except ImportError as e: 
    print (e) 
    from core.ImageAnalysis import rigid_transform as rigid_transform


class AppForm(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Vasculature Map Matching')

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        self.spinbox_Xoffset.setValue(0)
        self.spinbox_Yoffset.setValue(0)
        self.spinbox_rotation.setValue(0)

        self.Xoffset=0
        self.Yoffset=0
        self.rotation=0
        self.zoom = 1.
        self.reference_contrast = 1.
        self.matching_contrast = 1.

        self.getAdjustment()

        self.ReferenceVasMap = None
        self.MatchingVasMap = None
        self.MatchingVasMapAfterChange = None
        self.trialDict = None

        # self.connect(self.button_RPath,SIGNAL('clicked()'),self.get_RPath)
        # self.connect(self.button_MPath,SIGNAL('clicked()'),self.get_MPath)
        # self.connect(self.button_draw,SIGNAL('clicked()'), self.on_draw)
        self.button_RPath.clicked.connect(self.get_RPath)
        self.button_MPath.clicked.connect(self.get_MPath)
        self.button_draw.clicked.connect(self.on_draw)

        self.spinbox_Xoffset.valueChanged.connect(self.adjustVasMap)
        self.spinbox_Yoffset.valueChanged.connect(self.adjustVasMap)
        self.spinbox_rotation.valueChanged.connect(self.adjustVasMap)
        self.doubleSpinbox_zoom.valueChanged.connect(self.adjustVasMap)

        self.radiobutton_reference.clicked.connect(self.on_draw)
        self.radiobutton_matching.clicked.connect(self.on_draw)
        self.radiobutton_both.clicked.connect(self.on_draw)

        self.reference_contrast_slider.valueChanged.connect(self._sliding_reference_contrast)
        self.matching_contrast_slider.valueChanged.connect(self._sliding_matching_contrast)

        self.currReferenceFolder = r'C:\JunZhuang\labwork\data\2014-04-30-vasculature-maps\147861'
        self.currMatchingFolder = r'C:\JunZhuang\labwork\data\2014-04-30-vasculature-maps\147861'
        self.currSaveFolder = r'E:\data2\2015-04-17-VasculatureMapMatching'


    def save_alignment_json(self):
        path = unicode(QFileDialog.getSaveFileName(self,
                        'Save file', self.currSaveFolder,
                        '*.json'))

        if path:
            bigDict = {}

            if self.ReferenceVasMap is not None:
                bigDict.update({'ReferencePathList': str(self.textbrowser_RPath.toPlainText()).split(';'),
                                'ReferenceMapHeight': self.ReferenceVasMap.shape[0],
                                'ReferenceMapWidth': self.ReferenceVasMap.shape[1]})
            else:
                bigDict.update({'ReferencePathList': None,
                                'ReferenceMapHeight': None,
                                'ReferenceMapWidth': None})

            if self.MatchingVasMap is not None:
                bigDict.update({'MatchingPathList': str(self.textbrowser_MPath.toPlainText()).split(';'),
                                'MatchingMapHeight': self.MatchingVasMap.shape[0],
                                'MatchingMapWidth': self.MatchingVasMap.shape[1],
                                'Zoom': float(self.zoom),
                                'Rotation': float(self.rotation),
                                'Xoffset': self.Xoffset,
                                'Yoffset': self.Yoffset})
            else:
                bigDict.update({'MatchingPathList': None,
                                'MatchingMapHeight': None,
                                'MatchingMapWidth': None,
                                'Zoom': None,
                                'Rotation': None,
                                'Xoffset': None,
                                'Yoffset': None})

            if path[-5:] == '.json':
                path_surfix = path[:-5]
            else:
                path_surfix = path

            with open(path_surfix+'_VasculatureMapMatchingParameters.json', 'w') as f:
                json.dump(bigDict, f, sort_keys=True, indent=4, separators=(',',': '))

            if self.MatchingVasMapRaw is not None:
                tf.imsave(path_surfix+'_VasculatureMapBeforeMatching.tif',self.MatchingVasMapRaw)
                MatchVasMapAfterChange = rigid_transform(self.MatchingVasMapRaw,
                                                         zoom=self.zoom,
                                                         rotation = self.rotation,
                                                         offset=(self.Xoffset,self.Yoffset),
                                                         outputShape=(self.ReferenceVasMap.shape[0],self.ReferenceVasMap.shape[1]))
                tf.imsave(path_surfix+'_VasculatureMapAfterMatching.tif',MatchVasMapAfterChange)

            self.statusBar().showMessage('Saved to %s' % path, 2000)
            self.currSaveFolder = os.path.split(path)[0]


    def getAdjustment(self):
        self.Xoffset = self.spinbox_Xoffset.value()
        self.Yoffset = self.spinbox_Yoffset.value()
        self.rotation = self.spinbox_rotation.value()
        self.zoom = self.doubleSpinbox_zoom.value()


    def setZero(self):
        self.spinbox_Xoffset.setValue(0.)
        self.spinbox_Yoffset.setValue(0.)
        self.spinbox_rotation.setValue(0.)
        self.doubleSpinbox_zoom.setValue(1.)


    def on_about(self):
        msg = """ match two vasculature maps to get alignment:

         * input file fold and number list to get reference vasculature map
         * input retinotopic mapping trial dictionary to get matching vasculature map
         * Adjust X offset, Y offset and rotation to match two vasculature maps
         * hit menu -> File -> save alignment to save alignment parameter
        """
        QMessageBox.about(self, "About the GUI", msg.strip())


    def get_RPath(self):

        self.axes.clear()
        self.canvas.draw()

        self.button_RPath.setStyleSheet('QPushButton {color: #888888}')
        self.button_RPath.setEnabled(False)

        fnames = QFileDialog.getOpenFileNames(self, 'Choose Retinotopic Mapping Dictionary of TIFF/JCam file(s):',
                self.currReferenceFolder)

        fnames = list(fnames)
        fnames = [str(x) for x in fnames]

        try:
            if len(fnames) == 0: # no file is chosen

                print("no file is chosen! Setting reference map as None...")
                self.textbrowser_RPath.clear()
                self.ReferenceVasMap = None

            elif len(fnames) == 1: # only one file is chosen
                filePath = fnames[0]
                if filePath[-3:] == 'pkl': # mapping dictionary pkl file

                    self.trialDict = ft.loadFile(filePath)
                    self.ReferenceVasMap = pt.merge_normalized_images([self.trialDict['vasculatureMap']])
                    self.textbrowser_RPath.setText(filePath)

                elif filePath[-3:] == 'tif': # tiff file
                    self.ReferenceVasMap = pt.merge_normalized_images([tf.imread(filePath)])
                    self.textbrowser_RPath.setText(filePath)

                else: # Raw binary file
                    fileFolder,fileName = os.path.split(filePath)
                    if 'JCamF' in fileName:
                        currMap, _, _= ft.importRawJCamF(filePath,column=1024,row=1024)
                        self.ReferenceVasMap = pt.merge_normalized_images([currMap[0]])
                        self.textbrowser_RPath.setText(filePath)
                    elif 'JCam' in fileName:
                        currMap, _ = ft.importRawJCam(filePath)
                        self.ReferenceVasMap = pt.merge_normalized_images([currMap[0]])
                        self.textbrowser_RPath.setText(filePath)
                    else:
                        print('Can not read reference map '+filePath)
                        self.textbrowser_RPath.clear()
                        self.ReferenceVasMap = None


            else: # more than one file is chosen

                displayText = ';'.join(fnames)
                mapList = []

                for i, filePath in enumerate(fnames):

                    if filePath[-3:] == 'tif': # tiff file
                        mapList.append(tf.imread(filePath))

                    else: # raw binary file
                        fileFolder,fileName = os.path.split(filePath)
                        if 'JCamF' in fileName:
                            currMap, _, _ = ft.importRawJCamF(filePath,column=1024,row=1024)
                        elif 'JCam' in fileName:
                            currMap, _ = ft.importRawJCam(filePath)
                        else:
                            print('Can not read '+filePath)

                        mapList.append(currMap[0].astype(np.float32))

                if len(mapList) == 0:
                    print("no file can be read! Setting reference map as None...")
                    self.textbrowser_RPath.clear()
                    self.ReferenceVasMap = None
                else:
                    self.ReferenceVasMap = pt.merge_normalized_images(mapList).astype(np.float32)
                    self.textbrowser_RPath.setText(displayText)

        except Exception as e:
            print(e, '\n\n')
            print('Can not load reference Map! Setting it as None...')
            self.textbrowser_RPath.clear()
            self.ReferenceVasMap = None


        self.button_RPath.setEnabled(True)
        self.button_RPath.setStyleSheet('QPushButton {color: #000000}')
        self.setZero()
        self.currReferenceFolder = os.path.split(fnames[0])[0]


    def get_MPath(self):

        self.axes.clear()
        self.canvas.draw()

        self.button_MPath.setStyleSheet('QPushButton {color: #888888}')
        self.button_MPath.setEnabled(False)

        fnames = QFileDialog.getOpenFileNames(self, 'Choose Retinotopic Mapping Dictionary of TIFF/JCam file(s):',
                self.currMatchingFolder)

        fnames = list(fnames)
        fnames = [str(x) for x in fnames]

        try:
            if len(fnames) == 0: # no file is chosen

                print("no file is chosen! Setting matching map as None...")
                self.textbrowser_MPath.clear()
                self.MatchingVasMap = None
                self.MatchingVasMapRaw = None
                self.MatchingVasMapAfterChange = None

            elif len(fnames) == 1: # only one file is chosen
                filePath = fnames[0]
                if filePath[-3:] == 'pkl': # mapping dictionary pkl file

                    self.trialDict = ft.loadFile(filePath)
                    self.MatchingVasMap = pt.merge_normalized_images([self.trialDict['vasculatureMap']])
                    self.MatchingVasMapRaw = self.trialDict['vasculatureMap']
                    self.textbrowser_MPath.setText(filePath)
                    self.MatchingVasMapAfterChange = None

                elif filePath[-3:] == 'tif': # tiff file
                    self.MatchingVasMap = pt.merge_normalized_images([tf.imread(filePath)])
                    self.MatchingVasMapRaw = tf.imread(filePath)
                    self.textbrowser_MPath.setText(filePath)
                    self.MatchingVasMapAfterChange = None

                else: # raw binary file
                    fileFolder,fileName = os.path.split(filePath)
                    if 'JCamF' in fileName:
                        currMap, _, _ = ft.importRawJCamF(filePath,column=1024,row=1024)
                        self.MatchingVasMap = pt.merge_normalized_images([currMap[0]])
                        self.MatchingVasMapRaw = currMap[0]
                        self.textbrowser_MPath.setText(filePath)
                    elif 'JCam' in fileName:
                        currMap, _ = ft.importRawJCam(filePath)
                        self.MatchingVasMap = pt.merge_normalized_images([currMap[0]])
                        self.MatchingVasMapRaw = currMap[0]
                        self.textbrowser_MPath.setText(filePath)
                    else:
                        print('Can not read matching map '+filePath)
                        self.textbrowser_MPath.clear()
                        self.MatchingVasMap = None
                    self.MatchingVasMapAfterChange = None

            else: # more than one file is chosen

                displayText =  ';'.join(fnames)
                mapList = []

                for i, filePath in enumerate(fnames):

                    if filePath[-3:] == 'tif': # tiff file
                        mapList.append(tf.imread(filePath))

                    else: # raw binary file
                        fileFolder,fileName = os.path.split(filePath)
                        if 'JCamF' in fileName:
                            currMap, _ , _= ft.importRawJCamF(filePath,column=1024,row=1024)
                        elif 'JCam' in fileName:
                            currMap, _ = ft.importRawJCam(filePath)
                        else:
                            print('Can not read '+filePath)

                        mapList.append(currMap[0].astype(np.float32))

                if len(mapList) == 0:
                    print("no file can be read! Setting matching map as None...")
                    self.textbrowser_MPath.clear()
                    self.MatchingVasMap = None
                    self.MatchingVasMapRaw = None
                    self.MatchingVasMapAfterChange = None
                else:
                    self.MatchingVasMap = pt.merge_normalized_images(mapList, dtype=np.float32)
                    self.MatchingVasMapRaw = pt.merge_normalized_images(mapList, dtype=np.float32, isFilter=False)
                    self.textbrowser_MPath.setText(displayText)
                    self.MatchingVasMapAfterChange = None

        except Exception as e:
            print(e, '\n\n')
            print('Can not load matching Map! Setting it as None...')
            self.textbrowser_MPath.clear()
            self.MatchingVasMap = None
            self.MatchingVasMapRaw = None
            self.MatchingVasMapAfterChange = None


        self.button_MPath.setEnabled(True)
        self.button_MPath.setStyleSheet('QPushButton {color: #000000}')
        self.setZero()
        self.currMatchingFolder = os.path.split(fnames[0])[0]


    def adjustVasMap(self):

        self.getAdjustment()

        if type(self.MatchingVasMap) != type(None):

            if type(self.ReferenceVasMap) != type(None):
                width = self.ReferenceVasMap.shape[1]
                height = self.ReferenceVasMap.shape[0]
            else:
                width = self.MatchingVasMap.shape[1]
                height = self.MatchingVasMap.shape[0]

            self.MatchingVasMapAfterChange = rigid_transform(self.MatchingVasMap, zoom=self.zoom, rotation = self.rotation, offset=(self.Xoffset, self.Yoffset), outputShape=(height, width))

            self.on_draw()

    def _sliding_reference_contrast(self):
        contrast_map = [1/4.,1/3.,1/2.,1.,2.,3.,4.]
        self.reference_contrast = contrast_map[self.reference_contrast_slider.value()]
        self.on_draw()

    def _sliding_matching_contrast(self):
        contrast_map = [1/4.,1/3.,1/2.,1.,2.,3.,4.]
        self.matching_contrast = contrast_map[self.matching_contrast_slider.value()]
        self.on_draw()

    def on_draw(self):
        """ Redraws the figure
        """

        self.axes.clear()

        if type(self.ReferenceVasMap) != type(None):
            width = self.ReferenceVasMap.shape[1]
            height = self.ReferenceVasMap.shape[0]
        elif type(self.MatchingVasMapAfterChange) != type(None):
            width = self.MatchingVasMapAfterChange.shape[1]
            height = self.MatchingVasMapAfterChange.shape[0]
        elif type(self.MatchingVasMap) != type(None):
            width = self.MatchingVasMap.shape[1]
            height = self.MatchingVasMap.shape[0]
        else:
            width = 1344
            height = 1024

        if (type(self.ReferenceVasMap) != type(None)) and (self.radiobutton_reference.isChecked() or self.radiobutton_both.isChecked()):
            greenChannel = ia.resize_image(self.ReferenceVasMap, (height, width))
            greenChannel = (np.power(ia.array_nor(greenChannel), self.reference_contrast) * 255).astype(np.uint8)
        else:
            greenChannel = np.zeros((height,width)).astype(np.uint8)

        if (self.radiobutton_matching.isChecked() or self.radiobutton_both.isChecked()):
            if type(self.MatchingVasMapAfterChange) != type(None):
                redChannel = ia.resize_image(self.MatchingVasMapAfterChange, (height, width))
                redChannel = (np.power(ia.array_nor(redChannel), self.matching_contrast) * 255).astype(np.uint8)
            elif type(self.MatchingVasMap) != type(None):
                redChannel = ia.resize_image(self.MatchingVasMap, (height, width))
                redChannel = (np.power(ia.array_nor(redChannel), self.matching_contrast) * 255).astype(np.uint8)
            else:
                redChannel = np.zeros((height,width)).astype(np.uint8)
        else:
            redChannel = np.zeros((height,width)).astype(np.uint8)

        blueChannel = np.zeros((height,width)).astype(np.uint8)
        pltImg = cv2.merge((redChannel,greenChannel,blueChannel))

        self.axes.imshow(pltImg)

        self.axes.set_xlim([0,width])
        self.axes.set_ylim([0,height])
        self.axes.invert_yaxis()

        self.canvas.draw()


    def create_main_frame(self):
        self.main_frame = QtWidgets.QWidget()

        self.dpi = 300
        self.fig = Figure(dpi=self.dpi,)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axes = self.fig.add_axes([0, 0, 1, 1])
        self.axes.set_aspect(1)
        self.axes.set_frame_on(False)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self.main_frame)

        # Other GUI controls

        self.reference_contrast_label = QtWidgets.QLabel('reference contrast:')
        self.reference_contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.reference_contrast_slider.setMinimumWidth(200)
        self.reference_contrast_slider.setRange(0,6)
        self.reference_contrast_slider.setValue(3)
        self.matching_contrast_label = QtWidgets.QLabel('matching contrast:')
        self.matching_contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.matching_contrast_slider.setMinimumWidth(200)
        self.matching_contrast_slider.setMinimumWidth(200)
        self.matching_contrast_slider.setRange(0,6)
        self.matching_contrast_slider.setValue(3)

        self.radiobutton_reference = QtWidgets.QRadioButton('Reference')
        self.radiobutton_matching = QtWidgets.QRadioButton('Matching')
        self.radiobutton_both = QtWidgets.QRadioButton('Both')
        self.radiobutton_both.setChecked(True)

        self.label_Xoffset = QtWidgets.QLabel('X offset:')
        self.spinbox_Xoffset = QtWidgets.QSpinBox()
        self.spinbox_Xoffset.setRange(-2000,2000)
        self.spinbox_Xoffset.setSingleStep(10)
        self.spinbox_Xoffset.setMinimumWidth(60)

        self.label_Yoffset = QtWidgets.QLabel('Y offset:')
        self.spinbox_Yoffset = QtWidgets.QSpinBox()
        self.spinbox_Yoffset.setRange(-2000,2000)
        self.spinbox_Yoffset.setSingleStep(10)
        self.spinbox_Yoffset.setMinimumWidth(60)

        self.label_rotation = QtWidgets.QLabel('Rotation:')
        self.spinbox_rotation = QtWidgets.QSpinBox()
        self.spinbox_rotation.setRange(-180,180)
        self.spinbox_rotation.setMinimumWidth(60)

        self.label_zoom = QtWidgets.QLabel('Zoom:    ')
        self.doubleSpinbox_zoom = QtWidgets.QDoubleSpinBox()
        self.doubleSpinbox_zoom.setRange(0.001,64.)
        self.doubleSpinbox_zoom.setValue(1.)
        self.doubleSpinbox_zoom.setMinimumWidth(60)
        self.doubleSpinbox_zoom.setDecimals(3)

        self.label_RPath = QtWidgets.QLabel('Reference Dictionary/Map Path(s):')
        self.textbrowser_RPath = QtWidgets.QTextBrowser()
        self.textbrowser_RPath.setMinimumWidth(200)
        self.button_RPath = QtWidgets.QPushButton('Get Path')


        self.label_MPath = QtWidgets.QLabel('Matching Dictionary/Map Path(s):')
        self.textbrowser_MPath = QtWidgets.QTextBrowser()
        self.textbrowser_MPath.setMinimumWidth(200)
        self.button_MPath = QtWidgets.QPushButton('Get Path')

        self.button_draw = QtWidgets.QPushButton('Draw')
        self.button_draw.setMinimumWidth(100)
        self.button_draw.setFixedHeight(100)

        #
        # Layout with box sizers
        #

        vbox_Reference = QtWidgets.QVBoxLayout()
        for R in [  self.label_RPath, self.textbrowser_RPath, self.button_RPath]:
            vbox_Reference.addWidget(R)
            vbox_Reference.setAlignment(R, QtCore.Qt.AlignLeft)

        vbox_Match = QtWidgets.QVBoxLayout()
        for M in [  self.label_MPath, self.textbrowser_MPath, self.button_MPath]:
            vbox_Match.addWidget(M)
            vbox_Match.setAlignment(M, QtCore.Qt.AlignLeft)


        vbox_checkbox = QtWidgets.QVBoxLayout()
        for P in [  self.radiobutton_reference, self.radiobutton_matching, self.radiobutton_both]:
            vbox_checkbox.addWidget(P)
            vbox_checkbox.setAlignment(P, QtCore.Qt.AlignLeft)

        vbox_contrast = QtWidgets.QVBoxLayout()
        for P in [  self.reference_contrast_label,
                    self.reference_contrast_slider,
                    self.matching_contrast_label,
                    self.matching_contrast_slider]:
            vbox_contrast.addWidget(P)
            vbox_contrast.setAlignment(P, QtCore.Qt.AlignLeft)

        hbox_Zoom = QtWidgets.QHBoxLayout()
        for Z in [self.label_zoom, self.doubleSpinbox_zoom]:
            hbox_Zoom.addWidget(Z)
            hbox_Zoom.setAlignment(Z, QtCore.Qt.AlignVCenter)

        hbox_Xoffset = QtWidgets.QHBoxLayout()
        for X in [  self.label_Xoffset,self.spinbox_Xoffset]:
            hbox_Xoffset.addWidget(X)
            hbox_Xoffset.setAlignment(X, QtCore.Qt.AlignVCenter)

        hbox_Yoffset = QtWidgets.QHBoxLayout()
        for Y in [  self.label_Yoffset,self.spinbox_Yoffset]:
            hbox_Yoffset.addWidget(Y)
            hbox_Yoffset.setAlignment(Y, QtCore.Qt.AlignVCenter)

        hbox_Rotation = QtWidgets.QHBoxLayout()
        for R in [  self.label_rotation,self.spinbox_rotation]:
            hbox_Rotation.addWidget(R)
            hbox_Rotation.setAlignment(R, QtCore.Qt.AlignVCenter)

        vbox_Adjustment = QtWidgets.QVBoxLayout()
        for A in [  hbox_Zoom, hbox_Xoffset,hbox_Yoffset,hbox_Rotation]:
            vbox_Adjustment.addLayout(A)
            vbox_Adjustment.setAlignment(A, QtCore.Qt.AlignLeft)

        vbox_Adjustment2 = QtWidgets.QHBoxLayout()
        for A in [  vbox_Adjustment,self.button_draw]:
            try:
                vbox_Adjustment2.addLayout(A)
            except Exception:
                vbox_Adjustment2.addWidget(A)
            vbox_Adjustment2.setAlignment(A, QtCore.Qt.AlignVCenter)

        vbox_right = QtWidgets.QVBoxLayout()
        for RT in [ vbox_Reference, vbox_Match, vbox_contrast, vbox_checkbox, vbox_Adjustment2]:
            vbox_right.addLayout(RT)
            vbox_right.setAlignment(RT, QtCore.Qt.AlignLeft)
        vbox_right.insertSpacing(1,30)
        vbox_right.insertSpacing(3,30)
        vbox_right.insertSpacing(5,30)

        vbox_plot = QtWidgets.QVBoxLayout()
        for P in [self.canvas, self.mpl_toolbar]:
            vbox_plot.addWidget(P)

        hbox = QtWidgets.QHBoxLayout()
        for L in [  vbox_plot,vbox_right]:
            hbox.addLayout(L)
            hbox.setAlignment(L, QtCore.Qt.AlignVCenter)

        self.main_frame.setLayout(hbox)
        self.setCentralWidget(self.main_frame)


    def create_status_bar(self):
        self.status_text = QtWidgets.QLabel("This is a demo")
        self.statusBar().addWidget(self.status_text, 1)


    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        save_action = self.create_action("&Save alignment",
            shortcut="Ctrl+S", slot=self.save_alignment_json,
            tip="Save the alignment alignment")

        quit_action = self.create_action("&Quit", slot=self.close,
            shortcut="Ctrl+Q", tip="Close the application")

        self.add_actions(self.file_menu,
            (save_action, None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About",
            shortcut='F1', slot=self.on_about,
            tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))


    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)


    def create_action(self, text, slot=None, shortcut=None,icon=None, tip=None, checkable=False,signal="triggered"):
        action = QtWidgets.QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            # self.connect(action, SIGNAL(signal), slot)
            getattr(action, signal).connect(slot)
        if checkable:
            action.setCheckable(True)
        return action


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()

