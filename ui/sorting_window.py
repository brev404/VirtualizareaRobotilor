# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
from io import StringIO
from multiprocessing import process
import ast
from interfaces.robot_interface import Robot

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QImage
from PyQt5.QtWidgets import QLabel, QTextEdit
from PyQt5.QtGui import QPixmap

from interfaces.camera_interface import stream_camera, init_camera
from projects import palletizing_app


class CameraThread(QThread):
    image_data = pyqtSignal(np.ndarray)
    depth_data = pyqtSignal(np.ndarray)

    def __init__(self, pipeline, profile):
        super(CameraThread, self).__init__()
        self.pipeline = pipeline
        self.profile = profile
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            color_data, colorized_depth = stream_camera(pipeline=self.pipeline, new_frame=False, color_depth=True)
            self.image_data.emit(color_data)
            self.depth_data.emit(colorized_depth)

    def stop(self):
        self.running = False

class Ui_MainWindow2(object):
    def setupUi(self, MainWindow):
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1493, 878)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow.setMouseTracking(True)
        MainWindow.setStyleSheet("QMainWindow{\n"
"background-color: rgb(211, 211, 211);\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.MainTab = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MainTab.sizePolicy().hasHeightForWidth())
        self.MainTab.setSizePolicy(sizePolicy)
        self.MainTab.setMinimumSize(QtCore.QSize(300, 400))
        self.MainTab.setMaximumSize(QtCore.QSize(9000, 1000))
        self.MainTab.setSizeIncrement(QtCore.QSize(1, 1))
        self.MainTab.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.MainTab.setMouseTracking(True)
        self.MainTab.setStyleSheet("QTabWidget::pane {\n"
"}\n"
"QTabWidget::tab-bar {\n"
"\n"
"}\n"
"QTabBar::tab {\n"
"    color: white;\n"
"    border: 5px solid #C4C4C3;\n"
"    border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"    min-width: 8px;\n"
"    padding: 2px;\n"
"}\n"
"\n"
"QTabBar::tab:selected, QTabBar::tab:hover {\n"
"    background: darkgrey;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    border-color: #00aeff;\n"
"    border-bottom-color: #3c4957; /* same as pane color */\n"
"}\n"
"\n"
"QTabBar::tab:!selected {\n"
"    margin-top: 2px; /* make non-selected tabs look smaller */\n"
"}\n"
"QWidget #programEdit_tab,#flowView_tab,#palletView_tab{\n"
"background-color: #3c4957;\n"
"}")
        self.MainTab.setMovable(True)
        self.MainTab.setObjectName("MainTab")
        self.palletView_tab = QtWidgets.QWidget()
        self.palletView_tab.setMouseTracking(True)
        self.palletView_tab.setObjectName("palletView_tab")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.palletView_tab)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        # self.graphicsView = PlotWidget(self.palletView_tab)
        # self.graphicsView.setObjectName("graphicsView")
        # self.gridLayout_5.addWidget(self.graphicsView, 0, 2, 1, 1)
        self.imageLabel3 = QLabel(self.palletView_tab)
        self.imageLabel3.setObjectName("imageLabel")
        self.imageLabel3.setStyleSheet("background-color: white;")
        self.gridLayout_5.addWidget(self.imageLabel3, 0, 2, 1, 1)
        self.frame_6 = QtWidgets.QFrame(self.palletView_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(100, 0))
        self.frame_6.setMaximumSize(QtCore.QSize(200, 16777215))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.recognizePalletButton = QtWidgets.QPushButton(self.frame_6)
        self.recognizePalletButton.setObjectName("recognizePalletButton")
        self.gridLayout_13.addWidget(self.recognizePalletButton, 2, 0, 1, 1)
        self.widget_3 = QtWidgets.QWidget(self.frame_6)
        self.widget_3.setMaximumSize(QtCore.QSize(180, 299))
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_13 = QtWidgets.QLabel(self.widget_3)
        self.label_13.setStyleSheet("QLabel{\n"
                "color:white;\n"
                "font-size:18px;\n"
                "}")
        self.label_13.setObjectName("label_13")
        self.gridLayout_6.addWidget(self.label_13, 0, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.nr_classes_Input = QtWidgets.QLineEdit(self.widget_3)
        self.nr_classes_Input.setMaximumSize(QtCore.QSize(100, 16777215))
        self.nr_classes_Input.setObjectName("nr_classes_Input")
        self.nr_classes_Input.setReadOnly(True)
        self.nr_classes_Input.setFont(font)
        # self.onlyInt = QIntValidator()
        # self.nr_classes_Input.setValidator(self.onlyInt)

        self.horizontalLayout_8.addWidget(self.nr_classes_Input, 0, QtCore.Qt.AlignLeft)
        self.gridLayout_6.addLayout(self.horizontalLayout_8, 1, 0, 1, 1)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.current_object = QtWidgets.QLineEdit(self.widget_3)
        self.current_object.setMaximumSize(QtCore.QSize(100, 16777215))
        self.current_object.setObjectName("current_object")

        # self.onlyInt = QIntValidator()
        self.current_object.setReadOnly(True)
        self.current_object.setFont(font)
        self.horizontalLayout_11.addWidget(self.current_object, 0, QtCore.Qt.AlignLeft)
        self.gridLayout_6.addLayout(self.horizontalLayout_11, 3, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.widget_3)
        self.label_15.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_15.setObjectName("label_15")
        self.gridLayout_6.addWidget(self.label_15, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label_12 = QtWidgets.QLabel(self.widget_3)
        self.label_12.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_12.setObjectName("label_12")
        self.verticalLayout_12.addWidget(self.label_12)
        self.lineEdit = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit.setMaximumSize(QtCore.QSize(90, 30))
        self.lineEdit.setObjectName("lineEdit")

        self.lineEdit.setReadOnly(True)
        self.lineEdit.setFont(font)

        self.verticalLayout_12.addWidget(self.lineEdit)
        self.horizontalLayout.addLayout(self.verticalLayout_12)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_19 = QtWidgets.QLabel(self.widget_3)
        self.label_19.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_19.setObjectName("label_19")
        self.verticalLayout_14.addWidget(self.label_19)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_3.setMaximumSize(QtCore.QSize(90, 30))
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.lineEdit_3.setReadOnly(True)
        self.lineEdit_3.setFont(font)
        self.verticalLayout_14.addWidget(self.lineEdit_3)
        self.horizontalLayout.addLayout(self.verticalLayout_14)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.label_20 = QtWidgets.QLabel(self.widget_3)
        self.label_20.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_20.setObjectName("label_20")
        self.verticalLayout_15.addWidget(self.label_20)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_4.setMaximumSize(QtCore.QSize(90, 30))
        self.lineEdit_4.setObjectName("lineEdit_4")

        self.lineEdit_4.setReadOnly(True)
        self.lineEdit_4.setFont(font)
        self.verticalLayout_15.addWidget(self.lineEdit_4)
        self.horizontalLayout.addLayout(self.verticalLayout_15)
        self.gridLayout_6.addLayout(self.horizontalLayout, 4, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.label_24 = QtWidgets.QLabel(self.widget_3)
        self.label_24.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_24.setObjectName("label_24")
        self.verticalLayout_19.addWidget(self.label_24)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_8.setMaximumSize(QtCore.QSize(90, 30))
        self.lineEdit_8.setObjectName("lineEdit_8")

        self.lineEdit_8.setReadOnly(True)
        self.lineEdit_8.setFont(font)
        self.verticalLayout_19.addWidget(self.lineEdit_8)
        self.horizontalLayout_5.addLayout(self.verticalLayout_19)
        self.verticalLayout_20 = QtWidgets.QVBoxLayout()
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.label_25 = QtWidgets.QLabel(self.widget_3)
        self.label_25.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_25.setObjectName("label_25")
        self.verticalLayout_20.addWidget(self.label_25)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_9.setMaximumSize(QtCore.QSize(90, 30))
        self.lineEdit_9.setObjectName("lineEdit_9")

        self.lineEdit_9.setReadOnly(True)
        self.lineEdit_9.setFont(font)
        self.verticalLayout_20.addWidget(self.lineEdit_9)
        self.horizontalLayout_5.addLayout(self.verticalLayout_20)
        self.verticalLayout_21 = QtWidgets.QVBoxLayout()
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.label_26 = QtWidgets.QLabel(self.widget_3)
        self.label_26.setStyleSheet("QLabel{\n"
"color:white;\n"
"font-size:18px;\n"
"}")
        self.label_26.setObjectName("label_26")
        self.verticalLayout_21.addWidget(self.label_26)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_10.setMaximumSize(QtCore.QSize(90, 30))
        self.lineEdit_10.setObjectName("lineEdit_10")

        self.lineEdit_10.setReadOnly(True)
        self.lineEdit_10.setFont(font)
        self.verticalLayout_21.addWidget(self.lineEdit_10)
        self.horizontalLayout_5.addLayout(self.verticalLayout_21)
        self.gridLayout_6.addLayout(self.horizontalLayout_5, 5, 0, 1, 1)
        self.gridLayout_13.addWidget(self.widget_3, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_13.addItem(spacerItem, 3, 0, 1, 1)
        self.gridLayout_5.addWidget(self.frame_6, 0, 1, 1, 1)
        self.horizontalLayout_4.addLayout(self.gridLayout_5)
        self.MainTab.addTab(self.palletView_tab, "")

        self.stream = QtWidgets.QWidget()
        self.stream.setObjectName("stream")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.stream)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.frame_3 = QtWidgets.QFrame(self.stream)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_11.setObjectName("gridLayout_11")
        # self.graphicsView_2 = QtWidgets.QGraphicsView(self.frame_3)
        # self.graphicsView_2.setMaximumSize(QtCore.QSize(640, 480))
        # self.graphicsView_2.setObjectName("graphicsView_2")
        # self.graphicsView_2.setStyleSheet("background-color: white;")

        self.imageLabel = QtWidgets.QLabel(self.stream)
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setMaximumSize(QtCore.QSize(640, 480))
        self.imageLabel.setMinimumSize(QtCore.QSize(640, 480))
        self.imageLabel.setStyleSheet("background-color: white;")

        self.imageLabel2 = QtWidgets.QLabel(self.stream)
        self.imageLabel2.setObjectName("imageLabel2")
        self.imageLabel2.setMaximumSize(QtCore.QSize(640, 480))
        self.imageLabel2.setMinimumSize(QtCore.QSize(640, 480))
        self.imageLabel2.setStyleSheet("background-color: white;")

        self.gridLayout_12.addWidget(self.imageLabel, 0, 0)
        self.gridLayout_12.addWidget(self.imageLabel2, 0, 1)

        # spacer_orizontal_2 = QtWidgets.QSpacerItem(20, 20,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum )
        # self.gridLayout_12.addItem(spacer_orizontal_2)

        # self.gridLayout_11.addWidget(self.graphicsView_2, 0, 0, 1, 1)
        # self.gridLayout_12.addWidget(self.frame_3, 0, 0, 1, 1)
        self.frame = QtWidgets.QFrame(self.stream)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.pushButton_6 = QtWidgets.QPushButton(self.frame)
        self.pushButton_6.setStyleSheet("QPushButton {\n"
                                        "  display: inline-block;\n"
                                        "  padding: 10px 20px;\n"
                                        "  border-radius: 50px; /* Adjust this value for the desired level of roundness */\n"
                                        "  background-color: #187042; /* Green colorr */\n"
                                        "  color: #ffffff; /* Text color */\n"
                                        "  border: none;\n"
                                        "  cursor: pointer;\n"
                                        "  font-size: 16px;\n"
                                        "  font-weight: bold;\n"
                                        "  text-align: center;\n"
                                        "  text-decoration: none;\n"
                                        "  transition: background-color 0.3s ease, transform 0.2s ease;\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:hover {\n"
                                        "  background-color: #187f42; /* Darker shade on hover */\n"
                                        "  transform: scale(1.05); /* Slight scale up on hover */\n"
                                        "}")
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_5.addWidget(self.pushButton_6)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame)
        self.pushButton_5.setMaximumSize(QtCore.QSize(1677215, 16777215))
        self.pushButton_5.setStyleSheet("QPushButton {\n"
                                        "  display: inline-block;\n"
                                        "  padding: 10px 20px;\n"
                                        "  border-radius: 30px; /* Adjust this value for the desired level of roundness */\n"
                                        "  background-color: #fa2042; /* Red color */\n"
                                        "  color: #ffffff; /* Text color */\n"
                                        "  border: none;\n"
                                        "  cursor: pointer;\n"
                                        "  font-size: 16px;\n"
                                        "  font-weight: bold;\n"
                                        "  text-align: center;\n"
                                        "  text-decoration: none;\n"
                                        "  transition: background-color 0.3s ease, transform 0.2s ease;\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:hover {\n"
                                        "  background-color: #fa2042; /* Darker shade on hover */\n"
                                        "  transform: scale(1.05); /* Slight scale up on hover */\n"
                                        "}")
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_5.addWidget(self.pushButton_5)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.gridLayout_12.addWidget(self.frame, 1, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_12)
        self.stream.setStyleSheet("background-color:#3c4957")
        self.MainTab.addTab(self.stream, "")

        self.gridLayout_3.addWidget(self.MainTab, 0, 8, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setMinimumSize(QtCore.QSize(300, 700))
        self.widget.setMaximumSize(QtCore.QSize(300, 700))
        self.widget.setObjectName("widget")
        self.robotSettings = QtWidgets.QVBoxLayout(self.widget)
        self.robotSettings.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.robotSettings.setObjectName("robotSettings")
        self.robotIP = QtWidgets.QLabel(self.widget)
        self.robotIP.setMinimumSize(QtCore.QSize(0, 20))
        self.robotIP.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.robotIP.setFont(font)
        self.robotIP.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:18px;\n"
"font: \"Verdana\";\n"
"}")
        self.robotIP.setObjectName("robotIP")
        self.robotSettings.addWidget(self.robotIP)
        self.robotIP_input = QtWidgets.QLineEdit(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.robotIP_input.sizePolicy().hasHeightForWidth())
        self.robotIP_input.setSizePolicy(sizePolicy)
        self.robotIP_input.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.robotIP_input.setFont(font)
        self.robotIP_input.setText("")
        self.robotIP_input.setObjectName("robotIP_input")

        self.robotIP_input.setInputMask("000.000.000.000;0")
        regexp = QtCore.QRegExp('^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){0,3}$')
        validator = QtGui.QRegExpValidator(regexp)
        self.robotIP_input.setValidator(validator)

        self.robotSettings.addWidget(self.robotIP_input)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMaximumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:18px;\n"
"font: \"Verdana\";\n"
"}")
        self.label.setObjectName("label")
        self.robotSettings.addWidget(self.label)
        self.widget1 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget1.sizePolicy().hasHeightForWidth())
        self.widget1.setSizePolicy(sizePolicy)
        self.widget1.setMaximumSize(QtCore.QSize(200, 70))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.widget1)
        self.label_5.setMaximumSize(QtCore.QSize(90, 20))
        self.label_5.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.payloadZInput = QtWidgets.QLineEdit(self.widget1)
        self.payloadZInput.setMaximumSize(QtCore.QSize(90, 30))
        self.payloadZInput.setObjectName("payloadZInput")

        self.payloadZInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_2.addWidget(self.payloadZInput)
        self.gridLayout_9.addLayout(self.verticalLayout_2, 0, 3, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget1)
        self.label_3.setMaximumSize(QtCore.QSize(90, 20))
        self.label_3.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.payloadXInput = QtWidgets.QLineEdit(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.payloadXInput.sizePolicy().hasHeightForWidth())
        self.payloadXInput.setSizePolicy(sizePolicy)
        self.payloadXInput.setMaximumSize(QtCore.QSize(90, 30))
        self.payloadXInput.setObjectName("payloadXInput")

        self.payloadXInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_3.addWidget(self.payloadXInput)
        self.gridLayout_9.addLayout(self.verticalLayout_3, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMaximumSize(QtCore.QSize(90, 20))
        self.label_2.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.payloadWInput = QtWidgets.QLineEdit(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.payloadWInput.sizePolicy().hasHeightForWidth())
        self.payloadWInput.setSizePolicy(sizePolicy)
        self.payloadWInput.setMaximumSize(QtCore.QSize(90, 30))
        self.payloadWInput.setObjectName("payloadWInput")

        self.payloadWInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout.addWidget(self.payloadWInput)
        self.gridLayout_9.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.widget1)
        self.label_4.setMaximumSize(QtCore.QSize(90, 20))
        self.label_4.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.payloadYInput = QtWidgets.QLineEdit(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.payloadYInput.sizePolicy().hasHeightForWidth())
        self.payloadYInput.setSizePolicy(sizePolicy)
        self.payloadYInput.setMaximumSize(QtCore.QSize(90, 30))
        self.payloadYInput.setObjectName("payloadYInput")

        self.payloadYInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_4.addWidget(self.payloadYInput)
        self.gridLayout_9.addLayout(self.verticalLayout_4, 0, 2, 1, 1)
        self.horizontalLayout_6.addLayout(self.gridLayout_9)
        self.robotSettings.addWidget(self.widget1)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setMaximumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}\n"
"")
        self.label_6.setObjectName("label_6")
        self.robotSettings.addWidget(self.label_6)
        self.widget_2 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMaximumSize(QtCore.QSize(200, 70))
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label_9 = QtWidgets.QLabel(self.widget_2)
        self.label_9.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_9.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_9.setObjectName("label_9")
        self.verticalLayout_11.addWidget(self.label_9)
        self.tcpZInput = QtWidgets.QLineEdit(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tcpZInput.sizePolicy().hasHeightForWidth())
        self.tcpZInput.setSizePolicy(sizePolicy)
        self.tcpZInput.setMaximumSize(QtCore.QSize(90, 30))
        self.tcpZInput.setObjectName("tcpZInput")

        self.tcpZInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_11.addWidget(self.tcpZInput)
        self.gridLayout_10.addLayout(self.verticalLayout_11, 0, 2, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.widget_2)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_7.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}\n"
"")
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7)
        self.tcpXInput = QtWidgets.QLineEdit(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tcpXInput.sizePolicy().hasHeightForWidth())
        self.tcpXInput.setSizePolicy(sizePolicy)
        self.tcpXInput.setMaximumSize(QtCore.QSize(90, 30))
        self.tcpXInput.setObjectName("tcpXInput")

        self.tcpXInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_6.addWidget(self.tcpXInput)
        self.horizontalLayout_7.addLayout(self.verticalLayout_6)
        self.gridLayout_10.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_8 = QtWidgets.QLabel(self.widget_2)
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_8.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_10.addWidget(self.label_8)
        self.tcpYInput = QtWidgets.QLineEdit(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tcpYInput.sizePolicy().hasHeightForWidth())
        self.tcpYInput.setSizePolicy(sizePolicy)
        self.tcpYInput.setMaximumSize(QtCore.QSize(90, 30))
        self.tcpYInput.setObjectName("tcpYInput")

        self.tcpYInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_10.addWidget(self.tcpYInput)
        self.gridLayout_10.addLayout(self.verticalLayout_10, 0, 1, 1, 1)
        self.robotSettings.addWidget(self.widget_2)
        self.tcp_frame2 = QtWidgets.QFrame(self.widget)
        self.tcp_frame2.setMaximumSize(QtCore.QSize(200, 70))
        self.tcp_frame2.setObjectName("tcp_frame2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tcp_frame2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_10 = QtWidgets.QLabel(self.tcp_frame2)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_10.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}")
        self.label_10.setObjectName("label_10")
        self.verticalLayout_9.addWidget(self.label_10)
        self.tcpRxInput_2 = QtWidgets.QLineEdit(self.tcp_frame2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tcpRxInput_2.sizePolicy().hasHeightForWidth())
        self.tcpRxInput_2.setSizePolicy(sizePolicy)
        self.tcpRxInput_2.setMaximumSize(QtCore.QSize(90, 30))
        self.tcpRxInput_2.setObjectName("tcpRxInput_2")

        self.tcpRxInput_2.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_9.addWidget(self.tcpRxInput_2)
        self.gridLayout_2.addLayout(self.verticalLayout_9, 0, 0, 1, 1)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_11 = QtWidgets.QLabel(self.tcp_frame2)
        self.label_11.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_11.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}\n"
"")
        self.label_11.setObjectName("label_11")
        self.verticalLayout_8.addWidget(self.label_11)
        self.tcpRzInput = QtWidgets.QLineEdit(self.tcp_frame2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tcpRzInput.sizePolicy().hasHeightForWidth())
        self.tcpRzInput.setSizePolicy(sizePolicy)
        self.tcpRzInput.setMaximumSize(QtCore.QSize(90, 30))
        self.tcpRzInput.setObjectName("tcpRzInput")

        self.tcpRzInput.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_8.addWidget(self.tcpRzInput)
        self.gridLayout_2.addLayout(self.verticalLayout_8, 0, 2, 1, 1)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_16 = QtWidgets.QLabel(self.tcp_frame2)
        self.label_16.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_16.setStyleSheet("QLabel{\n"
"color:black;\n"
"font-size:20px;\n"
"font: \"Verdana\"\n"
"}\n"
"")
        self.label_16.setObjectName("label_16")
        self.verticalLayout_7.addWidget(self.label_16)
        self.tcpRzInput_2 = QtWidgets.QLineEdit(self.tcp_frame2)
        self.tcpRzInput_2.setMaximumSize(QtCore.QSize(90, 30))
        self.tcpRzInput_2.setObjectName("tcpRzInput_2")

        self.tcpRzInput_2.setValidator(QDoubleValidator(-999.999, 999.999, 3))

        self.verticalLayout_7.addWidget(self.tcpRzInput_2)
        self.gridLayout_2.addLayout(self.verticalLayout_7, 0, 1, 1, 1)
        self.robotSettings.addWidget(self.tcp_frame2)
        self.gridLayout_3.addWidget(self.widget, 0, 3, 1, 1, QtCore.Qt.AlignTop)
        self.InfoTab = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.InfoTab.sizePolicy().hasHeightForWidth())
        self.InfoTab.setSizePolicy(sizePolicy)
        self.InfoTab.setMinimumSize(QtCore.QSize(200, 400))
        self.InfoTab.setMaximumSize(QtCore.QSize(200, 700))
        self.InfoTab.setSizeIncrement(QtCore.QSize(1, 1))
        self.InfoTab.setMouseTracking(True)
        self.InfoTab.setStyleSheet("QTabWidget::pane {\n"
"}\n"
"QTabWidget::tab-bar {\n"
"\n"
"}\n"
"QTabBar::tab {\n"
"    color: white;\n"
"    border: 5px solid #C4C4C3;\n"
"    border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"    min-width: 8px;\n"
"    padding: 2px;\n"
"}\n"
"QTabBar::tab:selected, QTabBar::tab:hover {\n"
"    background: darkgrey;\n"
"}\n"
"QTabBar::tab:selected {\n"
"    border-color: #00aeff;\n"
"    border-bottom-color: #C2C7CB; /* same as pane color */\n"
"}\n"
"QTabBar::tab:!selected {\n"
"    margin-top: 2px; /* make non-selected tabs look smaller */\n"
"}\n"
"QWidget #joint_tab_2,#cart_tab_2,#monitor_2\n"
"{\n"
"background-color: #3c4957;\n"
"}")
        self.InfoTab.setMovable(True)
        self.InfoTab.setObjectName("InfoTab")
        self.joint_tab_2 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.joint_tab_2.sizePolicy().hasHeightForWidth())
        self.joint_tab_2.setSizePolicy(sizePolicy)
        self.joint_tab_2.setMinimumSize(QtCore.QSize(50, 50))
        self.joint_tab_2.setMaximumSize(QtCore.QSize(1500, 1000))
        self.joint_tab_2.setSizeIncrement(QtCore.QSize(1, 1))
        self.joint_tab_2.setMouseTracking(True)
        self.joint_tab_2.setObjectName("joint_tab_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.joint_tab_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.j2 = QtWidgets.QTextEdit(self.joint_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.j2.sizePolicy().hasHeightForWidth())
        self.j2.setSizePolicy(sizePolicy)
        self.j2.setMinimumSize(QtCore.QSize(50, 50))
        self.j2.setMaximumSize(QtCore.QSize(100, 50))
        self.j2.setSizeIncrement(QtCore.QSize(1, 1))
        self.j2.setStyleSheet("QTextEdit{\n"
"background: transparent;\n"
"border: none;\n"
"color: white;\n"
"}")
        self.j2.setObjectName("j2")

        self.j2.setReadOnly(True)

        self.gridLayout_4.addWidget(self.j2, 1, 0, 1, 1)
        self.j3 = QtWidgets.QTextEdit(self.joint_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.j3.sizePolicy().hasHeightForWidth())
        self.j3.setSizePolicy(sizePolicy)
        self.j3.setMinimumSize(QtCore.QSize(50, 50))
        self.j3.setMaximumSize(QtCore.QSize(100, 50))
        self.j3.setSizeIncrement(QtCore.QSize(1, 1))
        self.j3.setStyleSheet("QTextEdit{\n"
"background: transparent;\n"
"border: none;\n"
"color: white;\n"
"}")
        self.j3.setObjectName("j3")

        self.j3.setReadOnly(True)

        self.gridLayout_4.addWidget(self.j3, 2, 0, 1, 1)
        self.j5 = QtWidgets.QTextEdit(self.joint_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.j5.sizePolicy().hasHeightForWidth())
        self.j5.setSizePolicy(sizePolicy)
        self.j5.setMinimumSize(QtCore.QSize(50, 50))
        self.j5.setMaximumSize(QtCore.QSize(100, 50))
        self.j5.setSizeIncrement(QtCore.QSize(1, 1))
        self.j5.setStyleSheet("QTextEdit{\n"
"background: transparent;\n"
"border: none;\n"
"color: white;\n"
"}")
        self.j5.setObjectName("j5")

        self.j5.setReadOnly(True)

        self.gridLayout_4.addWidget(self.j5, 1, 1, 1, 1)
        self.j4 = QtWidgets.QTextEdit(self.joint_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.j4.sizePolicy().hasHeightForWidth())
        self.j4.setSizePolicy(sizePolicy)
        self.j4.setMinimumSize(QtCore.QSize(50, 50))
        self.j4.setMaximumSize(QtCore.QSize(100, 50))
        self.j4.setSizeIncrement(QtCore.QSize(1, 1))
        self.j4.setStyleSheet("QTextEdit{\n"
"background: transparent;\n"
"border: none;\n"
"color: white;\n"
"}")
        self.j4.setObjectName("j4")

        self.j4.setReadOnly(True)

        self.gridLayout_4.addWidget(self.j4, 0, 1, 1, 1)
        self.j6 = QtWidgets.QTextEdit(self.joint_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.j6.sizePolicy().hasHeightForWidth())
        self.j6.setSizePolicy(sizePolicy)
        self.j6.setMinimumSize(QtCore.QSize(50, 50))
        self.j6.setMaximumSize(QtCore.QSize(100, 50))
        self.j6.setSizeIncrement(QtCore.QSize(1, 1))
        self.j6.setStyleSheet("QTextEdit{\n"
"background: transparent;\n"
"border: none;\n"
"color: white;\n"
"}")
        self.j6.setObjectName("j6")

        self.j6.setReadOnly(True)

        self.gridLayout_4.addWidget(self.j6, 2, 1, 1, 1)
        self.j1 = QtWidgets.QTextEdit(self.joint_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.j1.sizePolicy().hasHeightForWidth())
        self.j1.setSizePolicy(sizePolicy)
        self.j1.setMinimumSize(QtCore.QSize(50, 50))
        self.j1.setMaximumSize(QtCore.QSize(100, 50))
        self.j1.setSizeIncrement(QtCore.QSize(1, 1))
        self.j1.setStyleSheet("QTextEdit{\n"
"background: transparent;\n"
"border: none;\n"
"color: white;\n"
"}")
        self.j1.setObjectName("j1")

        self.j1.setReadOnly(True)

        self.gridLayout_4.addWidget(self.j1, 0, 0, 1, 1)
        self.InfoTab.addTab(self.joint_tab_2, "")
        self.cart_tab_2 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cart_tab_2.sizePolicy().hasHeightForWidth())
        self.cart_tab_2.setSizePolicy(sizePolicy)
        self.cart_tab_2.setMaximumSize(QtCore.QSize(500, 900))
        self.cart_tab_2.setSizeIncrement(QtCore.QSize(1, 1))
        self.cart_tab_2.setMouseTracking(True)
        self.cart_tab_2.setObjectName("cart_tab_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.cart_tab_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.xC = QtWidgets.QTextEdit(self.cart_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.xC.sizePolicy().hasHeightForWidth())
        self.xC.setSizePolicy(sizePolicy)
        self.xC.setMinimumSize(QtCore.QSize(50, 50))
        self.xC.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.xC.setFont(font)
        self.xC.setObjectName("xC")

        self.xC.setReadOnly(True)

        self.gridLayout_7.addWidget(self.xC, 0, 0, 1, 1)
        self.rxC = QtWidgets.QTextEdit(self.cart_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.rxC.sizePolicy().hasHeightForWidth())
        self.rxC.setSizePolicy(sizePolicy)
        self.rxC.setMinimumSize(QtCore.QSize(50, 50))
        self.rxC.setMaximumSize(QtCore.QSize(100, 50))
        self.rxC.setObjectName("rxC")

        self.rxC.setReadOnly(True)

        self.gridLayout_7.addWidget(self.rxC, 0, 1, 1, 1)
        self.yC = QtWidgets.QTextEdit(self.cart_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.yC.sizePolicy().hasHeightForWidth())
        self.yC.setSizePolicy(sizePolicy)
        self.yC.setMinimumSize(QtCore.QSize(50, 50))
        self.yC.setMaximumSize(QtCore.QSize(100, 50))
        self.yC.setObjectName("yC")

        self.yC.setReadOnly(True)

        self.gridLayout_7.addWidget(self.yC, 1, 0, 1, 1)
        self.ryC = QtWidgets.QTextEdit(self.cart_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.ryC.sizePolicy().hasHeightForWidth())
        self.ryC.setSizePolicy(sizePolicy)
        self.ryC.setMinimumSize(QtCore.QSize(50, 50))
        self.ryC.setMaximumSize(QtCore.QSize(100, 50))
        self.ryC.setObjectName("ryC")

        self.ryC.setReadOnly(True)

        self.gridLayout_7.addWidget(self.ryC, 1, 1, 1, 1)
        self.zC = QtWidgets.QTextEdit(self.cart_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.zC.sizePolicy().hasHeightForWidth())
        self.zC.setSizePolicy(sizePolicy)
        self.zC.setMinimumSize(QtCore.QSize(50, 50))
        self.zC.setMaximumSize(QtCore.QSize(100, 50))
        self.zC.setObjectName("zC")

        self.zC.setReadOnly(True)

        self.gridLayout_7.addWidget(self.zC, 2, 0, 1, 1)
        self.rzC = QtWidgets.QTextEdit(self.cart_tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.rzC.sizePolicy().hasHeightForWidth())
        self.rzC.setSizePolicy(sizePolicy)
        self.rzC.setMinimumSize(QtCore.QSize(50, 50))
        self.rzC.setMaximumSize(QtCore.QSize(100, 50))
        self.rzC.setObjectName("rzC")

        self.rzC.setReadOnly(True)

        self.gridLayout_7.addWidget(self.rzC, 2, 1, 1, 1)
        self.InfoTab.addTab(self.cart_tab_2, "")
        self.gridLayout_3.addWidget(self.InfoTab, 0, 9, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_3)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.console = QtWidgets.QTextEdit(self.centralwidget)
        self.console.setMinimumSize(QtCore.QSize(400, 10))
        self.console.setMaximumSize(QtCore.QSize(20000, 200))
        self.console.setObjectName("console")
        self.gridLayout.addWidget(self.console, 0, 0, 1, 1)
        self.console.setFont(font)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.playButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.playButton.sizePolicy().hasHeightForWidth())
        self.playButton.setSizePolicy(sizePolicy)
        self.playButton.setMinimumSize(QtCore.QSize(20, 10))
        self.playButton.setMaximumSize(QtCore.QSize(100, 50))
        self.playButton.setSizeIncrement(QtCore.QSize(2, 1))
        self.playButton.setStyleSheet("QPushButton {\n"
"  display: inline-block;\n"
"  padding: 10px 20px;\n"
"  border-radius: 50px; /* Adjust this value for the desired level of roundness */\n"
"  background-color: #187042; /* Green colorr */\n"
"  color: #ffffff; /* Text color */\n"
"  border: none;\n"
"  cursor: pointer;\n"
"  font-size: 16px;\n"
"  font-weight: bold;\n"
"  text-align: center;\n"
"  text-decoration: none;\n"
"  transition: background-color 0.3s ease, transform 0.2s ease;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"  background-color: #187f42; /* Darker shade on hover */\n"
"  transform: scale(1.05); /* Slight scale up on hover */\n"
"}")
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_2.addWidget(self.playButton)
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stopButton.sizePolicy().hasHeightForWidth())
        self.stopButton.setSizePolicy(sizePolicy)
        self.stopButton.setMinimumSize(QtCore.QSize(20, 10))
        self.stopButton.setMaximumSize(QtCore.QSize(100, 50))
        self.stopButton.setSizeIncrement(QtCore.QSize(2, 1))
        self.stopButton.setStyleSheet("QPushButton {\n"
"  display: inline-block;\n"
"  padding: 10px 20px;\n"
"  border-radius: 30px; /* Adjust this value for the desired level of roundness */\n"
"  background-color: #fa2042; /* Red color */\n"
"  color: #ffffff; /* Text color */\n"
"  border: none;\n"
"  cursor: pointer;\n"
"  font-size: 16px;\n"
"  font-weight: bold;\n"
"  text-align: center;\n"
"  text-decoration: none;\n"
"  transition: background-color 0.3s ease, transform 0.2s ease;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"  background-color: #fa2042; /* Darker shade on hover */\n"
"  transform: scale(1.05); /* Slight scale up on hover */\n"
"}")
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_2.addWidget(self.stopButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.homeButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.homeButton.sizePolicy().hasHeightForWidth())
        self.homeButton.setSizePolicy(sizePolicy)
        self.homeButton.setMinimumSize(QtCore.QSize(20, 10))
        self.homeButton.setMaximumSize(QtCore.QSize(100, 50))
        self.homeButton.setSizeIncrement(QtCore.QSize(2, 1))
        self.homeButton.setStyleSheet("QPushButton {\n"
                                      "  display: inline-block;\n"
                                      "  padding: 10px 20px;\n"
                                      "  border-radius: 30px; /* Adjust this value for the desired level of roundness */\n"
                                      "  background-color: #0000FF; /* Orange */\n"
                                      "  color: #ffffff; /* Text color */\n"
                                      "  border: none;\n"
                                      "  cursor: pointer;\n"
                                      "  font-size: 16px;\n"
                                      "  font-weight: bold;\n"
                                      "  text-align: center;\n"
                                      "  text-decoration: none;\n"
                                      "  transition: background-color 0.3s ease, transform 0.2s ease;\n"
                                      "}\n"
                                      "\n"
                                      "QPushButton:hover {\n"
                                      "  background-color:#0000FF; /* Darker shade on hover */\n"
                                      "  transform: scale(1.05); /* Slight scale up on hover */\n"
                                      "}")
        self.homeButton.setObjectName("homeButton")
        self.horizontalLayout_2.addWidget(self.homeButton)

        self.verticalLayout_5.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1493, 31))
        self.menubar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.menubar.setAutoFillBackground(False)
        self.menubar.setStyleSheet("QMenuBar{\n"
"color: white;\n"
"background-color: grey;\n"
"padding-top: 5 px;\n"
"font-size: 10 px;\n"
"padding-left: 20px;\n"
"border-size: 1px;\n"
"border-color:white;\n"
"}")
        self.menubar.setDefaultUp(False)
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setStyleSheet("QMenu{\n"
"color: white;\n"
"background-color: grey;\n"
"padding-top: 3 px;\n"
"border:2px white;\n"
"}")
        self.menuFile.setObjectName("menuFile")
        self.actionNew_2 = QtWidgets.QMenu(self.menuFile)
        self.actionNew_2.setObjectName("actionNew_2")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setStyleSheet("QMenu{\n"
"color: white;\n"
"background-color: grey;\n"
"}")
        self.menuHelp.setObjectName("menuHelp")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setStyleSheet("QMenu{\n"
"color: white;\n"
"background-color: grey;\n"
"}")
        self.menuSettings.setObjectName("menuSettings")
        self.menuEditor = QtWidgets.QMenu(self.menubar)
        self.menuEditor.setStyleSheet("QMenu{\n"
"color: white;\n"
"background-color: grey;\n"
"border: 2px white;\n"
"}")
        self.menuEditor.setObjectName("menuEditor")
        self.menuAppereance = QtWidgets.QMenu(self.menuEditor)
        self.menuAppereance.setObjectName("menuAppereance")
        self.menuFont_size = QtWidgets.QMenu(self.menuEditor)
        self.menuFont_size.setObjectName("menuFont_size")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionPreferences = QtWidgets.QAction(MainWindow)
        self.actionPreferences.setObjectName("actionPreferences")
        self.actionAutoStart = QtWidgets.QAction(MainWindow)
        self.actionAutoStart.setObjectName("actionAutoStart")
        self.actionCopy_Ctrl_C = QtWidgets.QAction(MainWindow)
        self.actionCopy_Ctrl_C.setObjectName("actionCopy_Ctrl_C")
        self.actionPaste_Ctrl_V = QtWidgets.QAction(MainWindow)
        self.actionPaste_Ctrl_V.setObjectName("actionPaste_Ctrl_V")
        self.actionPalletizing = QtWidgets.QAction(MainWindow)
        self.actionPalletizing.setObjectName("actionPalletizing")
        self.actionAutoStart_2 = QtWidgets.QAction(MainWindow)
        self.actionAutoStart_2.setCheckable(True)
        self.actionAutoStart_2.setChecked(False)
        self.actionAutoStart_2.setObjectName("actionAutoStart_2")
        self.actionDark = QtWidgets.QAction(MainWindow)
        self.actionDark.setObjectName("actionDark")
        self.actionWhite = QtWidgets.QAction(MainWindow)
        self.actionWhite.setObjectName("actionWhite")
        self.actionGrey = QtWidgets.QAction(MainWindow)
        self.actionGrey.setObjectName("actionGrey")
        self.action10_px = QtWidgets.QAction(MainWindow)
        self.action10_px.setObjectName("action10_px")
        self.action15_px = QtWidgets.QAction(MainWindow)
        self.action15_px.setObjectName("action15_px")
        self.action20px = QtWidgets.QAction(MainWindow)
        self.action20px.setObjectName("action20px")
        self.action12_5 = QtWidgets.QAction(MainWindow)
        self.action12_5.setObjectName("action12_5")
        self.action17_5 = QtWidgets.QAction(MainWindow)
        self.action17_5.setObjectName("action17_5")
        self.action7_5_px = QtWidgets.QAction(MainWindow)
        self.action7_5_px.setObjectName("action7_5_px")
        self.actionSave_File_As = QtWidgets.QAction(MainWindow)
        self.actionSave_File_As.setObjectName("actionSave_File_As")
        self.actionFanuc = QtWidgets.QAction(MainWindow)
        self.actionFanuc.setCheckable(True)
        self.actionFanuc.setObjectName("actionFanuc")
        self.actionUniversal_Robots = QtWidgets.QAction(MainWindow)
        self.actionUniversal_Robots.setCheckable(True)
        self.actionUniversal_Robots.setObjectName("actionUniversal_Robots")
        self.actionPalletizing_App = QtWidgets.QAction(MainWindow)
        self.actionPalletizing_App.setObjectName("actionPalletizing_App")
        self.menuFile.addAction(self.actionNew_2.menuAction())
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_File_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionPreferences)
        self.menuAppereance.addAction(self.actionDark)
        self.menuAppereance.addAction(self.actionWhite)
        self.menuAppereance.addAction(self.actionGrey)
        self.menuFont_size.addAction(self.action7_5_px)
        self.menuFont_size.addAction(self.action10_px)
        self.menuFont_size.addAction(self.action12_5)
        self.menuFont_size.addAction(self.action15_px)
        self.menuFont_size.addAction(self.action17_5)
        self.menuFont_size.addAction(self.action20px)
        self.menuEditor.addAction(self.menuAppereance.menuAction())
        self.menuEditor.addAction(self.menuFont_size.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuEditor.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.MainTab.setCurrentIndex(0)
        self.InfoTab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.recognizePalletButton.clicked.connect(self.sorting_app)
        self.playButton.clicked.connect(self.play_robot)
        self.homeButton.clicked.connect(self.go_home)
        # self.stopButton.clicked.connect(self.stop_sorting)

        self.pushButton_6.clicked.connect(self.start_capture)
        self.pushButton_5.clicked.connect(self.stop_capture)
        self.pushButton_5.setEnabled(False)
        pipeline, profile = init_camera()
        self.camera_thread = CameraThread(pipeline, profile)
        self.camera_thread.image_data.connect(self.update_image)
        self.camera_thread.depth_data.connect(self.update_image_depth)

    def start_capture(self):
            self.camera_thread.start()
            self.pushButton_6.setEnabled(False)
            self.pushButton_5.setEnabled(True)

    def stop_capture(self):
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.pushButton_6.setEnabled(True)
            self.pushButton_5.setEnabled(False)

    def update_image(self, image_data):
            h, w, ch = image_data.shape
            bytes_per_line = ch * w
            q_image = QImage(image_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.imageLabel.setPixmap(pixmap)
    def update_image_depth(self, depth_data):
            h, w, ch = depth_data.shape
            bytes_per_line = ch * w
            q_image = QImage(depth_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.imageLabel2.setPixmap(pixmap)
    def sorting_app(self):
            from projects import sorting_app

            result = sorting_app.result_list
            numbers_of_classes = len(result)
            self.console.insertPlainText(str(result))
            image_path = '../outputs/sorting_output_0_7_v2.jpg'
            pixmap = QPixmap(image_path)
            self.imageLabel3.setPixmap(pixmap)
            self.imageLabel3.setScaledContents(True)
            self.nr_classes_Input.setText(str(numbers_of_classes))
            self.current_object.setText(list(result.keys())[0])
            self.lineEdit.setText(str(result[list(result.keys())[0]][0][0]))
            self.lineEdit_3.setText(str(result[list(result.keys())[0]][0][1]))
            self.lineEdit_4.setText(str(result[list(result.keys())[0]][0][2]))
            self.lineEdit_8.setText('-')
            self.lineEdit_9.setText('-')
            self.lineEdit_10.setText('-')

    def go_home(self):
        homel, homej, originl, originj, approach, approachj, pickUp, pickUpJ = palletizing_app.get_all_positions(
                '../resources/parameters.yaml')
        robot = Robot('UR5', 'data.xlsx')
        robot.move_to_coords(homel, velocity=50, acceleration=50)

    def play_robot(self):
            from projects import sorting_app
            sorting_app.play()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Robot App Creator Tool"))
        self.recognizePalletButton.setText(_translate("MainWindow", "Recognize"))
        self.label_13.setText(_translate("MainWindow", "Number of Classes:"))
        self.label_15.setText(_translate("MainWindow", "Current object:"))
        self.label_12.setText(_translate("MainWindow", "X:"))
        self.label_19.setText(_translate("MainWindow", "Y:"))
        self.label_20.setText(_translate("MainWindow", "Z:"))
        self.label_24.setText(_translate("MainWindow", "RX:"))
        self.label_25.setText(_translate("MainWindow", "RY:"))
        self.label_26.setText(_translate("MainWindow", "RZ:"))
        self.MainTab.setTabText(self.MainTab.indexOf(self.palletView_tab), _translate("MainWindow", "Object Sorting"))
        self.robotIP.setText(_translate("MainWindow", "Robot IP:"))
        self.label.setText(_translate("MainWindow", "Payload [Kg]:"))
        self.label_5.setText(_translate("MainWindow", "  Z"))
        self.label_3.setText(_translate("MainWindow", "  X"))
        self.label_2.setText(_translate("MainWindow", "  W"))
        self.label_4.setText(_translate("MainWindow", "  Y"))
        self.label_6.setText(_translate("MainWindow", "TCP:"))
        self.label_9.setText(_translate("MainWindow", " Z"))
        self.label_7.setText(_translate("MainWindow", " X"))
        self.label_8.setText(_translate("MainWindow", " Y"))
        self.label_10.setText(_translate("MainWindow", "RX"))
        self.label_11.setText(_translate("MainWindow", "RZ"))
        self.label_16.setText(_translate("MainWindow", "RY"))
        self.j2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">J2:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.j3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">J3:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.j5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">J5:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.j4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">J4:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.j6.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">J6:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.j1.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">J1:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.InfoTab.setTabText(self.InfoTab.indexOf(self.joint_tab_2), _translate("MainWindow", "Joint Space"))
        self.xC.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:15pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">X:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.rxC.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">RX:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.yC.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">Y:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.ryC.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">RY:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.zC.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">Z:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.rzC.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">RZ:</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\"></span></p></body></html>"))
        self.InfoTab.setTabText(self.InfoTab.indexOf(self.cart_tab_2), _translate("MainWindow", "Cartesian Space"))
        self.MainTab.setTabText(self.MainTab.indexOf(self.stream), _translate("MainWindow", "Live Stream"))
        self.pushButton_6.setText(_translate("MainWindow", "Start Stream"))
        self.pushButton_5.setText(_translate("MainWindow", "Stop Stream"))
        self.playButton.setText(_translate("MainWindow", "Play"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.homeButton.setText(_translate("MainWindow", "Home"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionNew_2.setTitle(_translate("MainWindow", "New"))
        self.menuHelp.setStatusTip(_translate("MainWindow", "Help"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.menuEditor.setTitle(_translate("MainWindow", "Editor"))
        self.menuAppereance.setTitle(_translate("MainWindow", "Appereance"))
        self.menuFont_size.setTitle(_translate("MainWindow", "Font size"))
        self.actionLoad.setText(_translate("MainWindow", "Load"))
        self.actionLoad.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionPreferences.setText(_translate("MainWindow", "Preferences"))
        self.actionAutoStart.setText(_translate("MainWindow", "AutoStart"))
        self.actionCopy_Ctrl_C.setText(_translate("MainWindow", "Copy (Ctrl + C)"))
        self.actionPaste_Ctrl_V.setText(_translate("MainWindow", "Paste (Ctrl + V)"))
        self.actionPalletizing.setText(_translate("MainWindow", "Pallet App"))
        self.actionAutoStart_2.setText(_translate("MainWindow", "AutoStart"))
        self.actionDark.setText(_translate("MainWindow", "Dark"))
        self.actionWhite.setText(_translate("MainWindow", "White"))
        self.actionGrey.setText(_translate("MainWindow", "Grey"))
        self.action10_px.setText(_translate("MainWindow", "10 px"))
        self.action15_px.setText(_translate("MainWindow", "15 px"))
        self.action20px.setText(_translate("MainWindow", "20 px"))
        self.action12_5.setText(_translate("MainWindow", "13 px"))
        self.action17_5.setText(_translate("MainWindow", "18 px"))
        self.action7_5_px.setText(_translate("MainWindow", "8  px"))
        self.actionSave_File_As.setText(_translate("MainWindow", "Save File As"))
        self.actionSave_File_As.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionFanuc.setText(_translate("MainWindow", "Fanuc"))
        self.actionUniversal_Robots.setText(_translate("MainWindow", "Universal Robots"))
        self.actionPalletizing_App.setText(_translate("MainWindow", "Palletizing App"))
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow2()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
