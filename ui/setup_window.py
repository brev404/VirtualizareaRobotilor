# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setup.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets
from ui.palletizing_window import Ui_MainWindow
from ui.sorting_window import Ui_MainWindow2
from ui.depalletizing_window import Ui_MainWindow3


class Ui_Form2(object):
    def setupUi(self, Form2):
        Form2.setObjectName("Form2")
        Form2.resize(600, 400)
        Form2.setMinimumSize(QtCore.QSize(600, 400))
        Form2.setMaximumSize(QtCore.QSize(600, 400))
        Form2.setStyleSheet("background-color: #3c4957;")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Form2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Form2)
        self.label.setMinimumSize(QtCore.QSize(200, 40))
        self.label.setMaximumSize(QtCore.QSize(200, 40))
        self.label.setStyleSheet("background-color: #3c4957;\n"
                                 "font: 12pt \"Verdana\";\n"
                                 "color: rgb(211, 211, 211);")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(Form2)
        self.label_2.setMinimumSize(QtCore.QSize(200, 40))
        self.label_2.setMaximumSize(QtCore.QSize(200, 40))
        self.label_2.setStyleSheet("background-color: #3c4957;\n"
                                   "font: 12pt \"Verdana\";\n"
                                   "color: rgb(211, 211, 211);")
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem1)
        self.label_3 = QtWidgets.QLabel(Form2)
        self.label_3.setMinimumSize(QtCore.QSize(200, 40))
        self.label_3.setMaximumSize(QtCore.QSize(200, 40))
        self.label_3.setStyleSheet("background-color: #3c4957;\n"
                                   "font: 12pt \"Verdana\";\n"
                                   "color: rgb(211, 211, 211);")
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        spacerItem2 = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.comboBox = QtWidgets.QComboBox(Form2)
        self.comboBox.setMinimumSize(QtCore.QSize(300, 40))
        self.comboBox.setMaximumSize(QtCore.QSize(300, 40))
        self.comboBox.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                    "font: 12pt \"Verdana\";")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox)
        spacerItem5 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem5)
        self.comboBox_2 = QtWidgets.QComboBox(Form2)
        self.comboBox_2.setMinimumSize(QtCore.QSize(300, 40))
        self.comboBox_2.setMaximumSize(QtCore.QSize(300, 40))
        self.comboBox_2.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                      "font: 12pt \"Verdana\";\n"
                                      "")
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_2)
        spacerItem6 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem6)
        self.comboBox_3 = QtWidgets.QComboBox(Form2)
        self.comboBox_3.setMinimumSize(QtCore.QSize(300, 40))
        self.comboBox_3.setMaximumSize(QtCore.QSize(300, 40))
        self.comboBox_3.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                      "font: 12pt \"Verdana\";")
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_3)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem7)
        self.pushButton = QtWidgets.QPushButton(Form2)
        self.pushButton.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton.setMaximumSize(QtCore.QSize(100, 30))
        self.pushButton.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                      "font: 12pt \"Verdana\";")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton, 0, QtCore.Qt.AlignRight)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.retranslateUi(Form2)
        QtCore.QMetaObject.connectSlotsByName(Form2)

        self.pushButton.clicked.connect(self.project_screen)

    def project_screen(self):
        selected_project = self.comboBox.currentText()
        if selected_project == "Palletizing":
            self.MainWindow = QtWidgets.QMainWindow()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self.MainWindow)
            self.MainWindow.show()
        elif selected_project == "Object sorting":
            self.MainWindow2 = QtWidgets.QMainWindow()
            self.ui = Ui_MainWindow2()
            self.ui.setupUi(self.MainWindow2)
            self.MainWindow2.show()
        elif selected_project == "Depalletizing":
            self.MainWindow3 = QtWidgets.QMainWindow()
            self.ui = Ui_MainWindow3()
            self.ui.setupUi(self.MainWindow3)
            self.MainWindow3.show()

    def retranslateUi(self, Form2):
        _translate = QtCore.QCoreApplication.translate
        Form2.setWindowTitle(_translate("Form2", "Form"))
        self.label.setText(_translate("Form2", "Project type:"))
        self.label_2.setText(_translate("Form2", "Robot type:"))
        self.label_3.setText(_translate("Form2", "Camera type:"))
        self.comboBox.setItemText(0, _translate("Form2", "Select project"))
        self.comboBox.setItemText(1, _translate("Form2", "Palletizing"))
        self.comboBox.setItemText(2, _translate("Form2", "Depalletizing"))
        self.comboBox.setItemText(3, _translate("Form2", "Object sorting"))
        self.comboBox_2.setItemText(0, _translate("Form2", "Select robot"))
        self.comboBox_2.setItemText(1, _translate("Form2", "UR5"))
        self.comboBox_2.setItemText(2, _translate("Form2", "Fanuc CRX10"))
        self.comboBox_3.setItemText(0, _translate("Form2", "Select camera"))
        self.comboBox_3.setItemText(1, _translate("Form2", "Intel Realsense D435f"))
        self.comboBox_3.setItemText(2, _translate("Form2", "Cognex 9000"))
        self.comboBox_3.setItemText(3, _translate("Form2", "Kinect XboxOne"))
        self.pushButton.setText(_translate("Form2", "Next"))

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     Form2 = QtWidgets.QWidget()
#     ui = Ui_Form2()
#     ui.setupUi(Form2)
#     Form2.show()
#     sys.exit(app.exec_())
