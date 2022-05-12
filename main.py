import time
from socket import timeout
from struct import unpack

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow,QDialog, QApplication, QWidget, QTableWidgetItem, QGridLayout
from PyQt5.uic import loadUi
from PyQt5.QtGui import *
import cv2
import serial, sys
from PyQt5.QtCore import *

ser = serial.Serial('COM8', 9600, timeout=1)

data_1 = []
data_2 = []
data_3 = []
data_4 = []


class Worker(QThread):
    ImageUpdate = pyqtSignal(QImage)

    modelConfiguration = 'yolov4-tiny_kastem.cfg'
    modelWeights = 'yolov4-tiny_kastem_last.weights'
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    classesFile = 'classes.names'
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    whT = 320

    def run(self):
        Capture = cv2.VideoCapture(0)
        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.whT, self.whT), [0, 0, 0], 1, crop=False)
                self.net.setInput(blob)
                layerNames = self.net.getLayerNames()
                outputNames = [layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
                outputs = self.net.forward(outputNames)
                self.findobject(outputs, frame)

                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)

                Convert2QtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = Convert2QtFormat.scaled(281, 221, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def findobject(self,outputs,img):
        confThreshold = 0.3
        nmsThreshold = 0.3
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2]*wT), int(det[3]*hT)
                    x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        print(indices)
        for i in indices:
            box = bbox[i]
            x,y,w,h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # cv2.putText(img, f'{self.classNames[classIds[i]].upper()}{int(confs[i] * 100)}%',
            #             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)




class SerialThread(QThread):
    d1 = pyqtSignal(float)
    d2 = pyqtSignal(float)
    d3 = pyqtSignal(float)
    d4 = pyqtSignal(float)

    def run(self):
        #global d1, d2, d3, d4

        while True:
            dataRaw = ser.read()
            if (dataRaw == b'@'):
                dataRaw = ser.read(16)

                self.d1 = unpack('f', dataRaw[0:4])[0]
                self.d2 = unpack('f', dataRaw[4:8])[0]
                self.d3 = unpack('f', dataRaw[8:12])[0]
                self.d4 = unpack('f', dataRaw[12:16])[0]

            data_1.append('{:.2f}'.format(self.d1))
            data_2.append('{:.2f}'.format(self.d2))
            data_3.append('{:.2f}'.format(self.d3))
            data_4.append('{:.2f}'.format(self.d4))


class GraphScreen(QMainWindow):
    def __init__(self):
        super(GraphScreen, self).__init__()
        loadUi("UI.ui", self)

        self.pushButton_2.clicked.connect(self.display)
        self.cameraButton.clicked.connect(self.camDisplay)

    def camDisplay(self):
        self.cam = Worker()
        self.cam.start()
        self.cam.ImageUpdate.connect(self.imageUpdateSlot)
        self.cameraButton.setEnabled(False)

    def imageUpdateSlot(self, Image):
        self.label_7.setPixmap(QPixmap.fromImage(Image))

    def display(self):
        self.textBrowser.setText("0")
        self.textBrowser_2.setText("0")
        self.textBrowser_3.setText("0")
        self.textBrowser_4.setText("0")

        self.serth = SerialThread()
        self.serth.start()

        self.qTimer = QTimer()
        self.qTimer.setInterval(100)
        self.qTimer.timeout.connect(self.check)
        self.qTimer.start()
        self.pushButton_2.setEnabled(False)

    def check(self):
        self.textBrowser.setText("{:.2f}".format(self.serth.d1))
        self.textBrowser_2.setText("{:.2f}".format(self.serth.d2))
        self.textBrowser_3.setText("{:.2f}".format(self.serth.d3))
        self.textBrowser_4.setText("{:.2f}".format(self.serth.d4))


        #time.sleep(2)


app = QApplication(sys.argv)
graph = GraphScreen()
widget = QtWidgets.QStackedWidget()
widget.addWidget(graph)
widget.setFixedHeight(326)
widget.setFixedWidth(696)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("exiting")