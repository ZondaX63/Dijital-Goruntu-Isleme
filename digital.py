import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Import custom modules
from algorithms.sigmoid_functions import (
    standard_sigmoid, shifted_sigmoid, sloped_sigmoid, custom_sigmoid
)
from algorithms.hough_transform import (
    detect_lane_lines, detect_eyes, detect_circles
)
from algorithms.deblurring import deblur_image
from algorithms.object_analysis import (
    analyze_hyperspectral_image, visualize_regions
)
from utils.excel_export import create_excel_report

class DijiGorIsleGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(DijiGorIsleGUI, self).__init__()
        # Load UI file
        try:
            uic.loadUi('form.ui', self)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load UI file: {str(e)}")
            sys.exit(1)

        # Class variables
        self.current_image = None
        self.original_image = None
        self.file_path = None
        self.analysis_results = None

        # Setup widgets and connect signals
        self.setup_widgets()
        self.connect_signals()

        # Show home page initially
        self.stackedWidget.setCurrentIndex(0)
        self.statusbar.showMessage("Hazır")

    def setup_widgets(self):
        try:
            # StackedWidget pages
            self.homePage = self.findChild(QtWidgets.QWidget, 'homePage')
            self.homework1Page = self.findChild(QtWidgets.QWidget, 'homework1Page')
            self.homework2Page = self.findChild(QtWidgets.QWidget, 'homework2Page')
            self.homework3Page = self.findChild(QtWidgets.QWidget, 'homework3Page')
            self.homework4Page = self.findChild(QtWidgets.QWidget, 'homework4Page')
            self.homework5Page = self.findChild(QtWidgets.QWidget, 'homework5Page')
            self.homework6Page = self.findChild(QtWidgets.QWidget, 'homework6Page')

            # Image display labels for all pages
            self.imageDisplayLabel_home = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_home')
            self.imageDisplayLabel_hw1 = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_hw1')
            self.imageDisplayLabel_hw2 = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_hw2')
            self.imageDisplayLabel_hw3 = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_hw3')
            self.imageDisplayLabel_hw4 = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_hw4')
            self.imageDisplayLabel_hw5 = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_hw5')
            self.imageDisplayLabel_hw6 = self.findChild(QtWidgets.QLabel, 'imageDisplayLabel_hw6')

            # Buttons and controls for Homework 3
            self.sigmoidTypeComboBox = self.findChild(QtWidgets.QComboBox, 'sigmoidTypeComboBox')
            self.alphaSpinBox = self.findChild(QtWidgets.QDoubleSpinBox, 'alphaSpinBox')
            self.betaSpinBox = self.findChild(QtWidgets.QDoubleSpinBox, 'betaSpinBox')
            self.applySigmoidButton = self.findChild(QtWidgets.QPushButton, 'applySigmoidButton')
            self.resetButton3 = self.findChild(QtWidgets.QPushButton, 'resetButton3')

            # Buttons and controls for Homework 4
            self.houghTypeComboBox = self.findChild(QtWidgets.QComboBox, 'houghTypeComboBox')
            self.applyHoughButton = self.findChild(QtWidgets.QPushButton, 'applyHoughButton')
            self.resetButton4 = self.findChild(QtWidgets.QPushButton, 'resetButton4')

            # Buttons and controls for Homework 5
            self.deblurMethodComboBox = self.findChild(QtWidgets.QComboBox, 'deblurMethodComboBox')
            self.angleSpinBox = self.findChild(QtWidgets.QSpinBox, 'angleSpinBox')
            self.lengthSpinBox = self.findChild(QtWidgets.QSpinBox, 'lengthSpinBox')
            self.applyDeblurButton = self.findChild(QtWidgets.QPushButton, 'applyDeblurButton')
            self.resetButton5 = self.findChild(QtWidgets.QPushButton, 'resetButton5')

            # Buttons and controls for Homework 6
            self.hMinSpinBox = self.findChild(QtWidgets.QSpinBox, 'hMinSpinBox')
            self.hMaxSpinBox = self.findChild(QtWidgets.QSpinBox, 'hMaxSpinBox')
            self.sMinSpinBox = self.findChild(QtWidgets.QSpinBox, 'sMinSpinBox')
            self.sMaxSpinBox = self.findChild(QtWidgets.QSpinBox, 'sMaxSpinBox')
            self.analyzeButton = self.findChild(QtWidgets.QPushButton, 'analyzeButton')
            self.exportButton = self.findChild(QtWidgets.QPushButton, 'exportButton')
            self.resetButton6 = self.findChild(QtWidgets.QPushButton, 'resetButton6')

            # Menu actions
            self.actionGoruntuYukle = self.findChild(QtWidgets.QAction, 'actionGoruntuYukle')
            self.actionGoruntuKaydet = self.findChild(QtWidgets.QAction, 'actionGoruntuKaydet')
            self.actionCikis = self.findChild(QtWidgets.QAction, 'actionCikis')
            self.actionOdev1 = self.findChild(QtWidgets.QAction, 'actionOdev1')
            self.actionOdev2 = self.findChild(QtWidgets.QAction, 'actionOdev2')
            self.actionOdev3 = self.findChild(QtWidgets.QAction, 'actionOdev3')
            self.actionOdev4 = self.findChild(QtWidgets.QAction, 'actionOdev4')
            self.actionOdev5 = self.findChild(QtWidgets.QAction, 'actionOdev5')
            self.actionOdev6 = self.findChild(QtWidgets.QAction, 'actionOdev6')
            self.actionAnaSayfayaDon = self.findChild(QtWidgets.QAction, 'actionAnaSayfayaDon')
            self.actionHakkinda = self.findChild(QtWidgets.QAction, 'actionHakkinda')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to setup widgets: {str(e)}")
            sys.exit(1)

    def connect_signals(self):
        try:
            # Menu actions
            self.actionGoruntuYukle.triggered.connect(self.load_image)
            self.actionGoruntuKaydet.triggered.connect(self.save_image)
            self.actionCikis.triggered.connect(self.close)
            self.actionOdev1.triggered.connect(self.show_homework1)
            self.actionOdev2.triggered.connect(self.show_homework2)
            self.actionOdev3.triggered.connect(self.show_homework3)
            self.actionOdev4.triggered.connect(self.show_homework4)
            self.actionOdev5.triggered.connect(self.show_homework5)
            self.actionOdev6.triggered.connect(self.show_homework6)
            self.actionAnaSayfayaDon.triggered.connect(self.show_home_page)
            self.actionHakkinda.triggered.connect(self.show_about)

            # Homework 3 signals
            self.applySigmoidButton.clicked.connect(self.apply_sigmoid)
            self.resetButton3.clicked.connect(self.reset_image)

            # Homework 4 signals
            self.applyHoughButton.clicked.connect(self.apply_hough)
            self.resetButton4.clicked.connect(self.reset_image)

            # Homework 5 signals
            self.applyDeblurButton.clicked.connect(self.apply_deblur)
            self.resetButton5.clicked.connect(self.reset_image)

            # Homework 6 signals
            self.analyzeButton.clicked.connect(self.analyze_image)
            self.exportButton.clicked.connect(self.export_results)
            self.resetButton6.clicked.connect(self.reset_image)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect signals: {str(e)}")
            sys.exit(1)

    def show_home_page(self):
        self.stackedWidget.setCurrentIndex(0)
        self.statusbar.showMessage("Ana Sayfa")

    def show_homework1(self):
        self.stackedWidget.setCurrentIndex(1)
        self.statusbar.showMessage("Ödev 1: Temel İşlevselliği Oluştur")

    def show_homework2(self):
        self.stackedWidget.setCurrentIndex(2)
        self.statusbar.showMessage("Ödev 2: Filtre Uygulama")

    def show_homework3(self):
        self.stackedWidget.setCurrentIndex(3)
        self.statusbar.showMessage("Ödev 3: S-Curve Kontrast Güçlendirme")

    def show_homework4(self):
        self.stackedWidget.setCurrentIndex(4)
        self.statusbar.showMessage("Ödev 4: Hough Transform Uygulamaları")

    def show_homework5(self):
        self.stackedWidget.setCurrentIndex(5)
        self.statusbar.showMessage("Ödev 5: Deblurring Algoritması")

    def show_homework6(self):
        self.stackedWidget.setCurrentIndex(6)
        self.statusbar.showMessage("Ödev 6: Nesne Sayma ve Özellik Çıkarma")

    def load_image(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Görüntü Dosyası Aç",
                "",
                "Görüntü Dosyaları (*.png *.jpg *.bmp *.jpeg);;Tüm Dosyalar (*)",
                options=options
            )

            if file_path:
                # Check if file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {file_path}")

                # Store image
                self.file_path = file_path
                self.original_image = image
                self.current_image = image.copy()

                # Display image on all pages
                self.display_image(self.current_image)

                # Enable all controls
                self.enable_all_controls()

                # Update status
                self.statusbar.showMessage(f"Görüntü yüklendi: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def display_image(self, image):
        if image is None:
            return

        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert image to QImage
            height, width = image.shape[:2]
            bytes_per_line = 3 * width if len(image.shape) == 3 else width
            format_ = QImage.Format_RGB888 if len(image.shape) == 3 else QImage.Format_Grayscale8

            q_image = QImage(image.data, width, height, bytes_per_line, format_)

            if q_image.isNull():
                raise ValueError("Failed to create QImage")

            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)

            # Update all image labels
            for label in [self.imageDisplayLabel_home, self.imageDisplayLabel_hw1, self.imageDisplayLabel_hw2,
                          self.imageDisplayLabel_hw3, self.imageDisplayLabel_hw4, self.imageDisplayLabel_hw5,
                          self.imageDisplayLabel_hw6]:
                if label is not None:
                    label_size = label.size()
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            label_size,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        label.setPixmap(scaled_pixmap)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Image display error: {str(e)}")

    def apply_sigmoid(self):
        if self.current_image is not None:
            try:
                # Get parameters
                sigmoid_type = self.sigmoidTypeComboBox.currentText()
                alpha = self.alphaSpinBox.value()
                beta = self.betaSpinBox.value()

                # Apply selected sigmoid function
                if sigmoid_type == "Standart Sigmoid":
                    self.current_image = standard_sigmoid(self.current_image, alpha)
                elif sigmoid_type == "Yatay Kaydırılmış Sigmoid":
                    self.current_image = shifted_sigmoid(self.current_image, alpha, beta)
                elif sigmoid_type == "Eğimli Sigmoid":
                    self.current_image = sloped_sigmoid(self.current_image, alpha, beta)
                elif sigmoid_type == "Özel Sigmoid":
                    self.current_image = custom_sigmoid(self.current_image, alpha, beta, 1.0)

                # Update display
                self.display_image(self.current_image)
                self.statusbar.showMessage(f"{sigmoid_type} uygulandı")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply sigmoid: {str(e)}")

    def apply_hough(self):
        if self.current_image is not None:
            try:
                # Get selected detection type
                detection_type = self.houghTypeComboBox.currentText()

                # Apply selected detection
                if detection_type == "Yol Çizgisi Tespiti":
                    self.current_image = detect_lane_lines(self.current_image)
                elif detection_type == "Göz Tespiti":
                    self.current_image = detect_eyes(self.current_image)

                # Update display
                self.display_image(self.current_image)
                self.statusbar.showMessage(f"{detection_type} uygulandı")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply Hough transform: {str(e)}")

    def apply_deblur(self):
        if self.current_image is not None:
            try:
                # Get parameters
                method = self.deblurMethodComboBox.currentText().lower().replace(" ", "_")
                angle = self.angleSpinBox.value()
                length = self.lengthSpinBox.value()

                # Apply deblurring
                self.current_image = deblur_image(
                    self.current_image,
                    method=method,
                    angle=angle,
                    length=length
                )

                # Update display
                self.display_image(self.current_image)
                self.statusbar.showMessage(f"Deblurring uygulandı")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply deblurring: {str(e)}")

    def analyze_image(self):
        if self.current_image is not None:
            try:
                # Get HSV parameters
                hsv_lower = (
                    self.hMinSpinBox.value(),
                    self.sMinSpinBox.value(),
                    50
                )
                hsv_upper = (
                    self.hMaxSpinBox.value(),
                    self.sMaxSpinBox.value(),
                    255
                )

                # Analyze image
                self.analysis_results, mask = analyze_hyperspectral_image(
                    self.current_image,
                    hsv_lower=hsv_lower,
                    hsv_upper=hsv_upper
                )

                # Visualize results
                vis_image, mask_vis = visualize_regions(
                    self.current_image,
                    self.analysis_results['contours'],
                    mask
                )

                # Update display
                self.current_image = vis_image
                self.display_image(self.current_image)
                self.statusbar.showMessage("Analiz tamamlandı")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to analyze image: {str(e)}")

    def export_results(self):
        if self.analysis_results is not None:
            try:
                # Get save file path
                options = QFileDialog.Options()
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Excel Dosyasını Kaydet",
                    "",
                    "Excel Dosyaları (*.xlsx);;Tüm Dosyalar (*)",
                    options=options
                )

                if file_path:
                    # Export to Excel
                    create_excel_report(self.analysis_results, file_path)
                    self.statusbar.showMessage(f"Sonuçlar kaydedildi: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")

    def enable_all_controls(self):
        """Enable all controls when an image is loaded"""
        # Homework 3 controls
        self.sigmoidTypeComboBox.setEnabled(True)
        self.alphaSpinBox.setEnabled(True)
        self.betaSpinBox.setEnabled(True)
        self.applySigmoidButton.setEnabled(True)
        self.resetButton3.setEnabled(True)

        # Homework 4 controls
        self.houghTypeComboBox.setEnabled(True)
        self.applyHoughButton.setEnabled(True)
        self.resetButton4.setEnabled(True)

        # Homework 5 controls
        self.deblurMethodComboBox.setEnabled(True)
        self.angleSpinBox.setEnabled(True)
        self.lengthSpinBox.setEnabled(True)
        self.applyDeblurButton.setEnabled(True)
        self.resetButton5.setEnabled(True)

        # Homework 6 controls
        self.hMinSpinBox.setEnabled(True)
        self.hMaxSpinBox.setEnabled(True)
        self.sMinSpinBox.setEnabled(True)
        self.sMaxSpinBox.setEnabled(True)
        self.analyzeButton.setEnabled(True)
        self.exportButton.setEnabled(True)
        self.resetButton6.setEnabled(True)

    def reset_image(self):
        if self.original_image is not None:
            try:
                # Restore original image
                self.current_image = self.original_image.copy()

                # Update display
                self.display_image(self.current_image)

                # Update status
                self.statusbar.showMessage("Orijinal görüntüye dönüldü")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset image: {str(e)}")

    def save_image(self):
        if self.current_image is not None:
            try:
                options = QFileDialog.Options()
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Görüntüyü Kaydet",
                    "",
                    "PNG Dosyaları (*.png);;JPEG Dosyaları (*.jpg);;BMP Dosyaları (*.bmp)",
                    options=options
                )

                if file_path:
                    # Save image
                    cv2.imwrite(file_path, self.current_image)

                    # Update status
                    self.statusbar.showMessage(f"Görüntü kaydedildi: {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def show_about(self):
        QMessageBox.information(
            self,
            "Hakkında",
            "Dijital Görüntü İşleme Uygulaması\n\n"
            "Öğrenci: İzzet AYDIN\n"
            "Öğrenci No: 221229072\n\n"
            "Bu uygulama, Dijital Görüntü İşleme dersi ödevleri için geliştirilmiştir."
        )

if __name__ == "__main__":
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = DijiGorIsleGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application crashed: {str(e)}")
        import traceback
        traceback.print_exc()
