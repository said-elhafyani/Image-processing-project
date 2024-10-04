import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout , QMenu ,QFrame, QLabel ,QInputDialog
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QTransform
import numpy as np
import matplotlib.pyplot as plt
from qimage2ndarray import rgb_view

 # Définir une méthode pour gérer le survol des boutons
def apply_hover_style(button):
    button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #ffffff;
                border-radius: 2px;
                padding: 12px 10px;
            }
            QPushButton:hover {
                background-color: #1469a1;
            }
        """)
    button.setCursor(Qt.PointingHandCursor)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        style_file = QFile("styles.css")
        style_file.open(QFile.ReadOnly | QFile.Text)
        style_sheet = style_file.readAll()
        QApplication.instance().setStyleSheet(style_sheet.data().decode('utf-8'))

        self.init_ui()

     


    def init_ui(self):
        # Créer un bouton pour importer des fichiers
        self.button_Image = QPushButton('Image', self)
        self.button_Image.clicked.connect(self.on_button_Image_click)
        
        self.button_Transformation = QPushButton('Transformation', self)
        self.button_Transformation.clicked.connect(self.on_button_Transformation_click)

        self.button_Binarisation = QPushButton('Binarisation', self)
        self.button_Binarisation.clicked.connect(self.on_button_Binarisation_click)

        self.button_Filtrage = QPushButton('Filtrage', self)
        self.button_Filtrage.clicked.connect(self.on_button_Filtrage_click)
        
        self.button_Extraction_Contours = QPushButton("Extraction Contours", self)
        self.button_Extraction_Contours.clicked.connect(self.on_button_Extraction_Contours_click)

        self.button_Morphologie = QPushButton("Morphologie", self)
        self.button_Morphologie.clicked.connect(self.on_button_Morphologie_click)

        self.button_Segmentation = QPushButton("Segmentation", self)
        self.button_Segmentation.clicked.connect(self.on_button_Segmentation_click)

        self.button_Point_Interet = QPushButton("Point d'iteret", self)
        self.button_Point_Interet.clicked.connect(self.on_button_Point_Interet_click)

        self.button_Compression = QPushButton("Compression", self)
        self.button_Compression.clicked.connect(self.on_button_Compression_click)   

        self.button_reinitialiser = QPushButton("Reinitialiser l'image", self)
        self.button_reinitialiser.clicked.connect(self.on_button_reinitialiser_click)


        # Créer les menus pour les sous-options
        self.image_menu = QMenu(self)
        self.image_menu.addAction('Ouvrir', self.on_button_ouvrir_click)
        self.image_menu.addAction('Enregistrer', self.on_button_enregistrer_click)
        self.image_menu.addAction('Quitter', self.on_button_quitter_click)

        self.filtrage_menu = QMenu(self)
        self.filtrage_menu.addAction('Gaussien', self.on_button_gaussien_click)
        self.filtrage_menu.addAction('Moyenneur', self.on_button_moyenneur_click)
        self.filtrage_menu.addAction('Median', self.on_button_median_click)    

        self.transformation_menu = QMenu(self)
        self.transformation_menu.addAction('Rotation', self.on_button_rotation_click)
        self.transformation_menu.addAction('Negative', self.on_button_negative_click)
        self.transformation_menu.addAction('Histogramme RGB', self.on_button_histogramme_rgb_click)
        self.transformation_menu.addAction('Histogramme NG', self.on_button_histogramme_ng_click)
        self.transformation_menu.addAction('Redimension', self.on_button_redimension_click)
        self.transformation_menu.addAction('Egalisation', self.on_button_egalisation_click)
        self.transformation_menu.addAction('Etirement', self.on_button_Etirement_click)
        self.transformation_menu.addAction('Rectangle', self.on_button_Rectangle_click)
                                           

        self.binarisation_menu = QMenu(self)
        self.binarisation_menu.addAction('Seuillage', self.on_button_seuillage_click)
        self.binarisation_menu.addAction('OTSU', self.on_button_otsu_click)
        self.binarisation_menu.addAction('Moyenne pondérée', self.on_button_moyenne_ponderee_click)

        self.extraction_contours_menu = QMenu(self)
        self.extraction_contours_menu.addAction('Gradient', self.on_button_Gradient_click)
        self.extraction_contours_menu.addAction('Laplacien', self.on_button_Laplacien_click)
        self.extraction_contours_menu.addAction('Sobel', self.on_button_Sobel_click)
        self.extraction_contours_menu.addAction('Robert', self.on_button_robert_click)

        self.morphologie_menu = QMenu(self)
        self.morphologie_menu.addAction('Dilatation', self.on_button_dilatation_click)
        self.morphologie_menu.addAction('Erosion', self.on_button_erosion_click)
        self.morphologie_menu.addAction('Ouverture', self.on_button_ouverture_click)
        self.morphologie_menu.addAction('Fermeture', self.on_button_fermeture_click)
        self.morphologie_menu.addAction('Filtrage Morphologique', self.on_button_filtrage_morphologique_click)

        self.segmentation_menu = QMenu(self)
        self.segmentation_menu.addAction('K-means', self.on_button_k_means_click)
        self.segmentation_menu.addAction('Partition de regions D', self.on_button_partition_regions_d_click)
        self.segmentation_menu.addAction('Croissance de regions D', self.on_button_croissance_regions_d_click)

        self.point_interet_menu = QMenu(self)
        self.point_interet_menu.addAction('Harris', self.on_button_harris_click)
        self.point_interet_menu.addAction('shi tomasi', self.on_button_shi_tomasi_click)
        self.point_interet_menu.addAction('hough circles', self.on_button_hough_circles_click)
        self.point_interet_menu.addAction('hough lines', self.on_button_hough_lines_click)

        self.compression_menu = QMenu(self)
        self.compression_menu.addAction('Huffman', self.on_button_huffman_click)
        self.compression_menu.addAction('Ondelette', self.on_button_ondelette_click)
        self.compression_menu.addAction('LZW', self.on_button_lzw_click)
        

        apply_hover_style(self.button_Image)
        apply_hover_style(self.button_Transformation)
        apply_hover_style(self.button_Binarisation)
        apply_hover_style(self.button_Filtrage)
        apply_hover_style(self.button_Extraction_Contours)
        apply_hover_style(self.button_Morphologie)
        apply_hover_style(self.button_Segmentation)
        apply_hover_style(self.button_Point_Interet)
        apply_hover_style(self.button_Compression)
        apply_hover_style(self.button_reinitialiser)



         # Disposition des widgets dans une mise en page verticale
        button_container = QFrame(self)
        button_container.setFrameStyle(QFrame.Panel | QFrame.Raised)  # Style de bordure
        button_container.setStyleSheet("""
        background-color: #00031b;
        padding: 10px;
        border: 1px solid #ccc;
        """)
        # Créer un QVBoxLayout pour le conteneur des boutons
        button_container_layout = QVBoxLayout(button_container)
        button_container_layout.addWidget(self.button_Image)
        button_container_layout.addWidget(self.button_Transformation)
        button_container_layout.addWidget(self.button_Binarisation)
        button_container_layout.addWidget(self.button_Filtrage)
        button_container_layout.addWidget(self.button_Extraction_Contours)
        button_container_layout.addWidget(self.button_Morphologie)
        button_container_layout.addWidget(self.button_Segmentation)
        button_container_layout.addWidget(self.button_Point_Interet)
        button_container_layout.addWidget(self.button_Compression)
        button_container_layout.addWidget(self.button_reinitialiser)
        
        

        topBare = QFrame(self)
        topBare.setStyleSheet("background-color: #00031b; border: 0; margin: 0;")
        topBare.setGeometry(298, 1, 1400, 90)
        

        label_image = QLabel("Image Processing", topBare)
        label_image.setGeometry(350, 20, 400, 50)  # Position et taille du QLabel à l'intérieur de frame1
        label_image.setStyleSheet(" color: #ffffff; font-size: 40px; font-weight: bold;")  # Styles CSS pour le QLabel

        label_image_orginal = QLabel("Image Original", self)
        label_image_orginal.setGeometry(450, 150, 400, 50)  # Position et taille du QLabel à l'intérieur de frame1
        label_image_orginal.setStyleSheet("color: black; font-size: 30px; font-weight: bold;")  # Styles CSS pour le QLabel

        self.label_show_image_originale = QLabel(self)
        self.label_show_image_originale.setGeometry(370, 230, 420, 420)  # Définir la taille du QLabel
        self.label_show_image_originale.setScaledContents(True) # Redimensionner l'image pour s'adapter à la taille du QLabel

        self.label_show_image_transformer = QLabel(self)
        self.label_show_image_transformer.setGeometry(900, 230, 420, 420)  # Définir la taille du QLabel
        self.label_show_image_transformer.setScaledContents(True) # Redimensionner l'image pour s'adapter à la taille du QLabel

        label_image_transformer = QLabel("Image Transformer", self)
        label_image_transformer.setGeometry(970, 150, 400, 50)  # Position et taille du QLabel à l'intérieur de frame1
        label_image_transformer.setStyleSheet("color: black; font-size: 30px; font-weight: bold;")  # Styles CSS pour le QLabel
        
        line_frame = QFrame(self)
        line_frame.setStyleSheet("background-color: black; border: 0; margin: 0;")
        line_frame.setGeometry(840, 200, 3, 470)
        self.image_path = None
        self.setStyleSheet("background-color: #ffffff; border: 0; margin: 0;")  
        button_container.setGeometry(0, 0, 300, 700)
        icon = QIcon("image1.jpg")
        self.setWindowIcon(icon)
        self.setWindowTitle('Traitement d\'image')
        self.setGeometry(100, 100, 1400, 700)
        self.setFixedSize(1400, 700)
        # Afficher la fenêtre principale
        self.show()

    def apply_k_means(self, image, num_classes):
        # Convertir l'image en un tableau numpy
        pixels = np.float32(image.reshape(-1, 3))

        # Définir les critères pour l'algorithme K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Appliquer K-Means
        _, labels, centers = cv2.kmeans(pixels, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convertir les centres en valeurs entières
        centers = np.uint8(centers)

        # Convertir les pixels en fonction des étiquettes
        segmented_image = centers[labels.flatten()]

        # Remettre l'image sous sa forme d'origine
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image
    
    def convert_cv2_to_qimage(self, image):
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            # This is a color image
            channels = image.shape[2]
            bytesPerLine = channels * width
            format = QImage.Format_RGB888
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # This is a grayscale image
            bytesPerLine = width
            format = QImage.Format_Grayscale8
    
        return QImage(image.data, width, height, bytesPerLine, format)
    
    def load_image(self,file_path):
    
        pixmap = QPixmap(file_path)
        self.image_path = file_path
        self.label_show_image_originale.setPixmap(pixmap)
        self.label_show_image_originale.setAlignment(Qt.AlignCenter) 
        self.label_show_image_transformer.setPixmap(pixmap)
        self.label_show_image_transformer.setAlignment(Qt.AlignCenter)

    def on_button_Image_click(self):
        self.image_menu.exec_(self.button_Image.mapToGlobal(self.button_Image.rect().bottomLeft()))
    
    def on_button_reinitialiser_click(self):
        pixmap = QPixmap(self.image_path)
        self.label_show_image_transformer.setPixmap(pixmap)
        self.label_show_image_transformer.setAlignment(Qt.AlignCenter)
        self.label_show_image_transformer.setGeometry(900, 230, 420, 420)
        

    def on_button_Transformation_click(self):
        self.transformation_menu.exec_(self.button_Transformation.mapToGlobal(self.button_Transformation.rect().bottomLeft()))

    def on_button_Binarisation_click(self):
        self.binarisation_menu.exec_(self.button_Binarisation.mapToGlobal(self.button_Binarisation.rect().bottomLeft()))

    def on_button_Filtrage_click(self):
        # Ici vous pouvez écrire le code pour effectuer un filtrage d'image et l'afficher dans self.image_transformed
       self.filtrage_menu.exec_(self.button_Filtrage.mapToGlobal(self.button_Filtrage.rect().bottomLeft()))

    def on_button_Extraction_Contours_click(self):
        self.extraction_contours_menu.exec_(self.button_Extraction_Contours.mapToGlobal(self.button_Extraction_Contours.rect().bottomLeft()))

    def on_button_Morphologie_click(self):
        self.morphologie_menu.exec_(self.button_Morphologie.mapToGlobal(self.button_Morphologie.rect().bottomLeft()))

    def on_button_Segmentation_click(self):
        self.segmentation_menu.exec_(self.button_Segmentation.mapToGlobal(self.button_Segmentation.rect().bottomLeft()))

    def on_button_Point_Interet_click(self):
        self.point_interet_menu.exec_(self.button_Point_Interet.mapToGlobal(self.button_Point_Interet.rect().bottomLeft()))

    def on_button_Compression_click(self):
        self.compression_menu.exec_(self.button_Compression.mapToGlobal(self.button_Compression.rect().bottomLeft()))

    def on_button_ouvrir_click(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Ouvrir une image', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif)')
        if file_path:
            self.load_image(file_path)
    
    def on_button_median_click(self):
        tailefenetre, ok = QInputDialog.getInt(self, "Taille de Fenétre", "Entrer la taile de la fenétre:")
        if ok :
            image = cv2.imread(self.image_path)
            median_blurred_image = cv2.medianBlur(image, tailefenetre)
            q_image = self.convert_cv2_to_qimage(median_blurred_image)
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)

    def on_button_enregistrer_click(self):
       # Demander à l'utilisateur de choisir l'emplacement et le nom du fichier
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            # Récupérer la pixmap de l'image affichée dans image_label
            pixmap = self.label_show_image_transformer.pixmap()
            if pixmap:
                # Enregistrer l'image avec le format choisi
                pixmap.save(file_path)

    def on_button_quitter_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
  
    
    def on_button_gaussien_click(self):
        image = cv2.imread(self.image_path)
        # Appliquer un filtre gaussien sur l'image
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        q_image = self.convert_cv2_to_qimage(blurred_image)
        # Afficher l'image segmentée dans la QLabel centrale
        pixmap = QPixmap.fromImage(q_image)
        self.label_show_image_transformer.setPixmap(pixmap)

    def on_button_moyenneur_click(self):
        taileFilter, ok = QInputDialog.getInt(self, "Taille de filtre", "Entrer la taile de filtre:")
        if ok :
            image = cv2.imread(self.image_path)
            # Appliquer un filtre gaussien sur l'image
            averaged_image = cv2.blur(image, (taileFilter, taileFilter)) 
            q_image = self.convert_cv2_to_qimage(averaged_image)
            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)

    def on_button_rotation_click(self):
        if self.image_path:
            angle, ok = QInputDialog.getDouble(self, "Rotation", "Enter the rotation angle:")
            if ok:
                image = QImage(self.image_path)
                transform = QTransform().rotate(angle)
                rotated_image = image.transformed(transform)
                self.label_show_image_transformer.setPixmap(QPixmap.fromImage(rotated_image))
        
            
    def on_button_negative_click(self):
        if self.image_path:
            original_image = cv2.imread(self.image_path)

            # Obtenir l'image négative
            negative_image = 255 - original_image

            # Convertir l'image OpenCV en pixmap pour l'affichage dans QLabel
            height, width, _ = negative_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(negative_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)

            # Afficher l'image négative dans image_label
            self.label_show_image_transformer.setPixmap(pixmap)


    def on_button_histogramme_rgb_click(self):
        image = cv2.imread(self.image_path)
        if image is not None:
            blue_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            red_channel = image[:, :, 2]
        
            # Calculer les histogrammes pour chaque canal de couleur
            hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])
            hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
            hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
        
            # Afficher les histogrammes
            plt.plot(hist_blue, color='blue', label='Canal Bleu')
            plt.plot(hist_green, color='green', label='Canal Vert')
            plt.plot(hist_red, color='red', label='Canal Rouge')
            plt.xlabel('Intensité de la couleur')
            plt.ylabel('Nombre de pixels')
            plt.title("Histogramme en niveaux RGB de l'image")
            plt.legend()
            plt.show()


    def on_button_histogramme_ng_click(self):
        # Get the original image
        image = cv2.imread(self.image_path)
        if image is not None:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            # Display the histogram
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(hist, color='gray')
            plt.xlabel('Niveau de gris')
            plt.ylabel('Nombre de pixels')
            plt.title("Histogramme en niveaux de gris de l'image originale")
            
            # Get the transformed image from label_show_image_transformer
            pixmap = self.label_show_image_transformer.pixmap()
            qimage = pixmap.toImage()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            transformed_image = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)  # Copies the data
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGBA2GRAY)
    
            # Calculate and display the histogram of the transformed image
            hist_transformed = cv2.calcHist([transformed_image], [0], None, [256], [0, 256])
            plt.subplot(1, 2, 2)
            plt.plot(hist_transformed, color='gray')
            plt.xlabel('Niveau de gris')
            plt.ylabel('Nombre de pixels')
            plt.title("Histogramme en niveaux de gris de l'image transformée")
            
            plt.show()
    def on_button_redimension_click(self):
        # Obtenir la taille actuelle de l'image originale
        current_size = self.label_show_image_originale.pixmap().size()
        new_size, ok = QInputDialog.getText(self, "Redimensionner l'image", "Nouvelle taille (largeur, hauteur) ou pourcentage de redimensionnement:")
        
        if ok:
            if "%" in new_size:
                percentage = int(new_size.strip("%"))
                print(percentage)
                new_width = int(current_size.width() * percentage / 100)
                new_height = int(current_size.height() * percentage / 100)
            else:
                 new_width, new_height = map(int, new_size.strip('()').split(","))
            self.label_show_image_transformer.setGeometry(900, 230, new_width, new_height)   
       

            

    
    def on_button_egalisation_click(self):
        # Get the original image
        image = self.label_show_image_originale.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_ARGB32)

        # Convert the QImage to a numpy array
        image_array = rgb_view(image)
    
        # Convert the image to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
        # Equalize the histogram
        equalized_image = cv2.equalizeHist(gray)
    
        # Convert the equalized image to QImage
        equalized_qimage = self.convert_cv2_to_qimage(equalized_image)
    
        # Display the equalized image
        self.label_show_image_transformer.setPixmap(QPixmap.fromImage(equalized_qimage))
        self.label_show_image_transformer.setAlignment(Qt.AlignCenter)


    def on_button_Etirement_click(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        min_intensity = np.min(image)
        max_intensity = np.max(image)
    
        # Etirer l'intervalle dynamique de l'image sur toute la plage [0, 255]
        stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 255
        stretched_image = stretched_image.astype(np.uint8)
        q_image = self.convert_cv2_to_qimage(stretched_image)

        # Afficher l'image segmentée dans la QLabel centrale
        pixmap = QPixmap.fromImage(q_image)
        self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_Rectangle_click(self):
        pass
    def on_button_seuillage_click(self):
        if self.image_path:
        # Demander à l'utilisateur le nombre de classes pour K-Means
            seuil, ok = QInputDialog.getInt(self, "seuil", "Entrez le seuil", min=1)
        if ok :
            image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
            _, image_binarisee = cv2.threshold(image, seuil, 255, cv2.THRESH_BINARY)
            q_image = self.convert_cv2_to_qimage(image_binarisee)

            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_otsu_click(self):
        image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
        _, image_binarisee = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        q_image = self.convert_cv2_to_qimage(image_binarisee)
        pixmap = QPixmap.fromImage(q_image)
        self.label_show_image_transformer.setPixmap(pixmap)

    def on_button_moyenne_ponderee_click(self):
        image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
        image_binarisee = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        q_image = self.convert_cv2_to_qimage(image_binarisee)
        pixmap = QPixmap.fromImage(q_image)
        self.label_show_image_transformer.setPixmap(pixmap)

    def on_button_Gradient_click(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # Calculate the horizontal and vertical gradients with the Sobel operator
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the gradient magnitude
        magnitude_gradient = cv2.magnitude(gradient_x, gradient_y)
        # Apply thresholding to detect edges
        _, edges = cv2.threshold(magnitude_gradient, 50, 255, cv2.THRESH_BINARY)
    
        # Convert the edges array to a QImage
        q_image = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
    
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)
    
        # Display the image in label_show_image_transformer
        self.label_show_image_transformer.setPixmap(pixmap)
        self.label_show_image_transformer.setAlignment(Qt.AlignCenter)
        
        
    def on_button_Laplacien_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_Sobel_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_robert_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_dilatation_click(self):
        noyau, ok = QInputDialog.getInt(self, "le noyau de dilatation", "Entrez le noyau de dilatation", min=1)
        if ok :
            image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noyau, noyau))  # Définir le noyau de dilatation
            image_dilate = cv2.dilate(image, kernel, iterations=1)  # Appliquer la dilatation une fois
            q_image = self.convert_cv2_to_qimage(image_dilate)

            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_erosion_click(self):
        noyau, ok = QInputDialog.getInt(self, "le noyau d'érosion", "Entrez le noyau d'érosion", min=1)
        if ok :
            image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noyau, noyau))  # Définir le noyau d'érosion
            image_erode = cv2.erode(image, kernel, iterations=1)  # Appliquer l'érosion une fois
            q_image = self.convert_cv2_to_qimage(image_erode)

            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_ouverture_click(self):
        noyau, ok = QInputDialog.getInt(self, "Définir le noyau de l'ouverture", "Entrez le noyau de l'ouverture", min=1)
        if ok :
            image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
               # Appliquer l'ouverture
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (noyau, noyau))  # Définir le noyau de l'ouverture
            image_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
            q_image = self.convert_cv2_to_qimage(image_open)

            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_fermeture_click(self):
        noyau, ok = QInputDialog.getInt(self, "le noyau de la fermeture", "Entrez le noyau de la fermeture", min=1)
        if ok :
            image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (noyau, noyau))  # Définir le noyau de la fermeture
            image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
            q_image = self.convert_cv2_to_qimage(image_close)

            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_filtrage_morphologique_click(self):
        noyau, ok = QInputDialog.getInt(self, "le noyau de filtrage", "Entrez le noyau de filtrage", min=1)
        if ok :
            image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (noyau, noyau))  # Définir le noyau de l'ouverture
            image_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
        
            # Appliquer la fermeture
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (noyau, noyau))  # Définir le noyau de la fermeture
            image_close = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel_close)
            q_image = self.convert_cv2_to_qimage(image_close)
    
            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    def on_button_k_means_click(self):
        if self.image_path:
        # Demander à l'utilisateur le nombre de classes pour K-Means
            num_classes, ok = QInputDialog.getInt(self, "Nombre de classes", "Entrez le nombre de classes pour K-Means:", min=1)
        
        if ok:
            # Charger l'image
            image = cv2.imread(self.image_path)

            # Appliquer K-Means sur l'image
            segmented_image = self.apply_k_means(image, num_classes)

            # Convertir l'image segmentée en format QImage
            q_image = self.convert_cv2_to_qimage(segmented_image)

            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
    
    def on_button_partition_regions_d_click(self):
        if self.image_path:
        # Demander à l'utilisateur le nombre de classes pour K-Means
            threshold, ok = QInputDialog.getInt(self, "Nombre de classes", "Entrez le nombre de classes pour K-Means:", min=1)
        
        if ok:
            image = cv2.imread(self.image_path)
            # Convertir l'image en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            q_image = self.convert_cv2_to_qimage(binary_image)
            # Afficher l'image segmentée dans la QLabel centrale
            pixmap = QPixmap.fromImage(q_image)
            self.label_show_image_transformer.setPixmap(pixmap)
     

    def on_button_croissance_regions_d_click(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        segmented = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        seed = (100, 100)
        threshold = 20
        cv2.floodFill(segmented, None, seed, (255,255,255), (threshold,) * 3, (threshold,) * 3, cv2.FLOODFILL_FIXED_RANGE)
        q_image = self.convert_cv2_to_qimage(segmented)
        # Afficher l'image segmentée dans la QLabel centrale
        pixmap = QPixmap.fromImage(q_image)
        self.label_show_image_transformer.setPixmap(pixmap)


    def on_button_harris_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_shi_tomasi_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_hough_circles_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_hough_lines_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_huffman_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_ondelette_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass
    def on_button_lzw_click(self):
        # Ici vous pouvez écrire le code pour compresser une image et l'afficher dans self.image_transformed
        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
