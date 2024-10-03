import os
from pdf2image import convert_from_path
from ultralytics import YOLO
from PIL import Image


class PDFObjectDetector:
    """_summary_ 
    Cette classe permet de tester votre modèlevyolo sur de nouveaux documents
    
    il suffit de spécifier le path du dossier contenant les fichiers pdf, le path du dossier de sortie et le chemin d'accès aux meilleurs poids du modèle (dossier mlruns)
    
    """
    
    def __init__(self, yolo_model_path, output_folder):
       
        self.model = YOLO(yolo_model_path)
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def create_class_folders(self, classes):
        """
        Crée des dossiers pour chaque classe détectée si nécessaire.
        """
        for class_name in classes:
            class_folder = os.path.join(self.output_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

    def save_cropped_image(self, image, bbox, output_path):
        """
        Découpe et enregistre une image selon les coordonnées de la bounding box.
        """
        bbox = [int(coord) for coord in bbox]
        cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        cropped_image.save(output_path)

    def process_page(self, page_image, pdf_name, page_number):
        """
        Traite une seule page en effectuant une prédiction YOLO et en enregistrant les images découpées.
        """
        results = self.model.predict(page_image)

       
        for result in results:
            for i, box in enumerate(result.boxes):
                bbox = box.xyxy[0].tolist()  
                class_id = int(box.cls.item())  
                class_name = self.model.names[class_id]  

                class_output_folder = os.path.join(self.output_folder, class_name)
                if not os.path.exists(class_output_folder):
                    os.makedirs(class_output_folder)

                output_image_name = f"image{page_number}_page_{pdf_name}_{i}.png"
                output_image_path = os.path.join(class_output_folder, output_image_name)

                self.save_cropped_image(page_image, bbox, output_image_path)

    def process_pdf(self, pdf_path):
        """
        Traite un fichier PDF en analysant chaque page avec YOLO et en sauvegardant les résultats.
        """
        pdf_name = os.path.basename(pdf_path).split('.')[0]
        pages = convert_from_path(pdf_path)

        for page_number, page_image in enumerate(pages, start=1):
            self.process_page(page_image, pdf_name, page_number)
            print(f"Page {page_number} du fichier {pdf_name} traitée.")

    def process_folder(self, pdf_folder):
        """
        Traite un dossier contenant plusieurs fichiers PDF.
        """
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Traitement du fichier : {pdf_file}")
            self.process_pdf(pdf_path)
        
        print("Traitement terminé pour tous les fichiers PDF.")


