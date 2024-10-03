import os
import sys
from modules.yolov10.test import PDFObjectDetector


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__=="__main__":

    pdf_folder = "D:/work/yolotex/les_pdf"  
    output_folder = "D:\work\yolotex\sortie"  
    yolo_model_path = "D:/work/yolotex/mlruns/287129418118735565/025d74a0844d43fa8c298c4ee2154c2e/artifacts/weights/best.pt"  

    test = PDFObjectDetector(yolo_model_path, output_folder)
    test.process_folder(pdf_folder)


