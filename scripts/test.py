import pathlib
from ultralytics import YOLO
import os
if __name__ == '__main__':

    dossier_images = "D:/work/yolotex/vos_images_de_test"
    images = [os.path.join(dossier_images, fichier).replace("\\", "/") for fichier in os.listdir(dossier_images) ]
    model =YOLO("D:/yolo/mlruns/287129418118735565/025d74a0844d43fa8c298c4ee2154c2e/artifacts/weights/best.pt")
    results = model.predict(images[22])
    
    for result in results:
        result.show()



    