import sys
import os
from modules.utils.config import Config
from modules.yolov10 import YoloV10


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    config_file = "D:/yolo/modules/yolov10/config.json"
    config =  Config(config_name=config_file)
    yolo = YoloV10(config=config)
    yolo.fit()