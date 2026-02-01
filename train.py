import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'./ultralytics/cfg/models/11/yolo11-Fuzzy_Conv.yaml')
    model.train(data=r"./ultralytics/data8/data.yaml",

                task='detect',
                cache=False,
                imgsz=640,
                epochs=500,
                single_cls=False,
                batch=32,
                close_mosaic=0,
                workers=5,
                device='0',
                optimizer='SGD',
                # resume=,
                amp=True,
                project='runs/train',
                name='exp',
                )

