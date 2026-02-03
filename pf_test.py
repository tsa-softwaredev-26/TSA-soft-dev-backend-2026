from ultralytics import YOLO
from time import sleep 

model = YOLO("yoloe-11l-seg-pf.pt")
results = model.predict(source="desk.heic", conf = .6, iou = .45, show = True, save=True, show_conf=True)
sleep(20)
for r in results:
    boxes = r.boxes.xyxy.cpu().tolist()
    cls = r.boxes.cls.cpu().tolist()
    # for b, c in zip(boxes, cls):
    #     print(f"box: {b}, cls: {model.names[c]}")
    print(f"detected {len(boxes)} objects")

