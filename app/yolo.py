from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame)
        return results
    
    def confidence(self, result, index=0):
        if not result.boxes:
            return None
        conf = result.boxes.conf
        return conf[index]
    
    def confidence_obb(self, result, index=0):
        if not result.obb:
            return None
        conf = result.obb.conf
        return conf[index]
    
    def find_center(self, result, index=0):
        if not result.boxes:
            return None
        xywh = result.boxes.xywh
        return [float(xywh[index][0]), float(xywh[index][1]), float(xywh[index][2]), float(xywh[index][3])]

    def find_center_obb(self, result, index=0):
        if not result.obb:
            return None
        xywhr = result.obb.xywhr
        return [float(xywhr[index][0]), float(xywhr[index][1]), float(xywhr[index][2]), float(xywhr[index][3]), float(xywhr[index][4])]

    def find_box(self, result, index=0):
        if not result.boxes:
            return None
        xyxy = result.boxes.xyxy
        return [float(xyxy[index][0]), float(xyxy[index][1]), float(xyxy[index][2]), float(xyxy[index][3])]

    def find_box_obb(self, result, index=0):
        if not result.obb:
            return None
        xyxyxyxy = result.obb.xyxyxyxy
        return [float(xyxyxyxy[index][0]), float(xyxyxyxy[index][1]), float(xyxyxyxy[index][2]), float(xyxyxyxy[index][3]),
                float(xyxyxyxy[index][4]), float(xyxyxyxy[index][5]), float(xyxyxyxy[index][6]), float(xyxyxyxy[index][7])]

if __name__ == "__main__":

    # yolo = YoloModel("runs/obb/train6/weights/best.pt")
    # image = "data/val/images/image1.jpg"    
    # results = yolo.predict(image)
    # for result in results:
    #     center = yolo.find_center_obb(result)
    #     box_points = yolo.find_box_obb(result)
    #     confidance = yolo.confidence_obb(result)
    #     print(f"Center: {center}")
    #     print(f"Box Points: {box_points}")
    #     print(f"Confidence: {confidance}")
    pass