
import time
import sys
sys.path.append("../Computer Vision")
from belt_objects import *
# Going to need to tune these
ms_per_100pixel = 100
similarity_threshold = 30 #value is in frames
width = 1020 # pixels (I dont think this number is 
            # accurate but it doesnt actually matter for tracking position)
            # It just changes how accurate the predicitions is for what frame
            # the object will reach the end. Just needs to be greater than the actual number
belt_speed = 11 #pixels per frame

def trackObjects(objects, img_bgr, yolo, class_aliases, frame_number):
    centroids = detect_objects_in_frame(img_bgr, yolo, class_aliases)
    
    for centroid in centroids:
        cx, cy = centroid["centroid"]
        new_object = True
        predicted_arrival = frame_number + ((width - cx) / belt_speed)

        for i, obj in enumerate(objects):
            obj_predicted_arrival = objects[i]["predicted_arrival"]

            if abs(predicted_arrival - obj_predicted_arrival) < similarity_threshold:
                # print(f"difference: {predicted_arrival:.2f}-{obj_predicted_arrival:.2f}={(predicted_arrival-obj_predicted_arrival):.2f}")
                if centroid["confidence"] >= obj["confidence"]:
                    objects[i] = centroid
                    objects[i]["frame"] = frame_number
                    objects[i]["predicted_arrival"] = predicted_arrival
                else:
                    # updatinbg centroid and averaging prediction times
                    objects[i]["centroid"] = (cx, cy)
                    objects[i]["predicted_arrival"] = (predicted_arrival + obj_predicted_arrival) / 2
                new_object = False
                break

        if new_object:
            print(f"adding object '{centroid["class"]}' at frame {frame_number}")
            centroid["frame"] = frame_number
            centroid["predicted_arrival"] = predicted_arrival
            objects.append(centroid)
            return True

def loadModel():
    yolo = load_yolo(Path("../Computer Vision/trials/trial5-manual-auto/weights/best.pt"))
    class_aliases = parse_class_aliases(
        "bottle",
        "can",
        "6-pack,six-pack,six_pack,6pack",
    )
    return yolo, class_aliases

if __name__ == "__main__":
    yolo, class_aliases = loadModel()
    objects = []
    video_path = "./Computer Vision/BlankVideo.mov"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
    simulated_fram_rate = 20
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        added = trackObjects(objects, frame, yolo, class_aliases, frame_number)
        if added: 
            user_in = input(f"===== at frame {frame_number} =====")
            if "objects" in user_in:
                for object in objects:
                    print(f"type: '{object["class"]}', centroid: '{object["centroid"]}', added on frame '{object["frame"]}' predicted to arrive at fram '{object["predicted_arrival"]}'")
            if "show" in user_in: 
                cv2.imshow(f"frame {frame_number}", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        for i in range(len(objects) - 1, -1, -1):
            if objects[i]["predicted_arrival"] < frame_number:
                print(f"removing '{objects[i]['class']}' at frame: {frame_number}")
                objects.pop(i)
        frame_number += 1
        time.sleep(1/simulated_fram_rate)