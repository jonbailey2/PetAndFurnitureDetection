import cv2
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F

# Define COCO classes (truncated for relevant classes: animals and furniture)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Define the class IDs for animals and furniture (specific to COCO)
#ANIMAL_CLASSES = {15, 16, 17, 18, 19}  # bird, cat, dog, horse, sheep
#FURNITURE_CLASSES = {56, 57, 59, 60}  # chair, couch, bed, dining table

# Load pre-trained SSD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssd300_vgg16(pretrained=True).to(device)
model.eval()

# Start video capture
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    img_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Perform detection
    with torch.no_grad():
        detections = model(img_tensor)[0]

    # Loop through detections
    for idx in range(len(detections["boxes"])):
        score = detections["scores"][idx].item()
        if score > 0.5:  # Confidence threshold
            class_id = int(detections["labels"][idx].item())
            #if class_id in ANIMAL_CLASSES.union(FURNITURE_CLASSES):
            label = COCO_CLASSES[class_id]
            box = detections["boxes"][idx].cpu().numpy().astype("int")

            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Add label
            cv2.putText(
                frame,
                f"{label}: {score:.2f}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the frame with detections
    cv2.imshow("Live Object Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
