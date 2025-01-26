import torch
import torchvision
import cv2

# Load pre-trained SSD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.ssd300_vgg16(pretrained = True)
model.eval()

# Loading class names from external classes.txt file
classnames = []
with open('../classes.txt', 'r') as f:
    classnames = f.read().splitlines()

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
    img_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).float() / 255.0  # [H, W, C] -> [C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension

    #img_tensor = F.to_tensor(frame).to(device)

    # Perform detection
    with torch.no_grad():
        detections = model(img_tensor)[0]

    # Loop through detections
    for idx in range(len(detections["boxes"])):
        score = detections["scores"][idx].item()
        if score > 0.5:  # Confidence threshold
            class_id = int(detections["labels"][idx].item())
            # if class_id in ANIMAL_CLASSES.union(FURNITURE_CLASSES):
            label = classnames[class_id]
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
