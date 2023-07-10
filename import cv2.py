import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define a function to perform object detection
def perform_object_detection(image):
    # Check if the image is valid
    if image is None:
        raise ValueError("Invalid image")

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to Torch Tensor
    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()

    # Reshape the image tensor to include batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Run object detection on the image tensor
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the bounding boxes and labels from the predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter out objects based on confidence score (optional)
    confidence_threshold = 0.5
    filtered_boxes = boxes[predictions[0]['scores'].cpu().numpy() > confidence_threshold]
    filtered_labels = labels[predictions[0]['scores'].cpu().numpy() > confidence_threshold]

    # Return the filtered bounding boxes and labels
    return filtered_boxes, filtered_labels

# Load the image
image_path = "C:/Users/TANISHQ/Desktop/knife.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Failed to load image")

# Perform object detection on the image
detected_boxes, detected_labels = perform_object_detection(image)

# Initialize counters
harmful_objects_count = 0
harmless_objects_count = 0

# Process each detected object
for label in detected_labels:
    # Convert label index to class name if available
    class_names = ["knife", "gun", "scissors", "poison"]
    if label < len(class_names):
        class_name = class_names[label]
    else:
        class_name = f'Label {label}'

# Define a list of harmful objects for classification
harmful_objects = ["knife", "gun", "scissors", "poison"]


# Print the results
print("Harmful Objects Count:", harmful_objects_count,harmful_objects)
print("Harmless Objects Count:", harmless_objects_count)
