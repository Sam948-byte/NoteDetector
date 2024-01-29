from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('/home/sam/gitClones/NoteDetector/best.pt')

# Define path to video file
source = '/home/sam/gitClones/NoteDetector/videos/2024 Field Tour Video_ Source.mp4'

# Run inference on the source
results = model(source, stream=True, show=True)  # generator of Results objects