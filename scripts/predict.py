import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
#v7 es malo
# model = YOLO('models/v13.pt')
model = YOLO('models/final.pt')
# model = YOLO('models/datasetN.pt')
# model = YOLO('models/datasetS.pt')

# Open the video file
video = "videos/2024 Field Tour Video_ Source.mp4"
# video = "videos/2024 Field Tour Video_ Speaker.mp4"
cap = cv2.VideoCapture(video)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # rename to note due to wrong labeling
        # for result in results:
        #     result.names[1] = "note"

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()