import cv2
import face_recognition
import recognize_face
from pathlib import Path

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("data/videos/everyone.mov")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'MP42')
output_movie = cv2.VideoWriter('everyone.mp4', fourcc, 29.97, (1920, 1080))

known_path = Path("data/sample-2/jpeg/picked/known")
known_images = list(known_path.glob('*.jpeg'))

# Load some sample pictures and learn how to recognize them.
known_faces = [recognize_face.image_to_known_face(str(image_path), image_path.stem) for image_path in known_images]

# Initialize some variables
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    detected_faces = recognize_face.recognize(known_faces, rgb_frame)

    # Label the results
    for name, (top, right, bottom, left), distance in detected_faces:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        label = name + ' - ' + str("{0:.2f}".format(distance))
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
