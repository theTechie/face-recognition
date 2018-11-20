import face_recognition
from PIL import Image, ImageDraw
from pathlib import Path
import recognize_face

know_path = Path("data/known_people")
known_images = list(know_path.glob('*.jpeg'))

known_face_encodings = []
known_face_names = []

known_faces = [recognize_face.image_to_known_face(str(image_path), image_path.name) for image_path in known_images]

print('I just learned to recognize %d persons... \n' % len(known_images))

unknown_path = Path("data/train")
unknown_images = list(unknown_path.glob('**/*.jpeg'))

print('I am starting to identify %d unknown persons; lets see how many i know !! \n' % len(unknown_images))

output_path = Path("data/output")

for image_to_identify in unknown_images:
    unknown_image = face_recognition.load_image_file(str(image_to_identify))
    # face_locations = face_recognition.face_locations(unknown_image)
    # face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    detected_faces = recognize_face.recognize(known_faces, unknown_image)
    
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for name, (top, right, bottom, left), distance in detected_faces:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        label = name + ' ' + distance
        text_width, text_height = draw.textsize(label)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255))

    # Display the resulting image
    # pil_image.show()

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # You can also save a copy of the new image to disk if you want by uncommenting this line
    pil_image.save(output_path/image_to_identify.name)
