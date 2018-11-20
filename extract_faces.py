import face_recognition
from pathlib import Path
from PIL import Image

path = Path("data")

files_to_process = path.glob('known_people/*.jpeg')

dest = path/'faces'

for image_file in files_to_process:
  image = face_recognition.load_image_file(str(image_file))
  face_locations = face_recognition.face_locations(image)

  for loc in face_locations:
    top, right, bottom, left = loc
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.save(str(dest/image_file.name))
    print('found %d face; wrote to %s' % (len(face_locations), str(dest/image_file.name)))
