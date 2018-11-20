import face_recognition
from pathlib import Path
from PIL import Image

path = Path("data/sample-2/jpeg/picked/known-full")

files_to_process = path.glob('*.jpeg')

dest = Path('data/sample-2/jpeg/picked/known')

for image_file in files_to_process:
  image = face_recognition.load_image_file(str(image_file))
  face_locations = face_recognition.face_locations(image)

  i = 0
  for loc in face_locations:
    top, right, bottom, left = loc
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    filename = str(i) + '-' + image_file.name
    pil_image.save(dest/filename)
    print('found %d face; wrote to %s' % (len(face_locations), str(dest/filename)))
    i = i + 1
