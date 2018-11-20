import face_recognition
import collections
from PIL import Image
from pathlib import Path

KnownFace = collections.namedtuple('KnownFace', 'name encoding')
DetectedFace = collections.namedtuple('DetectedFace', 'name location distance')

def recognize(known_faces, image_to_recognize):
  '''
  input -> 
    known_faces - List<KnownFace>
    image_to_recognize - Image data / Frame
  return -> 
    List<DetectedFace>
  '''
  known_face_encodings = [each.encoding for each in known_faces]
  known_face_names = [each.name for each in known_faces]

  face_locations = face_recognition.face_locations(image_to_recognize)
  face_encodings = face_recognition.face_encodings(image_to_recognize, face_locations)
  
  detected_faces = []

  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    if True in matches:
      first_match_index = matches.index(True)
      name = known_face_names[first_match_index]
      distance = face_distances[first_match_index]
    else:
      name = "Unknown"
      distance = 0

    detected_faces.append(DetectedFace(name, (top, right, bottom, left), distance))

  return detected_faces


def image_to_known_face(image_path, name):
  '''
  input -> image_path as a string
  returns -> Face
  '''

  image = face_recognition.load_image_file(image_path)
  face_encodings = face_recognition.face_encodings(image)

  if len(face_encodings) == 1:
    return KnownFace(name, face_encodings[0])

  raise MULTIPLE_FACES_NOT_IMPLEMENTED


class MULTIPLE_FACES_NOT_IMPLEMENTED(Exception):
  pass