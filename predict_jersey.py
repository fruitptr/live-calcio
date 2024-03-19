from IPython import display as dp
from ultralytics import YOLO
from IPython.display import display, Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from glob import glob
from random import sample
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter
from pathlib import Path
import os
import shutil
import numpy as np
import cv2
import random
import torch
from tensorflow.keras.preprocessing.image import array_to_img

def get_last_frame(video_url):
  cap = cv2.VideoCapture(video_url)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
  ret, frame = cap.read()
  cap.release()

  cv2.imwrite('last_frame.png', frame)

  return frame

def load_person_dect_model(video_url):
  model = YOLO('yolov8m.pt')
  results = model.track(source=video_url, conf=0.25, save=True, tracker='bytetrack.yaml', classes=0)
  return results

def get_tapped_box(results, tap_coord):
  last_frame_index = len(results) - 1
  last_frame_boxes = results[last_frame_index].boxes.xyxy.tolist()
  required_track_id = None

  for index, box in enumerate(last_frame_boxes):
      x_min, y_min, x_max, y_max = box
      if x_min <= tap_coord[0] <= x_max and y_min <= tap_coord[1] <= y_max:
          required_track_id = results[last_frame_index].boxes[index].id.tolist()[0]
          print("Tracking ID: ", required_track_id)
          return required_track_id

  return None

def get_jersey_color(last_frame_image, tap_coord):
    cropped_img = None
    player_found = None

    model = YOLO('yolov8m.pt')
    frame_result = model.predict(source='last_frame.png', conf=0.25, save=True, classes=0)
    last_frame_boxes = frame_result[0].boxes.xyxy.tolist()

    for index, box in enumerate(last_frame_boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        if x_min <= tap_coord[0] <= x_max and y_min <= tap_coord[1] <= y_max:
            cropped_img = last_frame_image[y_min:y_min + (y_max - y_min) // 2, x_min:x_max].copy()
            cv2.imwrite('cropped_frame.png', cropped_img)
            player_found = True
            break

    if player_found is False:
        return "Not a player"

    hsv_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([155, 25, 0])
    upper_red = np.array([179, 255, 255])

    mask_red = cv2.inRange(hsv_cropped_img, lower_red, upper_red)
    cv2.imwrite('mask_red.png', mask_red)

    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    total_pixels = cv2.countNonZero(cropped_img_gray)
    red_pixels = cv2.countNonZero(mask_red)
    red_percentage = (red_pixels / total_pixels) * 100

    red_threshold = 15
    if red_percentage >= red_threshold:
        return True
    else:
        return False

def load_jersey_dect_model():
  jersey_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5ID_640_100epoch.pt')
  jersey_model.load_state_dict(torch.load('yolov5ID_640_100epoch.pt')['model'].state_dict(), strict=False)
  return jersey_model

def clear_output_folder():
  output_folder = 'cropped_images'

  if os.path.exists(output_folder):
      shutil.rmtree(output_folder)
  os.makedirs(output_folder)

def pre_process_images(results, required_track_id):
  image_paths = []
  output_folder = 'cropped_images'

  for result_index, result in enumerate(results):
      for box_index, box in enumerate(result.boxes):
          if box.id is not None:
            current_track_id = box.id.tolist()[0]

          if current_track_id == required_track_id:
              x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
              orig_img_bgr = result.orig_img

              orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

              orig_img = Image.fromarray(orig_img_rgb)

              cropped_img = orig_img.crop((x_min, y_min, x_max, y_max))

              enlarged_size = (cropped_img.width * 2, cropped_img.height * 2)
              cropped_img = cropped_img.resize(enlarged_size, Image.BICUBIC)

              cropped_img_sharpened = cropped_img.filter(ImageFilter.SHARPEN)
              cropped_img_sharpened = cropped_img_sharpened.filter(ImageFilter.SHARPEN)

              image_filename = f'cropped_image_{result_index}_box_{box_index}_sharpened.png'
              image_path = os.path.join(output_folder, image_filename)
              cropped_img_sharpened.save(image_path)

              image_paths.append(image_path)

  return image_paths

def predict_number(img, model):
  im = array_to_img(img)
  output = model(im)
  results = output.pandas().xyxy[0].to_dict(orient="records")
  for result in results:
      cs = result['class']
      return cs

def get_jersey_number(image_paths, model):
  num_dict = {}

  for path in image_paths:
    img = cv2.imread(path)
    num = predict_number(img, model)
    if num is not None:
      if num not in num_dict:
        num_dict[num] = 1
      else:
        num_dict[num] += 1

  if bool(num_dict) == False:
    return None

  print(num_dict)

  return max(num_dict, key=num_dict.get)


def predict_jersey(video_url, tap_coords):
  dp.clear_output()
  last_frame_image = get_last_frame(video_url)
  if get_jersey_color(last_frame_image, tap_coords):
    results = load_person_dect_model(video_url)
    bbox_id = get_tapped_box(results, tap_coords)
    if bbox_id == None:
      return "Not a player"
    else:
      model = load_jersey_dect_model()
      clear_output_folder()
      image_paths = pre_process_images(results, bbox_id)
      jersey_number = get_jersey_number(image_paths, model)
      if jersey_number == None:
        return "Cannot identify"
      else:
        return jersey_number
  else:
    return "Not a player (Jersey color is not red)"

# if __name__ == "__main__":
#   video_url = 'unitedvseverton1080p.mp4'
#   tap_coords = ((469, 517))
#   number = predict_jersey(video_url, tap_coords)
#   print("Jersey number is: ", number)