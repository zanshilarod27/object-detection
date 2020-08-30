#tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import os
from bs4 import BeautifulSoup as bs
import pandas as pd

# Print Tensorflow version
# print(tf.__version__)

tf.enable_eager_execution()
CLASSES = {'Person': 1}

#definition to create the xmls
def create_XML_CSV(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    frameNum = 30
    xmls_path = "path to save xml files" + 'frame' + str(frameNum) + '.xml'
    image_path = "path from which images are loaded" + 'frame' + str(frameNum) + '.jpg'
    xml_list = []
    xml_soup = bs(features='lxml')
    xml_soup.append(xml_soup.new_tag("annotation"))
    dir_name, file_name = os.path.split(image_path)  # [0],os.path.split(xmls_path)[1]
    # folder
    new_tag = xml_soup.new_tag("folder")
    xml_soup.annotation.append(new_tag)
    xml_soup.annotation.folder.string = dir_name
    # filename
    new_tag = xml_soup.new_tag("filename")
    xml_soup.annotation.append(new_tag)
    xml_soup.annotation.filename.string = file_name
    # path
    file_path = image_path
    new_tag = xml_soup.new_tag("path")
    xml_soup.annotation.append(new_tag)
    xml_soup.annotation.path.string = file_path
    # source
    new_tag = xml_soup.new_tag("source")
    xml_soup.annotation.append(new_tag)
    # database
    new_tag = xml_soup.new_tag("database")
    xml_soup.annotation.source.append(new_tag)
    xml_soup.annotation.source.database.string = "Unknown"
    # size - of frame
    new_tag = xml_soup.new_tag("size")
    xml_soup.annotation.append(new_tag)
    image_depth = 3
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    im_width, im_height = image_pil.size
    # width
    new_tag = xml_soup.new_tag("width")
    xml_soup.annotation.size.append(new_tag)
    xml_soup.annotation.size.width.string = str(im_width)
    # height
    new_tag = xml_soup.new_tag("height")
    xml_soup.annotation.size.append(new_tag)
    xml_soup.annotation.size.height.string = str(im_height)
    # depth
    new_tag = xml_soup.new_tag("depth")
    xml_soup.annotation.size.append(new_tag)
    xml_soup.annotation.size.depth.string = str(image_depth)
    # segmented
    new_tag = xml_soup.new_tag("segmented")
    xml_soup.annotation.append(new_tag)
    xml_soup.annotation.segmented.string = '0'

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            class_name1 = class_names[i].decode("ascii")
            left = int(xmin * im_width)
            right = int(xmax * im_width)
            top = int(ymin * im_height)
            bottom = int(ymax * im_height)
            object_tag = xml_soup.new_tag("object")

        # name
            new_tag = xml_soup.new_tag("name")
            object_tag.append(new_tag)  # xml_soup.annotation.
            object_tag.find("name").string = class_name1  # name is reserved ib BS   #xml_soup.annotation.
        if not (class_name1 in CLASSES.keys()):
            # largest key +1
            number_of_classes = len(CLASSES.keys())
            classes = number_of_classes + 1
            CLASSES.update({class_name1: classes})  # add new class
        else:
            classes = CLASSES.get(class_name1)
        ###### if not name in CLASSES: update CLASSES
        # pose
        new_tag = xml_soup.new_tag("pose")
        object_tag.append(new_tag)
        object_tag.pose.string = "Unspecified"
        # truncated
        new_tag = xml_soup.new_tag("truncated")
        object_tag.append(new_tag)
        object_tag.truncated.string = '0'
        # difficult
        new_tag = xml_soup.new_tag("difficult")
        object_tag.append(new_tag)
        object_tag.difficult.string = '0'
        # bndbox
        box_tag = xml_soup.new_tag("bndbox")
        # xmin
        new_tag = xml_soup.new_tag("xmin")
        box_tag.append(new_tag)
        box_tag.xmin.string = str(left)
        # ymin
        new_tag = xml_soup.new_tag("ymin")
        box_tag.append(new_tag)  # xml_soup.annotation.object.bndbox.
        box_tag.ymin.string = str(top)  # xml_soup.annotation.object.bndbox.
        # xmax
        new_tag = xml_soup.new_tag("xmax")
        box_tag.append(new_tag)  # xml_soup.annotation.object.bndbox.
        box_tag.xmax.string = str(right)  # xml_soup.annotation.object.bndbox.
        # ymax
        new_tag = xml_soup.new_tag("ymax")
        box_tag.append(new_tag)  # xml_soup.annotation.object.bndbox.
        box_tag.ymax.string = str(bottom)  # xml_soup.annotation.object.bndbox.

        object_tag.append(box_tag)
        xml_soup.annotation.append(object_tag)

        value = (classes, class_name1, file_name, file_path, im_height, im_width, image_depth, left, right, top, bottom)
        xml_list.append(value)

        column_name = ['classes', 'classname', 'filename', 'filepath', 'height', 'width', 'depth', 'xmin', 'xmax', 'ymin',
                   'ymax']
        xml_pd = pd.DataFrame(xml_list, columns=column_name)

        with open(xmls_path, "w") as f:  # make xml file
            f.write(xml_soup.prettify())

    return xml_pd

#definition for displaying a colored image
def display_image(image,i):
    img = Image.fromarray(image, 'RGB')
    img.save('path to save the image' + i)

filename = 'C:/Users/zansh/OneDrive/Desktop/Dissertation/try2/Data/Animals/'

#definition to resize image
def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    for i in os.listdir(url):
        pil_image = Image.open(url + i)
        pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
        pil_image_rgb = pil_image.convert("RGB")
        pil_image_rgb.save(filename + i, format="JPEG", quality=90)
    return filename

#definition to draw a bounding boxes on image
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    print('image width, height', image.size)
    print('ymin, xmin, ymax, xmax: ', ymin, xmin, ymax, xmax)
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    print('left, right, top, bottom: ', left, right, top, bottom)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin

#definition to draw boxes around objects with class names and class numbers to the top
def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


image_path = 'load images from this path'
downloaded_image_path = download_and_resize_image(image_path, 1280, 856, True)

xmls_path = "path to save xmls"

#downloading the inception_resnet_v2 for object detection
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']


def load_img(path, i):
    img = tf.io.read_file(path + i)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path):
    for i in os.listdir(path):
        img = load_img(path, i)
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)
        #print(result) to understand the class number, class name and detection box dimension matrices
        result = {key: value.np() for key, value in result.items()}
        image_with_boxes = draw_boxes(img.numpy(), result["detection_boxes"],result["detection_class_entities"], result["detection_scores"])
        display_image(image_with_boxes, i)

#function call
run_detector(detector, downloaded_image_path)