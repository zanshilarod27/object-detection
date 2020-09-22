# Object-Detection with pretrained Inception_v2 and Mobilenet Neural Nets

**Problem Description** - Detect moving objects at a railway crossover and classify them as human (pedestrian, cyclist or vehicle) and not human to help improve safety.

**Steps for execution of Object Detection**

**1. Split Video into frames/images.**

The video output from the CCTV camera is split into images/frames using the OpenCV library of Python. These images are fed to the Inception Convolutional Neural Network to create annotations for objects detected in an image.The *video_to_image.py* file helps to achieve this with a framerate of 0.01 (image captured every 0.01sec of the video).

<p align="center">
    <img src="images readme/frame2051.jpg" alt="Image" width="500" height="250" />
</p>

**2. Object Detection and Mapping**

I have used the inception-v2 a pretrained model to create annotations for the objects deetcted ina an image. The *object_detection.py* code has functions to detect objects in a frame/image and create an XML file corresponding to each image/frame containing the class number, class name and dimensions of the detection boxes around each object in the frame. Below is an XML file for an image. Each <object> has specifications of the object detected by the model.

<p align="center">
    <img src="images readme/xml1206.jpg" alt="xml for image" width="400" height="600" />
</p>


Now, Create folders test and train in images as shown in the layout below. Copy 85% of the images into train and 15% into test folders.Past the XML files into the corresponding train and test folder.


<p align="center">
    <img src="images readme/ggif2po.gif" alt="obj-detn for image" width="500" height="250" />
</p>

**3. XML to CSV**

An XML file corresponding to each image/frame containing the class number, class name, and dimensions of the detection boxes around each object in the frame is created. The code *xml_to_csv.py* code creates a CSV file each for the train and test data present in the train and test data respectively.

**4. TfRecords creation**

The code *generate_tf_record.py* creates a tfrecord file each for the train and test data present in the train and test data respectively. Copy these tfrecords into the data folder along with a PBTXT which will contain mapping of all the class numbers with the class names. (check the contents of this file in my repo)

**5. Model Training**

Below is a brief layput of the various files required in the different directories before you can execute the *model_main.py* file.
<p align="center">
    <img src="images readme/layout.png" alt="Image" width="1000" height="150" />
</p>

To start training the model, you now need the model and its corresponding config file. Reduced network size and faster performance made the Mobilenet CNN a great choice to perform the training. You can pickup a model and its config from the official tensorflow/models directory in github.

<ins>Paramter changes in config file</ins>

num_classes: 8 #number of classes to detect while training
batch_size: 8 #based on your requirements
fine_tune_checkpoint: "path to the model folder/ssd_mobilenet_v1_coco/model.ckpt"
input_path: "path to the training tfrecord /train/test_label.record" -- in the train_input_reader and eval_input_reader
label_map_path: "path to the pbtxt mapping file /objectdetection.pbtxt" -- in the train_input_reader and eval_input_reader


Execute the model_main.py to start training. Launch TensorBoard to view the performance of training your model through Scalars and Images.
<p align="center">
    <img src="images readme/individualImage.png" alt="Image" width="1000" height="250" />
</p>
 
**6. Testing your Model**
Execute the *object_detection_tutorial.ipynb* file to perform testing on images that have not been used to train the model.

<p align="center">
    <img src="images readme/of.gif" alt="prediction1" width="500" height="250" />
</p>

<p align="center">
    <img src="images readme/of1.gif" alt="prediction2" width="500" height="250" />
</p>
 
 You will observe that in the figure above the person in the upper half of the frame is not predicted. There could be many reasons to this like darkness, small dataset etc. A detailed analysis of this will be uploaded as I am working to improve this model.
 
**Good Luck! You will come across 100's of errors but do not give up. Take a deep breath, sip your coffee and move on. ;)**
**You have Stackoverflow to your rescue.**
