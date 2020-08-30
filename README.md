# Object-Detection with pretrained Inception_v2 and Mobilenet model

Problem Description - Detect moving objects at a railway crossover and classify them as human (pedestrian, cyclist or vehicle) and not human. Accordingly trigger/adjust audible alarms (increase volume if human is present, decrease otherwise) at the railway crossover.

**Steps for execution of Object Detection**
1. Split Video into frames/images. 
The video_to_image.py file helps to achieve this with a framerate of 0.01 (image captured every 0.01sec of the video)

2. Object Detection and Mapping
I have used the inception-v2 a pretrained model for object detection. The object_detection.py code has functions to detect objects in a frame/image and create an XML file corresponding to each image/frame which contains the class number, class name and dimensions of the detection box around each object in the frame.

3. XML to CSV
The code xml_to_csv.py code creates a CSV file each for the train and test data present in the train and test data respectively.

4. TfRecords creation
The code generate_tf_record.py creates a tfrecord file each for the train and test data present in the train and test data respectively.

5. Model Training
To train the model, the tfrecords created in the previous step have to be used. Below is a brief layput of the various files required in the different directories before you can execute the model_main.py file.

![layout image](layout.png)

