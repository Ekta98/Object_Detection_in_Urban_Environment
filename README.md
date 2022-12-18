Object Detection in an Urban Environment
Data
For this project, we will be using data from the Waymo Open dataset.

We have already provided the data required to finish this project in the workspace, so you don't need to download it separately. However, if you are still interested in downloading the data locally, you can get it from the Google Cloud Bucket

Structure
Data
The data you will use for training, validation and testing is organized in the /home/workspace/data/ directory as follows:

train: contain the training data
val: contain the validation data
test - contains 10 files to test your model and create inference videos.
Experiments
The /home/workspace/experiments folder is organized as follow:

pretrained_model
reference - reference training with the unchanged config file
exporter_main_v2.py - to create an inference model
model_main_tf2.py - to launch training
experiment0 - create a new folder for each experiment you run
experiment1 - create a new folder for each experiment you run
label_map.pbtxt
Instructions
Step 1 - Exploratory Data Analysis (EDA)
You should use the data already present in /home/workspace/data/ directory to explore the dataset! This is the most important task of any machine learning project.

Implement the display_images function in the Exploratory Data Analysis notebook. This should be very similar to the function you created during the course. The output of this function should look like the image below:
Desired output of the `display_images` function
Desired output of the display_images function

Additional EDA: Feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.
You should refer to this analysis to create the different spits (training and validation).

Step 2 - Edit the config file
Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on config files. The config that we will use for this project is pipeline.config, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector here.

First, let's download the pretrained model and move it to /home/workspace/experiments/pretrained_model/. Follow the steps below:
cd /home/workspace/experiments/pretrained_model/

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
cd /home/workspace/
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
A new config file called pipeline_new.config will be created in the /home/workspace/ directory. Move this file to the /home/workspace/experiments/reference/ directory.
Step 3 - Model Training and Evaluation
Launch the training process:

a training process:
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
To monitor the training, you can launch a tensorboard instance by running python -m tensorboard.main --logdir experiments/reference/. You will report your findings in the writeup. The logs would look like the image shown below:
Training logs in Tensorboard
Training logs in Tensorboard

Once the training is finished, launch the evaluation process. Launching evaluation process in parallel with training process will lead to OOM error in the workspace.

an evaluation process:
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
By default, the evaluation script runs for only one epoch. Therefore, the eval logs in Tensorboard will look like a blue dot.

Note: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using CTRL+C.

Step 4 - Improve the performances
Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model.

One obvious change consists in improving the data augmentation strategy. The preprocessor.proto file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: Explore augmentations.ipynb. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:

experiment with the optimizer: type of optimizer, learning rate, scheduler etc
experiment with the architecture. The Tf Object Detection API model zoo offers many architectures. Keep in mind that the pipeline.config file is unique for each architecture and you will have to edit it.
Important: If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the tf.events files located in the train and eval folder of your experiments. You can also keep the saved_model folder to create your videos.

Creating an animation
Export the trained model
Modify the arguments of the following function to adjust it to your models:

python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
This should create a new folder experiments/reference/exported/saved_model. You can read more about the Tensorflow SavedModel format here.

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):

python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
