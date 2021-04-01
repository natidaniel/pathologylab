# PathologyLab
## Description
This platform was created to provide a general and simple framework for pathological uses.
Our net (PDL1Net) deals with segmantation and counting of PD-L1 cells in WSIs of lung cancer patients, but can use for other implementations as well.
For easy orientation in this github please take a look at the following explanation about the building blocks.
## Setup
In order to run PDL1_main.py few requirements are needed. The code ran and been tested on python 3.5, and used TF 1.11.0
and keras 2.1.5. For your convenient a requirement.txt file containing all the modules in our environment can be found under
the project main folder. Use the next command to install all the modules to your python environment:
```commandline
pip install -r requirements.txt
``` 
Important notes:
* The environment was carefully obtained, using a lot of time on trial and error. It is highly unstable
and every minor change can break it.
* As a result we highly recommend you to start a clean environment and install only the `requirements.txt` file from the main
folder.
 
## Data
Creation of the data: For training and evaluation, our network's inputs are segmented patches of about 800×800 pixels
from IHC slides. Due to our cooperation with RAMBAM hospital, the raw data was given as real patients' scans.
<br />
the raw data consists 254 segmented and labeled patches
### Data for Sessions
Both train session and test session expect the next folder structure:
```
<root-folder>
└-- val
|    | <image 1>
|    |     :
|    | <image n>
|    | via_export_json.json
|
└-- train
     | <image 1>
     |     :
     | <image m>
     | via_export_json.json

```
The code expects the json files in `val` and `train` to be formatted the way "VIA tool" exports json (only the "_via_img_metadata" attribute of the exported json).
<br />
Each json file has to contain only `polygon` region shapes and not other variates (like `point` or `triangle` etc.).
<br />
`PDL1NetDataLoader` is the class which is responsible to read the json file and load the images.
It expects to find the json file contains the annotation for the images in the folder, like shown above.
<br />
The labels have priority-order to insure that when collision is made, the higher priority label will prevail.
Priority is determine by the index position in the class list, the higher the index, the higher the priority.
<br />
In our data the classes are listed as: 1 - inflammation, 2 - PDL1-negative, 3 - PDL1-positive, 4 - other, 5 - air.
<br />
the inflammation and air classes are not used in the net. the `PDL1NetDataLoader` transforms 'inflammation' into 'other',
and air class was created automatically by the code (by marking white pixels), but it seems that the use of it reduced the accuracy of the net.

## Usage
To initiate a train session use the next command in the command line:
```commandline
PDL1_main.py train --datatset path\to\root\folder --weights path\to\weight\file [optional --augment]
```
To initiate a test session use the next command in the command line:
```commandline
PDL1_main.py test --datatset path\to\root\folder --weights path\to\weight\file [optional --result_dir path\to\result\dir]
```
**The Flags in Details:**
* *train* \ *test* - choose the session type to start
* *--datatset* path to the dataset folder that holds `val` & `train` subfolders. 
For more details on the data look at **Data for Sessions** 
* *--weights* few options are available:
    * "coco" - loads a weights that were trained on the coco data set
    * "last" - loads the last `.h5` file from the `logs` folder
    * exact path to `.h5` file that holds the weights
* *--augment* (optional) if this flag is being used the model will augment the 
data using various transformations (only available on train sessions).
* *--result_dir* (optional) save the results of the session to this folder.
if the folder contains unfinished session (the network sometimes crashes because of memory errors) - 
it continues the existing session (only available on test sessions).

**Example on Savir-Lab computer:** 
```commandline
PDL1_main.py test --dataset D:\Nati\Itamar_n_Dekel\data\ --weights D:\Nati\Itamar_n_Dekel\pathologylab\logs\pdl120210329T1425_aug_img1-256_medium_anchor_rpn-th-06_min-confidence-06_detect-th-06\mask_rcnn_pdl1_0105.h5
```
```commandline
PDL1_main.py train --dataset D:\Nati\Itamar_n_Dekel\data\ --weights coco --augment
```

## Train and Test Sequences
each ground-truth image in the relevant json (train or val) modified by pre-proccesing which removes all the air pixels from it's classes maskings 
(removing each white pixel from the class mask and converting it to 'other').
<br />
than the image is going trough the net.
<br />
lastly, air is filtered from the prediction of the net's masking from each class's mask.
<br />
in the test sequence, after passing all the images in the net, all the metrics are calculated, 
and the cell counting algorithm is run on each image's prediction in order to calculate the WSI's score by cell number.

## Test Output
The output of the test sequence consists:
<br />
* confusion matrix for the classes classification
* confusion matrix for the classesc lassification after air-filtering
* output file with additional metrics, like IoU of the prediction (before and after air-filtering), 
the WSI score (by area and by cell count), the WSI score of the ground-truth (by area and by cell count)
<br />
and for each image patch -
* image of the boxes of the rois the net found. the roi's confidence is written above each box 
* image of the masking of the pdl-negative and pdl-positive the net found
* the ground-truth masking of the image

## Helpful Scripts
in the scripts folder you can find some helpful script for data ordering and data pre-processing (like gamma correction of the images).
<br />
for example, there is a script which calculates the area covered by pdl1-positive and pdl1-negative for each image, in order to check the data balancing.
this script also helps to split the data into train and test jsons.

## Configuration
The configuration is a class that controls the meta parameters of the model - for example, the number of classes,
the backbone of the network, number of iterations per epoch, etc.
The right way to use the configuration is to create a class of your own, that derives `Config` class
(found in `algo\mrcnn\config`). In the new class, change only the parameter that you want to change,
the parameter you don't overload will be derived from the `Config` class. 
## Synthetic Data Generator Tool
A way to evaluate the goodness of the network without the limitations from small segmented dataset
is to use synthetic data. The script `tools\create_synth_data\main_synth.py` is used to create
synthetic data with 4 or less classes.  
The script produces `.json` file in COCO format. This format can then be loaded to `VIA tool` and exported
to `.json`, that way the file will be in the format that `PDL1NetDataLoader` expects. 
The configuration class in `main_synth.py` controls all the input and output parameters:
input folder path, output folder path, classes labels, number of images to generate,
and also shapes' color, background images etc.  
This tool can run only on **linux** because it needs `pycococreator` module, that currently
 supports only linux. To create the environment for the Synthetic Generator you can use `requirements_synth_generetor.txt`
 file with `pip install -r` as seen in the *Setup* section.

## License
Copyright 2020 Nati Daniel & Itamar Gruber & Shai Nahum Gefen & Itamar Goldman & Dekel Meirom

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
