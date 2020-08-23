# PathologyLab
## Description
this platform was created to provide a general and simple framework for pathological uses.
Our net (PDL1Net) deals with segmantation and counting of PD-L1 cells in WSIs of lung cancer patients, but can use for other implementations as well.
for easy orientation in this github please take a look at the following explanation about the building blocks.
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
The net uses Whole Slide Images (WSI) IHC stained. The patch size for the training step was **SIZE** and it was curated by pathologist.
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
The code expects the json file in `val` and `train` to be formatted as json file exported by "VIA tool".
The json file has to contain only `polygon` region shapes and not other variates (like `point` or `triangle` etc.).
`PDL1NetDataLoader` the class responsible to load the data that reads the json file and loads only images that has non-trivial 
annotations. It expects to find the json file contains the annotation for the images in the folder.
The labels has priority order to insure that when collision is made, the higher priority label will prevail.
Priority is determine by the index position in the class list, the higher the index, the higher the priority.

## Usage
To initiate a train session use the next command in the command line:
```commandline
PDL1_main.py train --datatset path\to\root\folder --weights path\to\weight\file [optinal --augment]
```
To initiate a test session use the next command in the command line:
```commandline
PDL1_main.py test --datatset path\to\root\folder --weights path\to\weight\file
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
## Configuration
The configuration is a class that controls the meta parameters of the model - for example, the number of classes,
the backbone of the network, number of iteration per epoch, etc.
The right way to use the configuration is to create a class of your own, that derives `Config` class
(found in `algo\mrcnn\config`). In the new class, change only the parameter that you want to change,
the parameter you don't overload will be derived from the `Config` class. 
## Design

## License
Copyright 2020 Nati Daniel & Itamar Gruber & Shai Nahum Gefen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
