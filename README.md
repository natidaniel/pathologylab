# PathologyLab
## Description
this platform was created to provide a general and simple framework for pathological uses.
Our net (PDL1Net) deals with segmantation and counting of PDL1Net cells in WSIs of lung cancer patients, but can use for other implementations as well.
for easy orientation in this github please take a look in the following explanation about the building blocks.
## Setup

## Data
The net uses Whole Slide Images (WSI) IHC tainted. The patch size for the training step was **SIZE** and it was curated by pathologist.
### Data for Sessions
Both train session and test session expects the next folder structure:
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
The json file has to contain only `polygon` types and not other variates(like `point` or `triangle` etc.).
`PDL1NetDataLoader` the class responsible to load the data reads the json file and loads only images that has non-trivial 
annotations. It expects to find the json file contains the annotation for the images in the folder.

> PDL1_main.py train --dataset <path_to_dataset> --weights <see explanation below> [options]
The train stage expects json file as it produced by VIA tool (Oxford annotation tool) placed in the folder with the annotated images.


## Usage
To initiate a train session use the next command in the command line:
```commandline
PDL1_main.py train --datatset path\to\root\folder --weights path\to\weight\file [optinal --augment]
```
To initiate a test session use the next command in the command line:
```commandline
PDL1_main.py test --datatset path\to\root\folder --weights path\to\weight\file
```

## Configuration file
how to use the configuration file
## Design
## License
Copyright 2020 Nati Daniel & Itamar Gruber & Shai Nahum Gefen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
