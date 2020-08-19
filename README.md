# PathologyLab
## Description
this platform was created to provide a general and simple framework for pathological uses.
Our net (PDL1Net) deals with segmantation and counting of PDL1Net cells in WSIs of lung cancer patients, but can use for other implementations as well.
for easy orientation in this github please take a look in the following explanation about the building blocks.
## Setup

## Data
The net uses Whole Slide Images (WSI) IHC tainted. The patch size for the training step was **SIZE** and it was curated by pathologist.
### Data for train
The following arguments will start a train session:
> PDL1_main.py train --dataset <path_to_dataset> --weights <see explanation below> [options]
The train stage expects json file as it produced by VIA tool (Oxford annotation tool) placed in the folder with the annotated images.


## Usage
Here add a description to make the code work. and include the code line to make it run.
## Configuration file
how to use the configuration file
## Design
## License
Copyright 2020 Nati Daniel & Itamar Gruber & Shai Nahum Gefen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
