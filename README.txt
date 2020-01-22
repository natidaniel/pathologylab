
this platform was created to provide a general and simple framework for pathological uses.
Our net (PD-L1_NET) deals with segmantation and counting of PD-L1 cells in WSIs of lung cancer patients, but can use for other implementations as well.
for easy orientation in this github please take a look in the following explanation about the building blocks.

pathologylab github structure:
files:
  1. README: introduction and instructions. you are here.
  2. setup: bash file which preforms all the needed installations.
  3. environment.yml / requairments.txt : packages, cloned gitHubs. all the python files in use in this project. 
     the setup activates this - and you have everything you need to use our platform.
  4. main_net.py - determine the characteristics for your needs (net's algorithm, task (train/test), dataset,...) and run.
     the format: ........


folders:
   1. algo: list of nets. each net should contain the following files:
            - X_Net.py: Net's architecture
            - trainer.py: initialization of the parameters from "params".
                          preforming the training.
            - tester.py: initialization of the parameters from "params".
                          preforming the inference.
                          analysis: results, comparations, ec. 
    2. datautils: data loader, data augmentation, formats converter, ...
    3. figs: ?
    4. logs: loggers outputs -  help to debuge and follow the flow.
       logging and its importance explaination: https://www.freecodecamp.org/news/you-should-have-better-logging-now-fbab2f667fac/
    5. logutils: ?
    6. params: parameters of all parts of each net. contains:
              json: 
                   train_params.json: net's parameters: batchsize, loss function, ec.
               paramsutils.py: class which its responsiblity is to parse the configuration files. 
    7. plotutils: visualization functions - creating plots, showing images, ec... 
    8. scripts:
          - playground: code fragments, "drafts"
          - tests: partial tests, unit tests
          - helpers: outsourced scripts
          - preprocessing: ?
    9. tools: outsourced tools in use - codes with specific functionality, collected from open sources 
                  
