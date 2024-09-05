### Imaris File Parsers for NIH

Update: 

#### Project Description
* This repo is designed to extract statistical information from raw Imaris file with (.ims) extension. 
* This script does not calculate any statistical information, it simply extracts existing information from each .ims file and formats in a manner that helps with downsteam stream analysis software such as FlowJo.

#### Setup
* Tested on: Windows 10, Ubuntu 20.04
* Install Time: <10 minutes
Create Enviroment:
```
conda env create -f environment.yml
```

#### Usage Tips
* We currently offer 4 types of data parsers
    * Surface Stats
    * Surface Track Stats
    * Surfaces at Unique Time Steps
    * All Tracks 
* Each of the jupyter notebooks for the parsers above uses a config file from parsers/config/ folder.
* To extract information from your own Imaris Files, simply grab the paths to each directory that contains the .ims files and update the config.yaml file, provide a save directory and run any one of the notebooks to extract the relevant information. 
* For each file a formatted csv file will be generated that contains all relevant statistical information from each .ims file for further analysis.

#### Folder Structure
- Parsers
    - data
    - parsers
        - config
        - imaris
        - parsers
        - scripts
        - utils

#### Contributors
* Shehan Perera
* Juraj Kabat
