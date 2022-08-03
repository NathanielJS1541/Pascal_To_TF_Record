# Pascal_To_TF_Record
Python script to convert a custom dataset labelled in a program such as 
[labelimg](https://github.com/heartexlabs/labelImg) which outputs data in the 
[PascalVOC ](http://host.robots.ox.ac.uk/pascal/VOC/) format to the 
[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format to be used with 
[TensorFlow](https://www.tensorflow.org/). This program is based on the 
[create_pascal_tf_record.py](https://github.com/tensorflow/models/blob/0ff8db0a62c0ef1bd5e80355f857889cafeadbf2/research/object_detection/dataset_tools/create_pascal_tf_record.py) 
script over at the [TensorFlow Model Garden](https://github.com/tensorflow/models) repo, but usable "out of the box" by 
anyone trying to label and create their own dataset. I found that I needed to edit a significant amount of the script 
in that repo to do what I needed, so I thought I'd publish it for other unfortunate souls trying to create their own 
datasets for TensorFlow.

## Requirements
- `Python3` and `pip` installed and __on your path__.
- This repo cloned locally with `git clone https://github.com/NathanielJS1541/Pascal_To_TF_Record.git`
- An annotated dataset created by a program such as [labelimg](https://github.com/heartexlabs/labelImg). The `.xml`
  annotations are assumed to be in the same folder as the images, with the same file names (but different extensions!).
- Images in `.jpeg` or `.jpg` format.
- A `label_map.pbtxt` file. These are relatively simple to create in a text editor, and I have left an example in this 
  repo to refer to. Simply change the label names and add extra entries with unique id's if required.
- A working TensorFlow install for Python to use (See https://www.tensorflow.org/install)
- (Preferably) A Python [venv](https://docs.python.org/3/library/venv.html) to avoid any dependency conflicts.
- Install the pip dependencies from the `requirements.txt` file as shown below.
- Installing the `object_detection` module from [TensorFlow Model Garden](https://github.com/tensorflow/models) as shown
  below.

### Installing Dependencies With `pip`
- Navigate to the locally cloned repo in a terminal.
- If you have a venv (which you should...), activate it with `source .venv/bin/activate` on Linux or 
  `.\tf2_api_env\Scripts\Activate.ps1` on Windows. You should see `(venv)` or similar appear on your terminal.
- Run `pip install -r requirements.txt`

### Installing the object_detection Module
The documentation for this step can be found at 
https://github.com/tensorflow/models/tree/master/research/object_detection.
- Locally clone the repo with `git clone https://github.com/tensorflow/models.git`. (If you have slow internet you can
  set the depth to 1 to only download current commits with 
  `git clone https://github.com/tensorflow/models.git --depth 1`).
- In a command prompt (starting from the folder containing the __models__ repo), navigate to the __research__ folder in 
  the __models__ repo: `cd ./models/research/`.
- Copy the object_detection `setup.py` to the working directory: `cp ./object_detection/packages/tf2/setup.py .`
- If you're using a `venv`, make sure it is activated like in the step above!
- Install the module with `python -m pip install .`
- If all is successful, you can test the module by running `python object_detection/builders/model_builder_tf2_test.py`

## Usage
Available options can be viewed with `python Pascal_To_TF_Record.py --help`:
```commandline
$ python Pascal_To_TF_Record.py --help
usage: Pascal_To_TF_Record.py [-h] -d DATASET -l LABEL_MAP -o OUTPUT [-s] [-f] [-v]

Convert a PascalVOC annotated dataset to a TFRecord dataset.

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Path to the folder containing the PascalVOC dataset. (Both .jpeg and .xml files should be in this directory)
  -l LABEL_MAP, --label_map LABEL_MAP
                        Path to the label map containing all the label names and IDs used to label the dataset. This file can be created manually and should be in the .pbtxt format.
  -o OUTPUT, --output OUTPUT
                        File to output the .record data to. This should be in the form [PATH]/[FILENAME].record
  -s, --skip-difficult  Skip any labelled images that have been marked as difficult in the labelling program.
  -f, --force
                        If there already exists a file with the desired output path and file name, overwrite it. WARNING: May cause data loss! Also allows recursive creation of directories to get to the specified output file parent directory.
  -v, --verbose         Include extra debug outputs. This will probably clutter yourscreen.
```

### Required Options
These options must be specified for the program to run.
- `-d DATASET`: This should be the folder where your labelled images and labels are stored.
- `-l LABEL_MAP`: This is the label map which you should create containing the same label names that you labelled your
  data with. You will probably get an error if it does not contain all of the label names you used. An example can be ]
  found in this repo called `pascal_label_map.pbtxt`, which was taken from the
  [TensorFlow Model Garden](https://github.com/tensorflow/models). Simply change the names of the labels to match what 
  you need and delete any extra entries (or add more if you need to). Just make sure each label has a unique ID.
- `-o OUTPUT`: This should be a path to a non-existent __.record__ file that the program will output the processed data 
  to. If you intend to generate a new version of an existing file, you will need to specify the `-f` flag.

### Optional Flags
These flags can be specified to slightly change the behaviour of the program.
- `-s`: In [labelimg](https://github.com/heartexlabs/labelImg) you have the option to mark an image as __difficult__. 
  This indicates that the labelled object is not fully visible or is difficult to see. Specifying this flag will skip 
  these images to allow you to quickly test whether it is causing issues for your model.
- `-f`: Allows the program to overwrite the specified output file if it already exists, and create any parent 
  directories specified by `-o` for the output file to be saved in. __USE WITH CAUTION!__
- `-v`: Prints an enormous amount of data to the screen to help you figure out what is going wrong.

## Disclaimer
I needed this script as a "fast and loose" solution to a problem I encountered. Chances are the repo will end up 
unmaintained, but I aim to try and make it easy to understand and modify if needed.
