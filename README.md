# Pascal_To_TF_Record
Python script to convert a custom dataset labelled in a program such as [labelimg](https://github.com/heartexlabs/labelImg) which outputs data in the [PascalVOC ](http://host.robots.ox.ac.uk/pascal/VOC/) format to the [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format to be used with [TensorFlow](https://www.tensorflow.org/). This program is based on the [create_pascal_tf_record.py](https://github.com/tensorflow/models/blob/0ff8db0a62c0ef1bd5e80355f857889cafeadbf2/research/object_detection/dataset_tools/create_pascal_tf_record.py) script over at the [TensorFlow Model Garden](https://github.com/tensorflow/models) repo, but usable "out of the box" by anyone trying to label and create their own dataset. I found that I needed to edit a significant amount of the script in that repo to do what I needed, so I thought I'd publish it for other unfortunate souls trying to create their own datasets for TensorFlow.

## Requirements
- Python3
- An annotated dataset created by a program such as [labelimg](https://github.com/heartexlabs/labelImg). The `.xml` annotations are assumed to be in the same folder as the images.
- Images in `.jpeg` or `.jpg` format.
- A `label_map.pbtxt` file. These are relatively simple to create in a text editor, and I have left an example in this repo to refer to. Simply change the label names and add extra entries with unique id's if required.
- A working TensorFlow install for Python to use (See https://www.tensorflow.org/install)
- (Preferably) A Python venv to avoid any dependency conflicts.

## Usage

## Disclaimer
I needed this script as a "fast and loose" solution to a problem I encountered. Chances are the repo will end up unmaintained, but I aim to try and make it easy to understand and modify if needed.
