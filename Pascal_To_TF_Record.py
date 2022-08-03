# Argparse is a Python parser for command-line options and arguments
import argparse
# Module to represent filesystem paths which is able to handle paths for different OS.
from pathlib import Path
# Library to show progress bars on the command line
from tqdm import tqdm
# function to generate a sha256sum for a file
from hashlib import sha256
# Function to create a "file-like" object in memory to store file bytes
from io import BytesIO
# Module implementing the ElementTree API for XML
from lxml import etree
# Module to decode images and store them as objects
import PIL.Image
# Function to write the TFRecord information to a file
from tensorflow.python.lib.io.tf_record import TFRecordWriter
# File IO wrappers without thread locking by Tensorflow
from tensorflow.python.platform.gfile import GFile
# Functions to convert raw data and features to a TFRecord format
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features
# Utility functions to create datasets and read label maps
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


# -------------------------------------------- Terminal Colour Definitions ---------------------------------------------
# Colours for output to terminal (Blender Style)
class OutputColours:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    INFO = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    VERBOSE = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------- Constants ------------------------------------------------------
progress_bar_ncols = 120
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------ Filetype Definitions ------------------------------------------------
# Glob-style patterns for input images
image_pattern = "*[.jpeg][.jpg]"
# Glob-style pattern for image labels
label_pattern = "*.xml"
# Image format name
image_format = "JPEG"
# Image label file format
label_ext = ".xml"
# File extension for the label map file
labelmap_ext = ".pbtxt"
# File extension for the output file
output_ext = ".record"
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------- Command Line Argument Definitions ------------------------------------------
# This is a list of the available arguments for the program along with their help texts
argument_parser = argparse.ArgumentParser(description="Convert a PascalVOC annotated dataset to a TFRecord dataset.",
                                          formatter_class=argparse.RawTextHelpFormatter)
argument_parser.add_argument("-d", "--dataset", required=True, type=Path,
                             help=f"Path to the folder containing the {OutputColours.INFO}PascalVOC{OutputColours.END} "
                                  f"dataset. (Both {OutputColours.INFO}.jpeg{OutputColours.END} and "
                                  f"{OutputColours.INFO}.xml{OutputColours.END} files should be in this directory)")
argument_parser.add_argument("-l", "--label_map", required=True, type=Path,
                             help=f"Path to the label map containing all the label names and IDs used to label the "
                                  f"dataset. This file can be created manually and should be in the "
                                  f"{OutputColours.INFO}{labelmap_ext}{OutputColours.END} format.")
argument_parser.add_argument("-o", "--output", required=True, type=Path,
                             help=f"File to output the {OutputColours.INFO}{output_ext}{OutputColours.END} data to. "
                                  f"This should be in the form {OutputColours.INFO}[PATH]/[FILENAME]{output_ext}"
                                  f"{OutputColours.END}")
argument_parser.add_argument("-s", "--skip-difficult", action="store_true",
                             help=f"Skip any labelled images that have been marked as {OutputColours.INFO}difficult"
                                  f"{OutputColours.END} in the labelling program.")
argument_parser.add_argument("-f", "--force", action="store_true",
                             help=f"If there already exists a file with the desired output path and "
                                  f"file name, overwrite it. {OutputColours.BOLD}{OutputColours.WARNING}WARNING: "
                                  f"May cause data loss!{OutputColours.END} Also allows recursive creation of "
                                  f"directories to get to the specified output file parent directory.")
argument_parser.add_argument("-v", "--verbose", action="store_true",
                             help=f"Include extra debug outputs. {OutputColours.WARNING}This will probably clutter your"
                                  f"screen.{OutputColours.END}")
# ----------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------ Function Definitions ------------------------------------------------
def pascal_to_tf_record(pascal_dict, image_path, label_map, skip_difficult=False):
    """Convert XML derived PascalVOC dict to the TFRecord format.
      Notice that this function normalizes the bounding box coordinates provided
      by the raw data.
      Args:
        pascal_dict: dict holding PascalVOC XML fields for a single image (obtained by running
            dataset_util.recursive_parse_xml_to_dict)
        image_path: Path to the image that matches the provided pascal_dict
        label_map: A map from string label names to integers ids.
        skip_difficult: Whether to skip images labelled as difficult (default: False).
      Returns:
        example: The converted tf.Example.
      Raises:
        ValueError: if the image pointed to by image_path is not a valid JPEG
    """
    # Open the image located at the specified file path
    with GFile(image_path, 'rb') as image_io:
        # Read the binary contents of the file from the disk
        encoded_jpg = image_io.read()
    # Store the contents of the file in memory as a "file-like" object
    encoded_jpg_io = BytesIO(encoded_jpg)
    # Open the image in pillow to decode it
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != image_format:
        # Raise this error if the JPEG encoding is a bit ~wack~, or just plain wrong.
        raise ValueError(f"Images need to be in {image_format} format. {image_path} has an invalid format.")

    # Create a sha256sum of the file
    key = sha256(encoded_jpg).hexdigest()

    # Get the width and height of the image from the PascalVOC dictionary
    width = int(pascal_dict['size']['width'])
    height = int(pascal_dict['size']['height'])

    # Initialise the array variables needed for the loop, these store properties about the different labels for the
    # given file in the order in which the labels are created.
    x_min = []              # The minimum x coordinates for the bounding boxes
    y_min = []              # The minimum y coordinates for the bounding boxes
    x_max = []              # The maximum x coordinates for the bounding boxes
    y_max = []              # The maximum y coordinates for the bounding boxes
    classes = []            # The IDs of the labels which are obtained by referencing the classes_text to the label_map
    classes_text = []       # The names of the labels corresponding to each bounding box
    truncated = []          # Property indicating whether the objects are fully or partially visible
    poses = []              # Property indicating whether the objects are skewed or not
    difficult_obj = []      # Property indicating whether the objects are difficult to recognise or not

    # If at least one label is defined within the .xml file
    if 'object' in pascal_dict:
        # Cycle through the individual labels in the .xml file
        for obj in pascal_dict['object']:
            # This dictates whether the label has been marked as difficult or not
            difficult = bool(int(obj['difficult']))

            # If the --skip-difficult flag is set, do not process labels marked as difficult
            if skip_difficult and difficult:
                print(f"{OutputColours.WARNING}[WARN] Label skipped as it was marked as difficult, and "
                      f"{OutputColours.BOLD}--skip-difficult{OutputColours.END}{OutputColours.WARNING} is set.")
                # Move on to the next label in the PascalVOC file
                continue

            # Populate the arrays that store information about each label in this image.
            difficult_obj.append(int(difficult))
            x_min.append(float(obj['bndbox']['xmin']) / width)
            y_min.append(float(obj['bndbox']['ymin']) / height)
            x_max.append(float(obj['bndbox']['xmax']) / width)
            y_max.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    # Pass the gathered data to tf.train to generate the .record file
    example = Example(features=Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            pascal_dict['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            pascal_dict['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_min),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_max),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_min),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_max),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    # Once the file is generated, return the .record file
    return example
# ----------------------------------------------------------------------------------------------------------------------


# Parse the arguments received by the program
args = argument_parser.parse_args()

# -------------------------------------------------- Data Validation ---------------------------------------------------
# Just warn the user about using the -f flag when it is not needed
if args.force:
    print(f"{OutputColours.WARNING}[WARN] The {OutputColours.BOLD}-f{OutputColours.END}{OutputColours.WARNING} flag "
          f"allows the program to overwrite existing files on your system, and recursively create directories if you "
          f"specify a directory that does not exist. Only use it if you understand this.{OutputColours.END}")

# Ensure that the dataset exists and is a path not a file
if not args.dataset.exists():
    raise FileNotFoundError(f"Dataset path does not exist. The -d/--dataset argument should be a path to an existing "
                            f"directory.")
elif args.dataset.is_file():
    raise ValueError(f"Dataset argument is a file. The -d/--dataset argument should be a path to a directory.")

# Ensure that the label map exists and is a file
if not args.label_map.exists():
    raise FileNotFoundError(f"Label map file does not exist. The -l/--label_map argument should be an existing file "
                            f"path.")
elif not args.label_map.is_file():
    raise ValueError(f"Label map supplied was a directory, not a file. The -l/--label_map argument should be an "
                     f"existing file path.")
elif args.label_map.suffix != labelmap_ext:
    raise ValueError(f"Label map should be in the {labelmap_ext} format, not {args.label_map.suffix}")

# Ensure that the output file does not already exist, and if it does require that the "-f" flag is present
if args.output.suffix != output_ext:
    raise FileExistsError(f"The wrong file extension was used. File extension should be {output_ext}, not "
                          f".{args.output.suffix}.")
elif args.output.exists() and args.output.is_file() and not args.force:
    raise FileExistsError(f"The specified output file already exists. If you're sure you want to overwrite, use the "
                          f"{OutputColours.BOLD}-f{OutputColours.END} flag.")
elif args.output.exists() and args.output.is_file() and args.force:
    # If the file already exists, but the -f flag is set, remove the file
    if args.verbose:
        # If being verbose, print the name of the file we are removing
        print(f"{OutputColours.VERBOSE}[DEBUG] Output file {args.output.__str__()} exists. Overwriting as "
              f"{OutputColours.BOLD}-f{OutputColours.END}{OutputColours.VERBOSE} was specified.{OutputColours.END}")
    args.output.unlink()
elif args.output.is_dir():
    raise ValueError(f"The specified output was a directory, not a file. The output is required to be a file in the "
                     f"{output_ext} format")
elif not args.output.parent.exists() and not args.force:
    raise ValueError(f"The parent directory of the specified output file does not exist. By default the directory will "
                     f"not be created. Use {OutputColours.BOLD}-f{OutputColours.END} to override this.")
elif not args.output.parent.exists() and args.force:
    if args.verbose:
        # If being verbose, print the name of the folder we are creating
        print(f"{OutputColours.VERBOSE}[DEBUG] Output parent folder {args.output.parent.__str__()} does not exist. "
              f"Recursively creating as {OutputColours.BOLD}-f{OutputColours.END}{OutputColours.VERBOSE} was specified."
              f"{OutputColours.END}")
    # If the parent folder doesn't already exist, recursively create it if the -f flag is set
    args.output.parent.mkdir(parents=True)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------- Glob Input Files --------------------------------------------------
# This creates the lists of Path() objects input_images and input_labels, based on the extensions we are looking for in
# the dataset folder.
input_images = sorted(args.dataset.glob(image_pattern))
print(f"{OutputColours.INFO}[INFO] Found {input_images.__len__()} images.{OutputColours.END}")
input_labels = sorted(args.dataset.glob(label_pattern))
print(f"{OutputColours.INFO}[INFO] Found {input_labels.__len__()} labels.{OutputColours.END}")
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------ Check for Labels Matching Images ------------------------------------------
# It may be the case that the user has not labelled all the images in the folder, or some images are missing. Here we
# pair them up and only process matching pairs. Look through the images as they could be either .jpg or .jpeg, but
# labels are always .xml. The user is also unlikely to have created labels without an image.
input_pairs = []
for image_file in tqdm(input_images, desc=f"{OutputColours.INFO}[INFO] Pairing images and labels...", unit=f" images",
                       ncols=progress_bar_ncols):
    # Get the image file name, and replace the extension with .xml to check whether a matching label file exists
    image_label_test = image_file.with_suffix(label_ext)
    if image_label_test in input_labels:
        # If a label exists for that image, add it to the array of input pairs
        input_pairs.append([image_file, image_label_test])
        if args.verbose:
            # If being verbose, print each image file name and label file name
            print(f"{OutputColours.VERBOSE}[DEBUG] Label found for {image_file.__str__()}: {image_label_test.__str__()}"
                  f"{OutputColours.END}")
    else:
        # Warn the user in the case that an image doesn't seem to have a label
        print(f"{OutputColours.WARNING}[WARN] No label found for {image_file.__str__()}{OutputColours.END}")
        if args.verbose:
            # If being verbose, print the image file name and the label file name that was tried
            print(f"{OutputColours.VERBOSE}[DEBUG] No label found for {image_file.__str__()}. File name tried: "
                  f"{image_label_test.__str__()}{OutputColours.END}")

# Print the number of input images to allow the user to check this is what they were expecting.
print(f"{OutputColours.INFO}[INFO] Found {input_pairs.__len__()} pairs of images and labels.{OutputColours.END}")

# Warn the user about the number of unused images
if input_pairs.__len__() < input_images.__len__():
    print(f"{OutputColours.WARNING}[WARN] {input_images.__len__() - input_pairs.__len__()} images unused as they aren't"
          f" labelled. Image labels should have the same name as the image.{OutputColours.END}")

# Warn the user about the number of unused labels
if input_pairs.__len__() < input_labels.__len__():
    print(f"{OutputColours.WARNING}[WARN] {input_labels.__len__() - input_pairs.__len__()} labels unused as they have "
          f"no matching image. Image labels should have the same name as the image.{OutputColours.END}")
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- Convert Files to TFRecord ----------------------------------------------
# Open a TFRecordWriter object to write to the specified file
record_writer = TFRecordWriter(args.output.__str__())
# Load the label map
label_map_dict = label_map_util.get_label_map_dict(args.label_map.__str__())
# Interpret each image/label pair one at a time
for file_pair in tqdm(input_pairs, desc=f"{OutputColours.INFO}[INFO] Processing Files...", unit=f" images",
                      ncols=progress_bar_ncols):
    if args.verbose:
        # If being verbose, print the files we are currently processing
        print(f"{OutputColours.VERBOSE}[DEBUG] Processing {file_pair[0].__str__()} and {file_pair[1].__str__()}"
              f"{OutputColours.END}")
    # Read the .xml file as a string
    with GFile(file_pair[1].__str__(), 'r') as label_file:
        xml_str = label_file.read()
    # Parse the xml file using etree
    xml = etree.fromstring(xml_str)
    # Create a Python dictionary from the parsed .xml file, and only store elements under the ["annotation"] tag.
    dict_from_xml = dataset_util.recursive_parse_xml_to_dict(xml)["annotation"]

    # Generate the TFRecord for this image and .xml file, using the label map.
    tf_record = pascal_to_tf_record(dict_from_xml, file_pair[0].__str__(), label_map_dict, args.skip_difficult)
    # Write this record to the output file.
    record_writer.write(tf_record.SerializeToString())
# ----------------------------------------------------------------------------------------------------------------------

# Close the writer before the program finishes
record_writer.close()
