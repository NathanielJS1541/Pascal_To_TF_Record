# Argparse is a Python parser for command-line options and arguments
import argparse
# Module to represent filesystem paths which is able to handle paths for different OS.
from pathlib import Path
# Library to show progress bars on the command line
from tqdm import tqdm
import time


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


# ------------------------------------------------ Filetype Definitions ------------------------------------------------
# Glob-style patterns for input images
image_pattern = "*[.jpeg][.jpg]"
# Glob-style pattern for image labels
label_pattern = "*.xml"
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
argument_parser.add_argument("-f", "--force-overwrite", action="store_true",
                             help="If there already exists a file with the desired output path and "
                                  f"file name, overwrite it. {OutputColours.BOLD}{OutputColours.WARNING}WARNING: "
                                  f"May cause data loss!{OutputColours.END} Also allows recursive creation of "
                                  f"directories to get to the specified output file parent directory.")
# ----------------------------------------------------------------------------------------------------------------------

# Parse the arguments received by the program
args = argument_parser.parse_args()

# -------------------------------------------------- Data Validation ---------------------------------------------------
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
elif args.output.exists() and args.output.is_file() and not args.force_overwrite:
    raise FileExistsError(f"The specified output file already exists. If you're sure you want to overwrite, use the "
                          f"{OutputColours.BOLD}-f{OutputColours.END} flag.")
elif args.output.is_dir():
    raise ValueError(f"The specified output was a directory, not a file. The output is required to be a file in the "
                     f"{output_ext} format")
elif not args.output.parent.exists() and not args.force_overwrite:
    raise ValueError(f"The parent directory of the specified output file does not exist. By default the directory will "
                     f"not be created. Use {OutputColours.BOLD}-f{OutputColours.END} to override this.")

# Just warn the user about using the -f flag when it is not needed
if args.force_overwrite:
    print(f"{OutputColours.WARNING}WARNING: the {OutputColours.BOLD}-f{OutputColours.END}{OutputColours.WARNING} flag "
          f"allows the program to overwrite existing files on your system, and recursively create directories if you "
          f"specify a directory that does not exist. Only use it if you understand this.{OutputColours.END}")
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------- Glob Input Files --------------------------------------------------
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
for image_file in tqdm(input_images, desc=f"{OutputColours.INFO}[INFO] Pairing images and labels...", unit=" images"):
    # Get the image file name, and replace the extension with .xml to check whether a matching label file exists
    image_label_test = image_file.with_suffix(label_ext)
    if image_label_test in input_labels:
        # If a label exists for that image, add it to the array of input pairs
        input_pairs.append([image_file, image_label_test])
    else:
        # Warn the user in the case that an image doesn't seem to have a label
        print(f"{OutputColours.WARNING}[WARN] No label found for {image_file.__str__()}{OutputColours.END}")

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
