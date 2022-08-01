# Argparse is a Python parser for command-line options and arguments
import argparse
# Module to represent filesystem paths which is able to handle paths for different OS.
from pathlib import Path


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
