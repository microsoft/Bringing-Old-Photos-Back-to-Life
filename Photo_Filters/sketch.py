import cv2
import numpy as np
import os
import argparse 

def sketch(input_folder,output_folder):
    """ A program to apply filters on Output Images
    Parameters:
    --input_folder (str): Input Image folder location
    --output_folder (str): Output Image folder location
    
    """
    images = os.listdir(opts.input_folder)
    for image in images:
        # opening image from input folder
        # image_file = Image.open(opts.input_folder +'/'+ x)
        image_file = cv2.imread(opts.input_folder +'/'+ image)
        # converting image to black and white
        dst_gray = cv2.pencilSketch(image_file, sigma_s=60, sigma_r=0.07, shade_factor=0.05) # inbuilt function to generate pencil sketch in both color and grayscale
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'sketch.png'), dst_gray)
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./output", help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./filtered_image",
        help="Restored images, please use the absolute path",
    )
    opts = parser.parse_args()
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    # checking for output folder
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    input_folder = opts.input_folder
    output_folder = opts.output_folder
    sketch(input_folder,output_folder)
    