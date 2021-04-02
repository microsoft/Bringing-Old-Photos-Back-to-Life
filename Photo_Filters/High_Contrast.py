import cv2
import numpy as np
import os
import argparse 

def contrast(input_folder,output_folder):
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
        gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
        xp = [0, 64, 112, 128, 144, 192, 255] # setting reference values
        fp = [0, 16, 64, 128, 192, 240, 255] # setting values to be taken for reference values
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8') # creating lookup table
        img = cv2.LUT(gray, table) # changing values based on lookup table
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'contrast.png'), img)
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
    contrast(input_folder,output_folder)
    