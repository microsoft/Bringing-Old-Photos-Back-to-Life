import cv2
import numpy as np
import os
import argparse 

def daylight(input_folder,output_folder):
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
        image_HLS = cv2.cvtColor(image_file,cv2.COLOR_BGR2HLS) # Conversion to HLS
        image_HLS = np.array(image_HLS, dtype = np.float64)
        daylight = 1.15
        image_HLS[:,:,1] = image_HLS[:,:,1]*daylight # scale pixel values up for channel 1(Lightness)
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 # Sets all values above 255 to 255
        image_HLS = np.array(image_HLS, dtype = np.uint8)
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) # Conversion to RGB
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'daylight.png'), image_RGB)
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
    daylight(input_folder,output_folder)
    