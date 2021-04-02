import cv2
import numpy as np
import os
import argparse 

def brightness(input_folder,output_folder):
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
        hsv = cv2.cvtColor(image_file, cv2.COLOR_BGR2HSV) # convert image to HSV color space
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*1.25 # scale pixel values up for channel 1
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*1.25 # scale pixel values up for channel 2
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'bright.png'), img)
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
    brightness(input_folder,output_folder)
