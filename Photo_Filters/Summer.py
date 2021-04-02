import cv2
import numpy as np
import os
import argparse 

def gamma_function(channel, gamma):
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8") #creating lookup table
    channel = cv2.LUT(channel, table)
    return channel
    
def summer(input_folder,output_folder):
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
        image_file[:, :, 0] = gamma_function(image_file[:, :, 0], 0.75) # down scaling blue channel
        image_file[:, :, 2] = gamma_function(image_file[:, :, 2], 1.25) # up scaling red channel
        hsv = cv2.cvtColor(image_file, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = gamma_function(hsv[:, :, 1], 1.2) # up scaling saturation channel
        image_file = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'summer.png'), image_file)
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
    summer(input_folder,output_folder)
    