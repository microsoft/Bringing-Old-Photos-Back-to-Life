import cv2
import numpy as np
import os
import argparse 

def vintage(input_folder,output_folder):
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
        rows, cols = image_file.shape[:2]
        # Create a Gaussian filter
        kernel_x = cv2.getGaussianKernel(cols,200)
        kernel_y = cv2.getGaussianKernel(rows,200)
        kernel = kernel_y * kernel_x.T
        filter = 255 * kernel / np.linalg.norm(kernel)
        vintage_im = np.copy(image_file)
        # for each channel in the input image, we will apply the above filter
        for i in range(3):
            vintage_im[:,:,i] = vintage_im[:,:,i] * filter
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'vintage.png'), vintage_im)
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
    vintage(input_folder,output_folder)
    