import cv2
import numpy as np
import os
import argparse 

def tv(input_folder,output_folder):
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
        height, width = image_file.shape[:2]
        gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
        thresh = 0.8 # creating threshold. This means noise will be added to 80% pixels
        for i in range(height):
            for j in range(width):
                if np.random.rand() <= thresh:
                    if np.random.randint(2) == 0:
                        gray[i, j] = min(gray[i, j] + np.random.randint(0, 64), 255) # adding random value between 0 to 64. Anything above 255 is set to 255.
                    else:
                        gray[i, j] = max(gray[i, j] - np.random.randint(0, 64), 0) # subtracting random values between 0 to 64. Anything below 0 is set to 0.

        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + '60sTv.png'), gray)
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
    tv(input_folder,output_folder)
    