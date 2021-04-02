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
        scale_percent = 0.60

        width = int(image_file.shape[1]*scale_percent)
        height = int(image_file.shape[0]*scale_percent)

        dim = (width,height)
        resized = cv2.resize(image_file,dim,interpolation = cv2.INTER_AREA)

        kernel_sharpening = np.array([[-1,-1,-1], 
                                    [-1, 9,-1],
                                    [-1,-1,-1]])
        sharpened = cv2.filter2D(resized,-1,kernel_sharpening)



        gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
        inv = 255-gray
        gauss = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)

        def dodgeV2(image,mask):
            return cv2.divide(image,255-mask,scale=256)

        pencil_jc = dodgeV2(gray,gauss)
        # saving image to output folder
        cv2.imwrite(str(opts.output_folder + '/' + 'sketch.png'), pencil_jc)
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
    