# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import skimage.io as io

# from face_sdk import FaceDetection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from PIL import Image, ImageFilter
import torch.nn.functional as F
import torchvision as tv
import torchvision.utils as vutils
import time
import cv2
import os
from skimage import img_as_ubyte
import json
import argparse
import dlib


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def _standard_face_pts():
    pts = (
        np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32) / 256.0
        - 1.0
    )

    return np.reshape(pts, (5, 2))


def _origin_face_pts():
    pts = np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32)

    return np.reshape(pts, (5, 2))


def compute_transformation_matrix(img, landmark, normalize, target_face_scale=1.0):

    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    # print(target_pts)

    h, w, c = img.shape
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    # print(landmark)

    affine = SimilarityTransform()

    affine.estimate(target_pts, landmark)

    return affine


def compute_inverse_transformation_matrix(img, landmark, normalize, target_face_scale=1.0):

    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    # print(target_pts)

    h, w, c = img.shape
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    # print(landmark)

    affine = SimilarityTransform()

    affine.estimate(landmark, target_pts)

    return affine


def show_detection(image, box, landmark):
    plt.imshow(image)
    print(box[2] - box[0])
    plt.gca().add_patch(
        Rectangle(
            (box[1], box[0]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none"
        )
    )
    plt.scatter(landmark[0][0], landmark[0][1])
    plt.scatter(landmark[1][0], landmark[1][1])
    plt.scatter(landmark[2][0], landmark[2][1])
    plt.scatter(landmark[3][0], landmark[3][1])
    plt.scatter(landmark[4][0], landmark[4][1])
    plt.show()


def affine2theta(affine, input_w, input_h, target_w, target_h):
    # param = np.linalg.inv(affine)
    param = affine
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0] * input_h / target_h
    theta[0, 1] = param[0, 1] * input_w / target_h
    theta[0, 2] = (2 * param[0, 2] + param[0, 0] * input_h + param[0, 1] * input_w) / target_h - 1
    theta[1, 0] = param[1, 0] * input_h / target_w
    theta[1, 1] = param[1, 1] * input_w / target_w
    theta[1, 2] = (2 * param[1, 2] + param[1, 0] * input_h + param[1, 1] * input_w) / target_w - 1
    return theta


def blur_blending(im1, im2, mask):

    mask *= 255.0

    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    mask = Image.fromarray(mask.astype("uint8")).convert("L")
    im1 = Image.fromarray(im1.astype("uint8"))
    im2 = Image.fromarray(im2.astype("uint8"))

    mask_blur = mask.filter(ImageFilter.GaussianBlur(20))
    im = Image.composite(im1, im2, mask)

    im = Image.composite(im, im2, mask_blur)

    return np.array(im) / 255.0


def blur_blending_cv2(im1, im2, mask):

    mask *= 255.0

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)

    mask_blur = cv2.GaussianBlur(mask, (25, 25), 0)
    mask_blur /= 255.0

    im = im1 * mask_blur + (1 - mask_blur) * im2

    im /= 255.0
    im = np.clip(im, 0.0, 1.0)

    return im


# def Poisson_blending(im1,im2,mask):


#     Image.composite(
def Poisson_blending(im1, im2, mask):

    # mask=1-mask
    mask *= 255
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask /= 255
    mask = 1 - mask
    mask *= 255

    mask = mask[:, :, 0]
    width, height, channels = im1.shape
    center = (int(height / 2), int(width / 2))
    result = cv2.seamlessClone(
        im2.astype("uint8"), im1.astype("uint8"), mask.astype("uint8"), center, cv2.MIXED_CLONE
    )

    return result / 255.0


def Poisson_B(im1, im2, mask, center):

    mask *= 255

    result = cv2.seamlessClone(
        im2.astype("uint8"), im1.astype("uint8"), mask.astype("uint8"), center, cv2.NORMAL_CLONE
    )

    return result / 255


def seamless_clone(old_face, new_face, raw_mask):

    height, width, _ = old_face.shape
    height = height // 2
    width = width // 2

    y_indices, x_indices, _ = np.nonzero(raw_mask)
    y_crop = slice(np.min(y_indices), np.max(y_indices))
    x_crop = slice(np.min(x_indices), np.max(x_indices))
    y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
    x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

    insertion = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
    insertion_mask = np.rint(raw_mask[y_crop, x_crop] * 255.0).astype("uint8")
    insertion_mask[insertion_mask != 0] = 255
    prior = np.rint(np.pad(old_face * 255.0, ((height, height), (width, width), (0, 0)), "constant")).astype(
        "uint8"
    )
    # if np.sum(insertion_mask) == 0:
    n_mask = insertion_mask[1:-1, 1:-1, :]
    n_mask = cv2.copyMakeBorder(n_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    print(n_mask.shape)
    x, y, w, h = cv2.boundingRect(n_mask[:, :, 0])
    if w < 4 or h < 4:
        blended = prior
    else:
        blended = cv2.seamlessClone(
            insertion,  # pylint: disable=no-member
            prior,
            insertion_mask,
            (x_center, y_center),
            cv2.NORMAL_CLONE,
        )  # pylint: disable=no-member

    blended = blended[height:-height, width:-width]

    return blended.astype("float32") / 255.0


def get_landmark(face_landmarks, id):
    part = face_landmarks.part(id)
    x = part.x
    y = part.y

    return (x, y)


def search(face_landmarks):

    x1, y1 = get_landmark(face_landmarks, 36)
    x2, y2 = get_landmark(face_landmarks, 39)
    x3, y3 = get_landmark(face_landmarks, 42)
    x4, y4 = get_landmark(face_landmarks, 45)

    x_nose, y_nose = get_landmark(face_landmarks, 30)

    x_left_mouth, y_left_mouth = get_landmark(face_landmarks, 48)
    x_right_mouth, y_right_mouth = get_landmark(face_landmarks, 54)

    x_left_eye = int((x1 + x2) / 2)
    y_left_eye = int((y1 + y2) / 2)
    x_right_eye = int((x3 + x4) / 2)
    y_right_eye = int((y3 + y4) / 2)

    results = np.array(
        [
            [x_left_eye, y_left_eye],
            [x_right_eye, y_right_eye],
            [x_nose, y_nose],
            [x_left_mouth, y_left_mouth],
            [x_right_mouth, y_right_mouth],
        ]
    )

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_url", type=str, default="./", help="origin images")
    parser.add_argument("--replace_url", type=str, default="./", help="restored faces")
    parser.add_argument("--save_url", type=str, default="./save")
    opts = parser.parse_args()

    origin_url = opts.origin_url
    replace_url = opts.replace_url
    save_url = opts.save_url

    if not os.path.exists(save_url):
        os.makedirs(save_url)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    count = 0

    for x in os.listdir(origin_url):
        img_url = os.path.join(origin_url, x)
        pil_img = Image.open(img_url).convert("RGB")

        origin_width, origin_height = pil_img.size
        image = np.array(pil_img)

        start = time.time()
        faces = face_detector(image)
        done = time.time()

        if len(faces) == 0:
            print("Warning: There is no face in %s" % (x))
            continue

        blended = image
        for face_id in range(len(faces)):

            current_face = faces[face_id]
            face_landmarks = landmark_locator(image, current_face)
            current_fl = search(face_landmarks)

            forward_mask = np.ones_like(image).astype("uint8")
            affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3)
            aligned_face = warp(image, affine, output_shape=(256, 256, 3), preserve_range=True)
            forward_mask = warp(
                forward_mask, affine, output_shape=(256, 256, 3), order=0, preserve_range=True
            )

            affine_inverse = affine.inverse
            cur_face = aligned_face
            if replace_url != "":

                face_name = x[:-4] + "_" + str(face_id + 1) + ".png"
                cur_url = os.path.join(replace_url, face_name)
                restored_face = Image.open(cur_url).convert("RGB")
                restored_face = np.array(restored_face)
                cur_face = restored_face

            ## Histogram Color matching
            A = cv2.cvtColor(aligned_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = cv2.cvtColor(cur_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = match_histograms(B, A)
            cur_face = cv2.cvtColor(B.astype("uint8"), cv2.COLOR_BGR2RGB)

            warped_back = warp(
                cur_face,
                affine_inverse,
                output_shape=(origin_height, origin_width, 3),
                order=3,
                preserve_range=True,
            )

            backward_mask = warp(
                forward_mask,
                affine_inverse,
                output_shape=(origin_height, origin_width, 3),
                order=0,
                preserve_range=True,
            )  ## Nearest neighbour

            blended = blur_blending_cv2(warped_back, blended, backward_mask)
            blended *= 255.0

        io.imsave(os.path.join(save_url, x), img_as_ubyte(blended / 255.0))

        count += 1

        if count % 1000 == 0:
            print("%d have finished ..." % (count))

