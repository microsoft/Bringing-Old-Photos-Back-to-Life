import argparse
from subprocess import call

import cv2
import cv2 as cv
import dlib
import skimage.io as io
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageFilter
from matplotlib.patches import Rectangle
from openvino.inference_engine import IECore
from skimage import img_as_ubyte
from skimage.transform import SimilarityTransform
from skimage.transform import warp

import Face_Enhancement.data as data
from Face_Enhancement.models.pix2pix_model import Pix2PixModel
from Face_Enhancement.options.test_options import TestOptions
from Face_Enhancement.util.visualizer import Visualizer
from Global.detection_models import networks
from Global.detection_util.util import *
from Global.models.mapping_model import Pix2PixHDModel_Mapping

ImageFile.LOAD_TRUNCATED_IMAGES = True


def new_face_detector(image):
    plugin = IECore()

    device = 'CPU'

    FACE_DETECT_XML = "models/face-detection-adas-0001.xml"
    FACE_DETECT_BIN = "models/face-detection-adas-0001.bin"
    FACE_DETECT_INPUT_KEYS = 'data'
    FACE_DETECT_OUTPUT_KEYS = 'detection_out'
    net_face_detect = plugin.read_network(FACE_DETECT_XML, FACE_DETECT_BIN)
    # Load the Network using Plugin Device

    exec_face_detect = plugin.load_network(net_face_detect, device)

    #  Obtain image_count, channels, height and width
    n_face_detect, c_face_detect, h_face_detect, w_face_detect = net_face_detect.input_info[
        FACE_DETECT_INPUT_KEYS].input_data.shape

    blob = cv.resize(image, (w_face_detect, h_face_detect))  # Resize width & height
    blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    blob = blob.reshape((n_face_detect, c_face_detect, h_face_detect, w_face_detect))
    req_handle = exec_face_detect.start_async(
        request_id=0, inputs={FACE_DETECT_INPUT_KEYS: blob})

    time.sleep(1)   # TODO we have to wait a bit before request
    res = req_handle.output_blobs[FACE_DETECT_OUTPUT_KEYS].buffer

    answer = dlib.rectangles()

    for detection in res[0][0]:  # TODO check if res[0][0] sorted by confidence
        confidence = float(detection[2])
        # Obtain Bounding box coordinate, +-10 just for padding
        xmin = int(detection[3] * image.shape[1] - 10)
        ymin = int(detection[4] * image.shape[0] - 10)
        xmax = int(detection[5] * image.shape[1] + 10)
        ymax = int(detection[6] * image.shape[0] + 10)
        face = dlib.rectangle(left=xmin, bottom=ymax, right=xmax, top=ymin)
        if confidence > 0.9:
            answer.append(face)
    return answer


def _standard_face_pts():
    pts = (np.array([196.0, 226.0, 316.0, 226.0, 256.0,
                     286.0, 220.0, 360.4, 292.0, 360.4], np.float32) / 256.0 - 1.0)
    return np.reshape(pts, (5, 2))


def _origin_face_pts():
    pts = np.array([196.0, 226.0, 316.0, 226.0, 256.0,
                    286.0, 220.0, 360.4, 292.0, 360.4], np.float32)
    return np.reshape(pts, (5, 2))


def get_landmark(face_landmarks, id):
    part = face_landmarks.part(id)
    x = part.x
    y = part.y
    return x, y


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


def show_detection(image, box, landmark):
    plt.imshow(image)
    print(box[2] - box[0])
    plt.gca().add_patch(
        Rectangle((box[1], box[0]), box[2] - box[0], box[3] - box[1],
                  linewidth=1, edgecolor="r", facecolor="none")
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


def data_transforms_global(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    if full_size == "resize_256":
        return img.resize((parser.image_size, parser.image_size), method)

    if full_size == "scale_256":

        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def blend_mask(img, mask):
    np_img = np.array(img).astype("float")
    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img
    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")
    return hole_img

    # stage 4


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


# if compute_inverse_transformation_matrix set param inverse=True
def compute_transformation_matrix(img, landmark, normalize, target_face_scale=1.0, inverse=False):
    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    h, w, c = img.shape
    if normalize:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    affine = SimilarityTransform()
    if inverse:
        affine.estimate(landmark, target_pts)
    else:
        affine.estimate(target_pts, landmark)

    return affine


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


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "Global/checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="", help="Test images")
    parser.add_argument("--output_folder", type=str, help="Restored images, please use the absolute path")
    parser.add_argument("--GPU", type=str, default="-1", help="CPU: -1, GPU: 0,1,2")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint")
    parser.add_argument("--with_scratch", action="store_true")
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    # Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    if not opts.with_scratch:
        stage_1_command = (f'python test.py --test_mode Full --Quality_restore --test_input '
                           f'{stage_1_input_dir} --outputs_dir {stage_1_output_dir} --gpu_ids {gpu1}')
        run_cmd(stage_1_command)
    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")

        print("initializing the dataloader")
        parser = argparse.ArgumentParser()

        parser.GPU = int(gpu1)
        parser.test_path = stage_1_input_dir
        parser.output_dir = mask_dir
        parser.input_size = "scale_256"

        model = networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )

        # load model
        checkpoint_path = "./Global/checkpoints/detection/FT_Epoch_latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print("model weights loaded")

        if parser.GPU < 0:
            model.cpu()
        else:
            model.to(parser.GPU)
        model.eval()

        # dataloader and transformation
        print(f'directory of testing image: {parser.test_path}')
        imagelist = os.listdir(parser.test_path)
        imagelist.sort()
        total_iter = 0

        P_matrix = {}
        save_url = os.path.join(parser.output_dir)
        mkdir_if_not(save_url)

        input_dir = os.path.join(save_url, "input")
        output_dir = os.path.join(save_url, "mask")
        # blend_output_dir=os.path.join(save_url, 'blend_output')
        mkdir_if_not(input_dir)
        mkdir_if_not(output_dir)
        # mkdir_if_not(blend_output_dir)

        idx = 0

        for image_name in imagelist:

            idx += 1

            print("processing", image_name)

            results = []
            scratch_file = os.path.join(parser.test_path, image_name)
            if not os.path.isfile(scratch_file):
                print(f'Skipping non-file {image_name}')
                continue
            scratch_image = Image.open(scratch_file).convert("RGB")

            w, h = scratch_image.size

            transformed_image_PIL = data_transforms_global(scratch_image, parser.input_size)

            scratch_image = transformed_image_PIL.convert("L")
            scratch_image = tv.transforms.ToTensor()(scratch_image)

            scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)

            scratch_image = torch.unsqueeze(scratch_image, 0)

            if parser.GPU < 0:
                scratch_image = scratch_image.cpu()
            else:
                scratch_image = scratch_image.to(parser.GPU)

            P = torch.sigmoid(model(scratch_image))

            P = P.data.cpu()

            tv.utils.save_image(
                (P >= 0.4).float(),
                os.path.join(output_dir, f'{image_name[:-4]}.png'),
                nrow=1,
                padding=0,
                normalize=True,
            )
            transformed_image_PIL.save(os.path.join(input_dir, f'{image_name[:-4]}.png'))

        opt = argparse.ArgumentParser()
        opt.Scratch_and_Quality_restore = True
        opt.test_input = new_input
        opt.test_mask = new_mask
        opt.outputs_dir = stage_1_output_dir
        opt.gpu_ids = [int(gpu1)]
        opt.isTrain = False
        opt.resize_or_crop = 'scale_width'
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.norm = 'instance'
        opt.spatio_size = 64
        opt.feat_dim = -1
        opt.use_segmentation_model = False
        opt.softmax_temperature = 1.0
        opt.use_self = False
        opt.cosin_similarity = False
        opt.mapping_net_dilation = 1
        opt.load_pretrain = ''
        opt.no_load_VAE = False
        opt.which_epoch = 'latest'
        opt.use_vae_which_epoch = 'latest'

        opt.Quality_restore = False
        parameter_set(opt)
        model = Pix2PixHDModel_Mapping()

        model.initialize(opt)
        model.eval()

        if not os.path.exists(f'{opt.outputs_dir}/input_image'):
            os.makedirs(f'{opt.outputs_dir}/input_image')
        if not os.path.exists(f'{opt.outputs_dir}/restored_image'):
            os.makedirs(f'{opt.outputs_dir}/restored_image')
        if not os.path.exists(f'{opt.outputs_dir}/origin'):
            os.makedirs(f'{opt.outputs_dir}/origin')

        input_loader = os.listdir(opt.test_input)
        dataset_size = len(input_loader)
        input_loader.sort()

        if opt.test_mask is not "":
            mask_loader = os.listdir(opt.test_mask)
            dataset_size = len(os.listdir(opt.test_mask))
            mask_loader.sort()

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        mask_transform = transforms.ToTensor()

        for i in range(dataset_size):

            input_name = input_loader[i]
            input_file = os.path.join(opt.test_input, input_name)
            if not os.path.isfile(input_file):
                print(f'Skipping non-file {input_name}')
                continue
            input = Image.open(input_file).convert("RGB")

            print(f'Now you are processing {input_name}')

            if opt.NL_use_mask:
                mask_name = mask_loader[i]
                mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
                origin = input
                input = irregular_hole_synthesize(input, mask)
                mask = mask_transform(mask)
                mask = mask[:1, :, :]  # Convert to single channel
                mask = mask.unsqueeze(0)
                input = img_transform(input)
                input = input.unsqueeze(0)
            else:
                if opt.test_mode == "Scale":
                    input = data_transforms(input, scale=True)
                if opt.test_mode == "Full":
                    input = data_transforms(input, scale=False)
                if opt.test_mode == "Crop":
                    input = data_transforms_rgb_old(input)
                origin = input
                input = img_transform(input)
                input = input.unsqueeze(0)
                mask = torch.zeros_like(input)
            # Necessary input

            try:
                generated = model.inference(input, mask)
            except Exception as ex:
                print(f'Skip {input_name} due to an error:\n {str(ex)}')
                continue

            if input_name.endswith(".jpg"):
                input_name = f'{input_name[:-4]}.png'

            vutils.save_image(
                (input + 1.0) / 2.0,
                f'{opt.outputs_dir}/input_image/{input_name}',
                nrow=1,
                padding=0,
                normalize=True,
            )
            vutils.save_image(
                (generated.data.cpu() + 1.0) / 2.0,
                f'{opt.outputs_dir}/restored_image/{input_name}',
                nrow=1,
                padding=0,
                normalize=True,
            )

            origin.save(f'{opt.outputs_dir}/origin/{input_name}')

    # Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...\n")

    # Stage 2: Face Detection

    print("Running Stage 2: Face Detection")
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)

    url = stage_2_input_dir
    save_url = stage_2_output_dir

    # If the origin url is None, then we don't need to reid the origin image

    os.makedirs(url, exist_ok=True)
    os.makedirs(save_url, exist_ok=True)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor("Face_Detection/shape_predictor_68_face_landmarks.dat")

    map_id = {}
    for x in os.listdir(url):
        img_url = os.path.join(url, x)
        pil_img = Image.open(img_url).convert("RGB")

        image = np.array(pil_img)

        start = time.time()
        faces = new_face_detector(image)
        done = time.time()

        if not faces:
            print(f'Warning: There is no face in {x}')
            continue
        else:
            for face_id, current_face in enumerate(faces):
                face_landmarks = landmark_locator(image, current_face)
                current_fl = search(face_landmarks)

                affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3,
                                                       inverse=False).params
                aligned_face = warp(image, affine, output_shape=(256, 256, 3))
                img_name = f'{x[:-4]}_{face_id + 1}'
                io.imsave(os.path.join(save_url, f'{img_name}.png'), img_as_ubyte(aligned_face))

    print("Finish Stage 2 ...\n")

    # Stage 3: Face Restore
    print("Running Stage 3: Face Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)

    opt = TestOptions()
    opt.initialize(argparse.ArgumentParser())
    opt.old_face_folder = stage_3_input_face
    opt.old_face_label_folder = stage_3_input_mask
    opt.tensorboard_log = True
    opt.name = opts.checkpoint_name
    opt.gpu_ids = [int(gpu1)]
    opt.load_size = 256
    opt.label_nc = 18
    opt.no_instance = True
    opt.preprocess_mode = 'resize'
    opt.batchSize = 4
    opt.results_dir = stage_3_output_dir
    opt.no_parsing_map = True
    opt.dataroot = "./Face_Enhancement/datasets/cityscapes/"
    opt.serial_batches = False
    opt.nThreads = 0
    opt.netG = "spade"
    opt.ngf = 64
    opt.num_upsampling_layers = 'normal'
    opt.crop_size = 256
    opt.aspect_ratio = 1.0
    opt.use_vae = False
    opt.injection_layer = 'all'
    opt.norm_G = 'spectralspadesyncbatch3x3'
    opt.norm_D = 'spectralinstance'
    opt.norm_E = 'spectralinstance'
    opt.contain_dontcare_label = False
    opt.semantic_nc = (
            opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
    )
    opt.init_type = 'normal'
    opt.init_variance = 0.02
    opt.which_epoch = 'latest'
    opt.checkpoints_dir = './Face_Enhancement/checkpoints'
    opt.display_winsize = 256
    opt.how_many = float("inf")
    dataloader = data.create_dataloader(opt)

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    single_save_url = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")

    if not os.path.exists(single_save_url):
        os.makedirs(single_save_url)

    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        generated = model(data_i, mode="inference")

        img_path = data_i["path"]

        for b in range(generated.shape[0]):
            img_name = os.path.split(img_path[b])[-1]
            save_img_url = os.path.join(single_save_url, img_name)

            vutils.save_image((generated[b] + 1) / 2, save_img_url)

    print("Finish Stage 3 ...\n")

    # Stage 4: Warp back
    print("Running Stage 4: Blending")
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)

    origin_url = stage_4_input_image_dir
    replace_url = stage_4_input_face_dir
    save_url = stage_4_output_dir

    if not os.path.exists(save_url):
        os.makedirs(save_url)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor("Face_Detection/shape_predictor_68_face_landmarks.dat")

    for x in os.listdir(origin_url):
        img_url = os.path.join(origin_url, x)
        pil_img = Image.open(img_url).convert("RGB")

        origin_width, origin_height = pil_img.size
        image = np.array(pil_img)

        start = time.time()
        faces = new_face_detector(image)
        done = time.time()

        if not faces:
            print(f'Warning: There is no face in {x}')
            continue

        blended = image
        for face_id, current_face in enumerate(faces):
            current_face = faces[face_id]
            face_landmarks = landmark_locator(image, current_face)
            current_fl = search(face_landmarks)

            forward_mask = np.ones_like(image).astype("uint8")
            affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3, inverse=False)
            aligned_face = warp(image, affine, output_shape=(256, 256, 3), preserve_range=True)
            forward_mask = warp(
                forward_mask, affine, output_shape=(256, 256, 3), order=0, preserve_range=True
            )

            affine_inverse = affine.inverse
            cur_face = aligned_face
            if replace_url is not "":
                face_name = f'{x[:-4]}_{face_id + 1}.png'
                cur_url = os.path.join(replace_url, face_name)
                restored_face = Image.open(cur_url).convert("RGB")
                restored_face = np.array(restored_face)
                cur_face = restored_face

            # Histogram Color matching
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
            )  # Nearest neighbour

            blended = blur_blending_cv2(warped_back, blended, backward_mask)
            blended *= 255.0

        io.imsave(os.path.join(save_url, x), img_as_ubyte(blended / 255.0))

    print("Finish Stage 4 ...\n")

    print("All the processing is done. Please check the results.")
