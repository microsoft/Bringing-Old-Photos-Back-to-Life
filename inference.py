# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import shutil
import sys
import time
import cv2
from subprocess import call

def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def split_dirs(input_dir, num_files=5):
    files = os.listdir(input_dir)
    files = [os.path.join(input_dir, x) for x in files]
    total_files = len(files)
    dirs = []

    for i in range(0, total_files, num_files):
        folder = f"{input_dir}_{i}_{i+num_files}"
        os.makedirs(folder, exist_ok=True)
        dirs.append(folder)
        for file in files[i:i+num_files]:
            new_path = os.path.join(folder, os.path.basename(file))
            shutil.copy(file, new_path)

    return dirs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images", help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Restored images, please use the absolute path",
    )
    parser.add_argument("--GPU", type=str, default="0", help="0,1,2")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint"
    )
    parser.add_argument("--with_scratch", action="store_true", default=True)
    parser.add_argument("--HR", action='store_true')
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    ## Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    os.chdir("./Global")
    stage_1_input_dir = opts.input_folder
    tmp_out = os.path.abspath(".")
    tmp_out = os.path.join(tmp_out, "tmp_out")
    os.makedirs(tmp_out, exist_ok=True)

    if len(os.listdir(stage_1_input_dir)) > 3:
        input_dirs = split_dirs(stage_1_input_dir, 3)
    else:
        input_dirs = [stage_1_input_dir]
    stage_1_output_dir = os.path.join(
        tmp_out, "stage_1_restore_output")
        # opts.output_folder, "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)
    os.makedirs(os.path.join(stage_1_output_dir, "restored_image"), exist_ok=True)

    cwd = os.getcwd()
    os.makedirs(opts.output_folder, exist_ok=True)

    for folder in input_dirs:
        os.chdir(cwd)
        if not opts.with_scratch:
            stage_1_command = (
                "python test.py --test_mode Full --Quality_restore --test_input "
                + folder
                + " --outputs_dir "
                + stage_1_output_dir
                + " --gpu_ids "
                + gpu1
            )
            start = time.time()
            run_cmd(stage_1_command)
            print(f"Stage 1 finished in {(time.time() - start):.3}")
        else:

            mask_dir = os.path.join(stage_1_output_dir, "masks")
            new_input = os.path.join(mask_dir, "input")
            new_mask = os.path.join(mask_dir, "mask")
            stage_1_command_1 = (
                "python detection.py --test_path "
                + folder
                + " --output_dir "
                + mask_dir
                + " --input_size full_size"
                + " --GPU "
                + gpu1
            )

            if opts.HR:
                HR_suffix=" --HR"
            else:
                HR_suffix=""

            stage_1_command_2 = (
                "python test.py --Scratch_and_Quality_restore --test_input "
                + new_input
                + " --test_mask "
                + new_mask
                + " --outputs_dir "
                + stage_1_output_dir
                + " --gpu_ids "
                + gpu1 + HR_suffix
            )

            start = time.time()
            run_cmd(stage_1_command_1)
            print(f"Stage 1.1 finished in {(time.time() - start):.3}")
            start = time.time()
            run_cmd(stage_1_command_2)
            print(f"Stage 1.2 finished in {(time.time() - start):.3}")

        ## Solve the case when there is no face in the old photo
        stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
        stage_4_output_dir = os.path.join(tmp_out, "final_output")
        os.makedirs(stage_1_results, exist_ok=True)
        os.makedirs(stage_4_output_dir, exist_ok=True)
        for x in os.listdir(stage_1_results):
            img_dir = os.path.join(stage_1_results, x)
            shutil.copy(img_dir, stage_4_output_dir)
        print("\n")

        ## Stage 2: Face Detection

        print("Running Stage 2: Face Detection")
        os.chdir(".././Face_Detection")
        stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
        stage_2_output_dir = os.path.join(tmp_out, "stage_2_detection_output")
        if not os.path.exists(stage_2_output_dir):
            os.makedirs(stage_2_output_dir)
        if opts.HR:
            stage_2_command = (
                "python detect_all_dlib_HR.py --url " + stage_2_input_dir + " --save_url " + stage_2_output_dir
            )
        else:
            stage_2_command = (
                "python detect_all_dlib.py --url " + stage_2_input_dir + " --save_url " + stage_2_output_dir
            )

        start = time.time()
        run_cmd(stage_2_command)
        print(f"Stage 2 finished in {(time.time() - start):.3}")
        print("\n")

        ## Stage 3: Face Restore
        print("Running Stage 3: Face Enhancement")
        os.chdir(".././Face_Enhancement")
        stage_3_input_mask = "./"
        stage_3_input_face = stage_2_output_dir
        stage_3_output_dir = os.path.join(tmp_out, "stage_3_face_output")
        if not os.path.exists(stage_3_output_dir):
            os.makedirs(stage_3_output_dir)

        if opts.HR:
            opts.checkpoint_name='FaceSR_512'
            stage_3_command = (
                "python test_face.py --old_face_folder "
                + stage_3_input_face
                + " --old_face_label_folder "
                + stage_3_input_mask
                + " --tensorboard_log --name "
                + opts.checkpoint_name
                + " --gpu_ids "
                + gpu1
                + " --load_size 512 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 1 --results_dir "
                + stage_3_output_dir
                + " --no_parsing_map"
            )
        else:
            stage_3_command = (
                "python test_face.py --old_face_folder "
                + stage_3_input_face
                + " --old_face_label_folder "
                + stage_3_input_mask
                + " --tensorboard_log --name "
                + opts.checkpoint_name
                + " --gpu_ids "
                + gpu1
                + " --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir "
                + stage_3_output_dir
                + " --no_parsing_map"
            )

        start = time.time()
        run_cmd(stage_3_command)
        print(f"Stage 3 finished in {(time.time() - start):.3}")
        print("\n")

        ## Stage 4: Warp back
        print("Running Stage 4: Blending")
        os.chdir(".././Face_Detection")
        stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
        stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
        stage_4_output_dir = os.path.join(tmp_out, "final_output")
        if not os.path.exists(stage_4_output_dir):
            os.makedirs(stage_4_output_dir)

        stage_4_command = (
            "python align_warp_back_multiple_dlib.py --origin_url "
            + stage_4_input_image_dir
            + " --replace_url "
            + stage_4_input_face_dir
            + " --save_url "
            + stage_4_output_dir
        )

        start = time.time()
        run_cmd(stage_4_command)
        print(f"Stage 4 finished in {(time.time() - start):.3}")
        print("\n")

        for im in os.listdir(stage_4_output_dir):
            old = os.path.join(stage_4_output_dir, im)
            new = os.path.join(opts.output_folder, im)
            shutil.move(old, new)
        shutil.rmtree(tmp_out)

    for folder in input_dirs:
        shutil.rmtree(folder)

    print("All the processing is done. Please check the results.")
