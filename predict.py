import tempfile
from pathlib import Path
import argparse
import shutil
import os
import glob
import cv2
import cog
from run import run_cmd


class Predictor(cog.Predictor):
    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_folder", type=str, default="input/cog_temp", help="Test images"
        )
        parser.add_argument(
            "--output_folder",
            type=str,
            default="output",
            help="Restored images, please use the absolute path",
        )
        parser.add_argument("--GPU", type=str, default="0", help="0,1,2")
        parser.add_argument(
            "--checkpoint_name",
            type=str,
            default="Setting_9_epoch_100",
            help="choose which checkpoint",
        )
        self.opts = parser.parse_args("")
        self.basepath = os.getcwd()
        self.opts.input_folder = os.path.join(self.basepath, self.opts.input_folder)
        self.opts.output_folder = os.path.join(self.basepath, self.opts.output_folder)
        os.makedirs(self.opts.input_folder, exist_ok=True)
        os.makedirs(self.opts.output_folder, exist_ok=True)

    @cog.input("image", type=Path, help="input image")
    @cog.input(
        "HR",
        type=bool,
        default=False,
        help="whether the input image is high-resolution",
    )
    @cog.input(
        "with_scratch",
        type=bool,
        default=False,
        help="whether the input image is scratched",
    )
    def predict(self, image, HR=False, with_scratch=False):
        try:
            os.chdir(self.basepath)
            input_path = os.path.join(self.opts.input_folder, os.path.basename(image))
            shutil.copy(str(image), input_path)

            gpu1 = self.opts.GPU

            ## Stage 1: Overall Quality Improve
            print("Running Stage 1: Overall restoration")
            os.chdir("./Global")
            stage_1_input_dir = self.opts.input_folder
            stage_1_output_dir = os.path.join(
                self.opts.output_folder, "stage_1_restore_output"
            )

            os.makedirs(stage_1_output_dir, exist_ok=True)

            if not with_scratch:

                stage_1_command = (
                        "python test.py --test_mode Full --Quality_restore --test_input "
                        + stage_1_input_dir
                        + " --outputs_dir "
                        + stage_1_output_dir
                        + " --gpu_ids "
                        + gpu1
                )
                run_cmd(stage_1_command)
            else:

                mask_dir = os.path.join(stage_1_output_dir, "masks")
                new_input = os.path.join(mask_dir, "input")
                new_mask = os.path.join(mask_dir, "mask")
                stage_1_command_1 = (
                        "python detection.py --test_path "
                        + stage_1_input_dir
                        + " --output_dir "
                        + mask_dir
                        + " --input_size full_size"
                        + " --GPU "
                        + gpu1
                )

                if HR:
                    HR_suffix = " --HR"
                else:
                    HR_suffix = ""

                stage_1_command_2 = (
                        "python test.py --Scratch_and_Quality_restore --test_input "
                        + new_input
                        + " --test_mask "
                        + new_mask
                        + " --outputs_dir "
                        + stage_1_output_dir
                        + " --gpu_ids "
                        + gpu1
                        + HR_suffix
                )

                run_cmd(stage_1_command_1)
                run_cmd(stage_1_command_2)

            ## Solve the case when there is no face in the old photo
            stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
            stage_4_output_dir = os.path.join(self.opts.output_folder, "final_output")
            os.makedirs(stage_4_output_dir, exist_ok=True)
            for x in os.listdir(stage_1_results):
                img_dir = os.path.join(stage_1_results, x)
                shutil.copy(img_dir, stage_4_output_dir)

            print("Finish Stage 1 ...")
            print("\n")

            ## Stage 2: Face Detection

            print("Running Stage 2: Face Detection")
            os.chdir(".././Face_Detection")
            stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
            stage_2_output_dir = os.path.join(
                self.opts.output_folder, "stage_2_detection_output"
            )
            os.makedirs(stage_2_output_dir, exist_ok=True)

            stage_2_command = (
                    "python detect_all_dlib_HR.py --url "
                    + stage_2_input_dir
                    + " --save_url "
                    + stage_2_output_dir
            )

            run_cmd(stage_2_command)
            print("Finish Stage 2 ...")
            print("\n")

            ## Stage 3: Face Restore
            print("Running Stage 3: Face Enhancement")
            os.chdir(".././Face_Enhancement")
            stage_3_input_mask = "./"
            stage_3_input_face = stage_2_output_dir
            stage_3_output_dir = os.path.join(
                self.opts.output_folder, "stage_3_face_output"
            )

            os.makedirs(stage_3_output_dir, exist_ok=True)

            self.opts.checkpoint_name = "FaceSR_512"
            stage_3_command = (
                    "python test_face.py --old_face_folder "
                    + stage_3_input_face
                    + " --old_face_label_folder "
                    + stage_3_input_mask
                    + " --tensorboard_log --name "
                    + self.opts.checkpoint_name
                    + " --gpu_ids "
                    + gpu1
                    + " --load_size 512 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 1 --results_dir "
                    + stage_3_output_dir
                    + " --no_parsing_map"
            )

            run_cmd(stage_3_command)
            print("Finish Stage 3 ...")
            print("\n")

            ## Stage 4: Warp back
            print("Running Stage 4: Blending")
            os.chdir(".././Face_Detection")
            stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
            stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
            stage_4_output_dir = os.path.join(self.opts.output_folder, "final_output")
            os.makedirs(stage_4_output_dir, exist_ok=True)

            stage_4_command = (
                    "python align_warp_back_multiple_dlib_HR.py --origin_url "
                    + stage_4_input_image_dir
                    + " --replace_url "
                    + stage_4_input_face_dir
                    + " --save_url "
                    + stage_4_output_dir
            )

            run_cmd(stage_4_command)
            print("Finish Stage 4 ...")
            print("\n")

            print("All the processing is done. Please check the results.")

            final_output = os.listdir(os.path.join(self.opts.output_folder, "final_output"))[0]

            image_restore = cv2.imread(os.path.join(self.opts.output_folder, "final_output", final_output))

            out_path = Path(tempfile.mkdtemp()) / "out.png"

            cv2.imwrite(str(out_path), image_restore)
        finally:
            clean_folder(self.opts.input_folder)
            clean_folder(self.opts.output_folder)
        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason:{e}")
