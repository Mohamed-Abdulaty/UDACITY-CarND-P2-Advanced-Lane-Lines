import numpy as np

from Logger import Logger
from Calibration import Calibration
from Parameters import Parameters
from Pipeline import Pipeline

# Paths
results_directory      = './results'

calibration_src_images = './camera_cal'
calibration_dst_images = 'Calibration_results'

test_images_src        = './test_images'
test_images_des        = 'test_images_results'

test_vedios_src         = './test_vedios'
test_vedios_des         = './output_vedios'

# Calibration
logger_obj      = Logger(str(results_directory))
calibration_obj = Calibration(calibration_src_images, calibration_dst_images, (9, 6), logger_obj)
## Get calibration data
camera_matrix, distortion_coefficent = calibration_obj.get_calibration_parameters()


## Get Parameters
pipeline_params = Parameters(logger_obj, camera_matrix, distortion_coefficent)

# Get Pipeline object 
Runner = Pipeline(pipeline_params)



def main():
    # Images
    Runner.process_test_images(test_images_src)

    # Project vedio
    Runner.process_test_vedio(test_vedios_src, 'project_video', test_vedios_des)


if __name__ == "__main__":
    main()