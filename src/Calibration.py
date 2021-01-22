import os
import cv2
import numpy as np


class Calibration:
    def __init__(
        self,
        source_images_directory,
        destination_image_sub_directory,
        chessboard_shape,
        logger
    ):
        self.source_images_directory        = source_images_directory
        self.destination_image_sub_directory= destination_image_sub_directory
        self.chessboard_y, self.chessboard_x= chessboard_shape
        self.logger                         = logger
        self.name_list_of_boards            = os.listdir(self.source_images_directory)
        self.number_of_boards               = len(self.name_list_of_boards)
        self.image_size                     = None
        self.object_points = []
        self.image_points  = []
        self.camera_matrix, self.distortion_coefficient = \
            self.__calculate_calibration_parameters()

        

    def get_calibration_parameters(self):
        return self.camera_matrix, self.distortion_coefficient

    def __calculate_calibration_parameters(self):
        object_points = np.zeros((self.chessboard_x*self.chessboard_y, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:self.chessboard_x, 0:self.chessboard_y].T.reshape(-1, 2)
        
        for img_name in self.name_list_of_boards:
            # Read the image
            image_path = '{}/{}'.format(str(self.source_images_directory), str(img_name))
            image_obj  = cv2.imread(image_path)
            # Gray it
            gray_image = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)
            self.image_size = gray_image.shape[::-1]

            # Find its corners
            ret, corners = cv2.findChessboardCorners(gray_image, (self.chessboard_y, self.chessboard_x), None)

            if ret:
                self.object_points.append(object_points)
                self.image_points.append(corners)

                # save image with corners
                image = cv2.drawChessboardCorners(\
                    image_obj, \
                    (self.chessboard_y, self.chessboard_x), \
                    corners, \
                    ret)
                self.logger.save_image(str(self.destination_image_sub_directory), img_name, image)
            else:
                self.logger.log_error('Can not find all needed corners in {}'.format(str(img_name)))
        
        # Calibrate the camera
        calibration_parameters = \
            cv2.calibrateCamera(self.object_points, \
                self.image_points, \
                self.image_size, \
                None, None)
        # return onlt camera_matrix, and dis_coef
        return calibration_parameters[1], calibration_parameters[2]
            

    
