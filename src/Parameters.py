import numpy as np

class Parameters:
    def __init__(self, logger, camera_matrix, distor_coeff):
        self.camera_matrix     = camera_matrix
        self.distor_coeff      = distor_coeff
        self.logging           = logger

        self.white_lane_hsv, \
        self.yellow_lane_hsv   = self.__get_lane_hsv_ranges()
        self.sobel_kernal_size = 7
        self.gradient_thrs_x   = (  0, 30)
        self.gradient_thrs_y   = ( 20, 90)
        self.gradient_thrs_mag = (  0, 10)
        self.gradient_thrs_dir = (  0, np.pi/4)
        self.ratio             = ( 10, 12)
        self.scale             =   10
        self.offset            = (600, 300)
        self.source_points     = self.__get_source_points()
        self.destination_points= self.__get_destination_points()
        self.window_marg       = 20
        self.window_min        = 5
        self.region            = self.__get_region()


    def __get_lane_hsv_ranges(self):
        yellow_lower_bound = np.array([int(0.2 * 255), int(0.3  * 255), int(0.10 * 255)], dtype="uint8")
        yellow_upper_bound = np.array([int(0.6 * 255), int(0.8  * 255), int(0.90 * 255)], dtype="uint8")

        white_lower_bound  = np.array([int(0.0 * 255), int(0.0  * 255), int(0.80 * 255)], dtype="uint8")
        white_upper_bound  = np.array([int(1.0 * 255), int(0.10 * 255), int(1.0  * 255)], dtype="uint8")
        return  ( white_lower_bound, white_upper_bound), \
                (yellow_lower_bound, yellow_upper_bound)

    def __get_source_points(self):
        return np.array([ \
            [540, 488],[750, 488],[777, 508],[507, 508]], dtype=np.float32)

    def __get_destination_points(self):
        return np.array([ \
            [self.offset[0]                             , self.offset[1]],
            [self.offset[0] + self.ratio[0] * self.scale, self.offset[1]],
            [self.offset[0] + self.ratio[0] * self.scale, self.offset[1] + self.ratio[1] * self.scale],
            [self.offset[0]                             , self.offset[1] + self.ratio[1] * self.scale]], dtype=np.float32)
    
    def __get_region(self):
        return np.array([[580, 435], [700, 435], [1100, 660], [190, 660]])