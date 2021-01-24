import os
import cv2

class Logger:
    '''
    Class responsible of logging messages/images for the user.
    '''
    def __init__(self, output_directory):
        '''
        :param: output_directory: root output directory for images.
        '''
        self.save_path = output_directory

    def log_error(self, error_message):
        '''
        Printing error messages to terminal

        :param  error_message:  String error message.
        :return: void
        '''
        print('[ERROR] {}'.format(str(error_message)))
    
    def log_info(self, error_message):
        '''
        Printing error messages to terminal

        :param  error_message:  String error message.
        :return: void
        '''
        print('[ INFO] {}'.format(str(error_message)))

    def save_image(self, sub_directory, file_name, image_obj):
        '''
        Printing/Saving an image to provided directory.
        
        :param  sub_directory: The subdirectory for the image to be saved under
        :param      file_name: The filename of the image
        :param          image: The image data/object
        :return: void
        '''
        directory = '{}/{}'.format(str(self.save_path), str(sub_directory))
        image_name= '{}/{}'.format(str(directory), str(file_name))
        self.log_info('Writing the image {} to {}.'.format(str(file_name), str(directory)))
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(image_name, image_obj)
        self.log_info('Done!')
        return