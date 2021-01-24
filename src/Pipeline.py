import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip



class Pipeline:
    ''' Constructor '''
    def __init__(self, parameters):
        self.params  = parameters
        self.enable_image_writing = True 

    

    def __undistort(self, image, image_name):
        """
        Undistorts the image using the calibration parameters
        :param      image:  The image data
        :return:            An undistorted image
        """
        image = cv2.undistort( image, self.params.camera_matrix, self.params.distor_coeff)
        if self.enable_image_writing == True:
            self.params.logging.save_image('Undistort', image_name, image)
        return image

    def __color_threshold(self, image, image_name):
        """
        Removes pixels that are not within the color ranges
        :param image: 
        :return: 
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        white_mask = cv2.inRange(
            image,
            lowerb=self.params.white_lane_hsv[0],
            upperb=self.params.white_lane_hsv[1]
        )

        yellow_mask = cv2.inRange(
            image,
            lowerb=self.params.yellow_lane_hsv[0],
            upperb=self.params.yellow_lane_hsv[1]
        )
        image = cv2.bitwise_or(
            white_mask,
            yellow_mask
        )
        if self.enable_image_writing == True:
            self.params.logging.save_image('Color', image_name, image)

        return image

    def __edge_detection(self, image, image_name):
        # Define a function that takes an image, gradient orientation,
        # and threshold min / max values.
        def abs_sobel_thresh(img, orient='x', thresh=None):
            thresh_min = thresh[0]
            thresh_max = thresh[1]
            # Convert to grayscale
            gray = img
            # Apply x or y gradient with the OpenCV Sobel() function
            # and take the absolute value
            if orient == 'x':
                abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.params.sobel_kernal_size))
            elif orient == 'y':
                abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.params.sobel_kernal_size))
            else:
                raise Exception('Invalid `orient`')
            # Rescale back to 8 bit integer
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            # Create a copy and apply the threshold
            binary_output = np.zeros_like(scaled_sobel)
            # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
            binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
            # Return the result
            return binary_output

        # Define a function to return the magnitude of the gradient
        # for a given sobel kernel size and threshold values
        def mag_thresh(img):
            # Convert to grayscale
            gray = img
            # Take both Sobel x and y gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.params.sobel_kernal_size)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.params.sobel_kernal_size)
            # Calculate the gradient magnitude
            gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
            # Rescale to 8 bit
            scale_factor = np.max(gradmag) / 255
            gradmag = (gradmag / scale_factor).astype(np.uint8)
            # Create a binary image of ones where threshold is met, zeros otherwise
            binary_output = np.zeros_like(gradmag)
            binary_output[(gradmag >= self.params.gradient_thrs_mag[0]) & (gradmag <= self.params.gradient_thrs_mag[1])] = 1

            # Return the binary image
            return binary_output

        def dir_threshold(image):
            # Grayscale
            # Calculate the x and y gradients
            gray = image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.params.sobel_kernal_size)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.params.sobel_kernal_size)
            # Take the absolute value of the gradient direction,
            # apply a threshold, and create a binary image result
            absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
            binary_output = np.zeros_like(absgraddir)
            binary_output[(absgraddir >= self.params.gradient_thrs_dir[0]) & (absgraddir <= self.params.gradient_thrs_dir[1])] = 1

            # Return the binary image
            return binary_output

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image, orient='x', thresh=self.params.gradient_thrs_x)
        grady = abs_sobel_thresh(image, orient='y', thresh=self.params.gradient_thrs_y)
        mag_binary = mag_thresh(image)
        dir_binary = dir_threshold(image)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 0) & (grady == 0)) | ((mag_binary == 0) & (dir_binary == 0))] = 255
        
        if self.enable_image_writing == True:
            self.params.logging.save_image('Binary', image_name, combined)

        return combined

    def __perspective_transform(self, image, image_name, source_image):
        """
        Transforms the image to a birds-eye view
        :param image: 
        :param source_image: 
        :return: 
        """
        image_shape = (image.shape[1], image.shape[0])

        destination_points = self.params.destination_points

        transformation_matrix = cv2.getPerspectiveTransform(self.params.source_points, destination_points)
        reverse_transformation_matrix = cv2.getPerspectiveTransform(destination_points, self.params.source_points)

        def warp(matrix, image, image_shape):
            return cv2.warpPerspective(image, M=matrix, dsize=image_shape, flags=cv2.INTER_LINEAR)

        image = warp(transformation_matrix, image, image_shape)

        perspective_image = warp(transformation_matrix, source_image, image_shape)

        if self.enable_image_writing == True:
            self.params.logging.save_image('Perspective', image_name, perspective_image)

        return image, reverse_transformation_matrix

    def __lane_pixels(self, image, image_name):
        """
        Finds the pixels associated to the lane using a sliding window
        :param image: 
        :return: 
        """

        binary_warped = image

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        half = int(binary_warped.shape[0] / 2)
        histogram = np.sum(binary_warped[half:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.params.window_marg
        # Set minimum number of pixels found to recenter window
        minpix = self.params.window_min
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if self.enable_image_writing == True:
            self.params.logging.save_image('Lane_pixels', image_name, out_img)

        return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty

    def __annotate_lane(self, source_image, warped_image, reverse_matrix, left_fitx, right_fitx, ploty):
        """
        Draws the detected region of the lane lines
        :param source_image: 
        :param warped_image: 
        :param reverse_matrix: Used for undistorting the image
        :param left_fitx: 
        :param right_fitx: 
        :param ploty: 
        :return: 
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = warp_zero

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, reverse_matrix, (source_image.shape[1], source_image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(source_image, 1, newwarp, 0.3, 0)

    def __lane_stats(self, image_shape, left_fit, right_fit, ploty, leftx, rightx, lefty, righty):
        """
        Generates stats about the position and curvature of the lane
        :param image_shape: 
        :param left_fit: 
        :param right_fit: 
        :param ploty: 
        :param leftx: 
        :param rightx: 
        :param lefty: 
        :param righty: 
        :return: 
        """
        y_eval = image_shape[1]

        ym_per_pix = 12.0 / (self.params.destination_points[3][1] - self.params.destination_points[0][1])
        xm_per_pix = 10.0 / (self.params.destination_points[1][0] - self.params.destination_points[0][0])

        left = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        # feetPerPixel = 12 / (rightx - leftx)

        lane_midpoint_px = (right + left) / 2
        camera_midpoint_px = image_shape[0] / 2

        offset_from_center = np.abs(lane_midpoint_px - camera_midpoint_px) * xm_per_pix

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        return offset_from_center, np.min([left_curverad, right_curverad])

    def __display_numbers(self, image, offset_from_center, curverad):
        cv2.putText(image, 'Offset: {:.1f} Curve radius: {:.1f}'.format(offset_from_center, curverad), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    def __region_of_interest(self, img, img_name, source_image):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """

        points = np.array([
            self.params.region[3],
            self.params.region[0],
            self.params.region[1],
            self.params.region[2],
        ])

        vertices = np.int32([points])

        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        annotated_image = source_image.copy()
        cv2.fillPoly(annotated_image, vertices, (255, 255, 255))

        if self.enable_image_writing == True:
            self.params.logging.save_image('Region', img_name, annotated_image)

        return masked_image

    def __process(self, image, image_name=''):
        """
        Find the lane lines in the provided image
        :param image: 
        :return: 
        """
        source_image = image.copy()

        image = self.__region_of_interest(image, image_name, source_image)

        image = self.__undistort(image, image_name)

        image, reverse = self.__perspective_transform(image, image_name, source_image)

        image = self.__color_threshold(image, image_name)
        image = self.__edge_detection(image , image_name)

        image, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx,lefty, righty = \
            self.__lane_pixels(image, image_name)

        image = self.__annotate_lane(source_image=source_image, 
                                     warped_image=image, 
                                     reverse_matrix=reverse,
                                     left_fitx=left_fitx, 
                                     right_fitx=right_fitx, 
                                     ploty=ploty)
        offset_from_center, curverad = \
            self.__lane_stats(  image_shape=(source_image.shape[1], source_image.shape[0]),
                                left_fit=left_fit, 
                                right_fit=right_fit, 
                                ploty=ploty, 
                                leftx=leftx, 
                                rightx=rightx, 
                                lefty=lefty, 
                                righty=righty)

        self.__display_numbers(image, offset_from_center=offset_from_center, curverad=curverad)

        return image

    
    def process_test_images(self, test_images_dir):
        images_list = os.listdir(str(test_images_dir))

        for image in images_list:
            image_name = image
            image_obj  = cv2.imread('{}/{}'.format(str(test_images_dir), str(image_name)))
            proc_image = self.__process(image_obj, image_name)
            cv2.imwrite('./output_images/{}'.format(str(image_name)), proc_image)


    def process_test_vedio(self, clip_dir, clip_name, output_path):
        self.enable_image_writing = False
        clip = VideoFileClip(filename="{}/{}.mp4".format(str(clip_dir), str(clip_name)))
        self.params.logging.log_info('Start processing {}'.format(str(clip_name)))
        def process(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.__process(image, '0_frame.png')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        clip = clip.fl_image(process)
        clip.write_videofile(filename='{}/{}_output.mp4'.format(str(output_path), str(clip_name)), audio=False)
        self.params.logging.log_info('Processing Finished!')
