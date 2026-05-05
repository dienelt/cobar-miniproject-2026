import numpy as np
import cv2
from miniproject.simulation import MiniprojectSimulation
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression


def odor_intensity_to_control_signal(
    odor_intensities,
    attractive_gain=-500,
):
    attractive_intensities = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[5, 5]
    )
    attractive_bias = (
        attractive_gain
        * (attractive_intensities[0] - attractive_intensities[1])
        / attractive_intensities.mean()
        if attractive_intensities.mean() != 0
        else 0
    )
    
    
    effective_bias = attractive_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)

    control_signal = np.ones(2)
    side_to_modulate = int(effective_bias_norm > 0)
    modulation_amount = np.abs(effective_bias_norm) * 0.8
    control_signal[side_to_modulate] -= modulation_amount
    return control_signal

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep, intrinsic_freqs=np.ones(6) * 50)

        self.prev_vision = None

        #Video recording
        self.video_writer = None
        self.frame_size = None

        self.raw_video_writer = None
        self.raw_frame_size = None

    def step(self, sim: MiniprojectSimulation):

        olfaction = sim.get_olfaction(sim.fly.name)
        olfaction_smooth = None

        if olfaction_smooth is None:
            olfaction_smooth = olfaction
        else:
            alpha = 0.001
            olfaction_smooth = (1-alpha) * olfaction_smooth + alpha * olfaction
        
        odor_drives = odor_intensity_to_control_signal(olfaction_smooth)

        # implement your control algorithm here
        # get other observations as needed
        # drives = np.array([1.0, 1.0])  # replace with your control logic

        raw_vision = sim.get_raw_vision(sim.fly.name)
        # self.plot_raw_vision(raw_vision)
        self.write_raw_vision_video(raw_vision)
        images = np.asarray(raw_vision)  # (2, H, W, 3)
        left_vis, skyline_L = self.full_process(images[0])
        right_vis, skyline_R = self.full_process(images[1])
        obstacle_drives, left_count, right_count, obstacle, _, _ = self.avoid_obstacle(left_vis, right_vis, skyline_L, skyline_R)

        if(obstacle):
            drives = obstacle_drives
        else:
            drives = odor_drives
        # drives = odor_drives
        
        # print("odor drive : ", odor_drives, "  obstacle drive : ", obstacle_drives)
        # drives = obstacle_drives*0 + odor_drives*1.0
        # drives = np.clip(drives, 0.2, 1.8) #clip for safety ?

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

    # ==============================

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print("Saved video: optic_flow.mp4")
        if self.raw_video_writer is not None:
            self.raw_video_writer.release()
            print("Saved video: raw_vision.mp4")

    ###########

    def keep_top(self, img):
        """
        Keep only the top third of the image.
        """
        h = img.shape[0]
        cutoff = int(h * 0.33)
        top_third = img[: cutoff, ...]  # keep rows from 0 to h/3
        return top_third

    def filter_green(self, img, g_min=100, b_max=50, r_max=120):
        # Ensure uint8
        img = img.astype(np.uint8)

        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]

        # Threshold mask
        mask = (g > g_min) & (b < b_max) #& (r < r_max)

        # Apply mask
        filtered = np.zeros_like(img)
        filtered[mask] = img[mask]

        return filtered


    # def draw_skyline(self, img, color=(255, 0, 0), thickness=3):
    #     """
    #     Draw estimated skyline on RGB image.
    #     color is RGB, default red.
    #     """
    #     img_out = img.copy()
    #     y_line = self.estimate_skyline(img)

    #     for x, y in enumerate(y_line):
    #         cv2.circle(img_out, (int(x), int(y)), thickness, color, -1)

    #     return img_out
    
    def draw_skyline(self, img, y_line, color=(255, 0, 0), thickness=3):
        """
        Draw estimated skyline on RGB image.
        y_line: precomputed skyline (array of shape [width])
        """
        img_out = img.copy()

        for x, y in enumerate(y_line):
            cv2.circle(img_out, (int(x), int(y)), thickness, color, -1)

        return img_out

    def estimate_skyline(self, img):
        """
        Estimate a straight sky/ground line while ignoring pointy green obstacles.
        Returns y_line: array of shape (width,), giving skyline y for each x.
        """
        img = img.astype(np.uint8)
        h, w, _ = img.shape

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Sky/cloud mask: bright and low/medium saturation
        sky_mask = ((S < 120) & (V > 80)).astype(np.uint8) * 255

        kernel = np.ones((7, 7), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

        points = []

        for x in range(w):
            ys = np.where(sky_mask[:, x] > 0)[0]
            if len(ys) > 0:
                # lowest sky pixel = sky/ground boundary candidate
                points.append([x, ys.max()])

        points = np.array(points)

        # Fallback if sky detection fails
        if len(points) < 20:
            return np.ones(w) * int(h * 0.45)

        # Keep lower boundary candidates only
        # This rejects pointy green obstacles because they create very high boundary points
        y_values = points[:, 1]
        threshold = np.percentile(y_values, 60)
        points_filtered = points[y_values >= threshold]

        if len(points_filtered) < 20:
            points_filtered = points

        X = points_filtered[:, 0].reshape(-1, 1)
        y = points_filtered[:, 1]

        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=12,
                random_state=0
            )
            ransac.fit(X, y)

            x_line = np.arange(w).reshape(-1, 1)
            y_line = ransac.predict(x_line)

        except Exception:
            y_line = np.ones(w) * np.median(y)

        y_line = y_line - 50#20
        y_line = np.clip(y_line, 0, h - 1)
        return y_line


    def keep_above_skyline(self, img):
        """
        Keep only the image region above the estimated skyline.
        Everything below the skyline is set to black.
        """
        h, w, _ = img.shape
        y_line = self.estimate_skyline(img)

        mask = np.zeros((h, w), dtype=np.uint8)

        for x in range(w):
            y = int(y_line[x])
            mask[:y, x] = 1

        result = np.zeros_like(img)
        result[mask == 1] = img[mask == 1]

        return result, y_line


    def full_process(self, img):
        above_skyline, skyline = self.keep_above_skyline(img)
        filtered = self.filter_green(above_skyline)
        return filtered, skyline
        

    # def avoid_obstacle(self, left_img, right_img, threshold=2600, k_turn=0.00005): #22900 : doesn't detect

    #     drives = np.array([1.0, 1.0])
    #     obstacle = False

    #     # left_count = np.count_nonzero(left_img)
    #     # right_count = np.count_nonzero(right_img)

    #     left_count = np.count_nonzero(np.any(left_img > 0, axis=-1))
    #     right_count = np.count_nonzero(np.any(right_img > 0, axis=-1))

    #     if(left_count > threshold):
    #         turn_strength = left_count*k_turn
    #         obstacle = True
    #         drives = np.array([
    #             1.0 + turn_strength,  # left
    #             1.0 - turn_strength   # right
    #         ])

    #     elif (right_count > threshold):
    #         if(right_count > left_count):
    #             turn_strength = right_count*k_turn
    #             obstacle = True
    #             drives = np.array([
    #                 1.0 - turn_strength,  # left
    #                 1.0 + turn_strength   # right
    #             ])

    #     return drives, left_count, right_count, obstacle

    # def avoid_obstacle(self, left_img, right_img, skyline_L, skyline_R, height_threshold=170, k_turn=0.001): #160x255 (green dist to hotizon)
    #     drives = np.array([1.0, 1.0])
    #     obstacle = False

    #     def get_highest_green(img, skyline):
    #         mask = np.any(img > 0, axis=-1)
    #         ys, xs = np.where(mask)

    #         if len(ys) == 0:
    #             return None

    #         # Compute distance above skyline
    #         distances = []

    #         for y, x in zip(ys, xs):
    #             skyline_y = skyline[x]
    #             if y < skyline_y:  # above skyline
    #                 distances.append(skyline_y - y)

    #         if len(distances) == 0:
    #             return None

    #         return max(distances)  # highest relative height
        
    #     def get_green_stats(img, skyline):
    #         mask = np.any(img > 0, axis=-1)
    #         ys, xs = np.where(mask)

    #         if len(ys) == 0:
    #             return None, 0

    #         distances = []
    #         count = 0

    #         for y, x in zip(ys, xs):
    #             skyline_y = skyline[x]
    #             if y < skyline_y:  # above skyline
    #                 d = skyline_y - y
    #                 distances.append(d)
    #                 count += 1

    #         if len(distances) == 0:
    #             return None, 0

    #         return max(distances), count


    #     # left_dist = get_highest_green(left_img, skyline_L)
    #     # right_dist = get_highest_green(right_img, skyline_R)

    #     left_dist, left_count = get_green_stats(left_img, skyline_L)
    #     right_dist, right_count = get_green_stats(right_img, skyline_R)

    #     # # Convert to "height from bottom" (more intuitive)
    #     # h = left_img.shape[0]

    #     # left_height = h - left_top if left_top is not None else 0
    #     # right_height = h - right_top if right_top is not None else 0

    #     # Decide obstacle
    #     left_trigger = left_dist is not None and left_dist > height_threshold
    #     right_trigger = right_dist is not None and right_dist > height_threshold
    #     # if left_dist is not None and left_dist > height_threshold:
    #     #     obstacle = True
    #     #     turn_strength = (left_dist - height_threshold) * k_turn
    #     #     drives = np.array([
    #     #         1.0 + turn_strength,  # turn right
    #     #         1.0 - turn_strength
    #     #     ])

    #     # elif right_dist is not None and right_dist > height_threshold:
    #     #     obstacle = True
    #     #     turn_strength = (right_dist - height_threshold) * k_turn
    #     #     drives = np.array([
    #     #         1.0 - turn_strength,
    #     #         1.0 + turn_strength   # turn left
    #     #     ])

    #     if left_trigger and right_trigger:
    #         obstacle = True

    #         if left_count > right_count:
    #             turn_strength = (left_dist - height_threshold) * k_turn
    #             drives = np.array([1.0 + turn_strength, 1.0 - turn_strength])
    #         else:
    #             turn_strength = (right_dist - height_threshold) * k_turn
    #             drives = np.array([1.0 - turn_strength, 1.0 + turn_strength])

    #     elif left_trigger:
    #         obstacle = True
    #         turn_strength = (left_dist - height_threshold) * k_turn
    #         drives = np.array([1.0 + turn_strength, 1.0 - turn_strength])

    #     elif right_trigger:
    #         obstacle = True
    #         turn_strength = (right_dist - height_threshold) * k_turn
    #         drives = np.array([1.0 - turn_strength, 1.0 + turn_strength])

    #     return drives, left_dist, right_dist, obstacle



    ###############

    def avoid_obstacle(self, left_img, right_img, skyline_L, skyline_R,
                   width_threshold=80, k_turn=0.055): #vid 27 : 100, 0.01 #vid 28 : 100, 0.035

        drives = np.array([1.0, 1.0])
        obstacle = False

        def get_green_width(img, skyline):
            mask = np.any(img > 0, axis=-1)
            ys, xs = np.where(mask)

            if len(ys) == 0:
                return 0, 0, (0,0,0)  # width, count

            # Keep only pixels above skyline
            sky_vals = skyline[xs]
            valid = ys < sky_vals

            if not np.any(valid):
                return 0, 0, (0,0,0)

            ys_valid = ys[valid]
            xs_valid = xs[valid]

            max_width = 0
            best_segment = None

            # Compute width per row
            for y in np.unique(ys_valid):
                xs_row = xs_valid[ys_valid == y]

                if len(xs_row) > 1:
                    xs_sorted = np.sort(xs_row)

                    # Find breaks where pixels are not consecutive
                    splits = np.where(np.diff(xs_sorted) > 1)[0] + 1
                    segments = np.split(xs_sorted, splits)

                    for seg in segments:
                        if len(seg) > 1:
                            x_min = seg[0]
                            x_max = seg[-1]
                            width = x_max - x_min

                            if width > max_width:
                                max_width = width
                                best_segment = (y, x_min, x_max)

            count = len(xs_valid)
            return max_width, count, best_segment

        # Compute widths
        left_width, left_count, left_seg = get_green_width(left_img, skyline_L)
        right_width, right_count, right_seg = get_green_width(right_img, skyline_R)

        # Trigger conditions
        left_trigger = left_width > width_threshold
        right_trigger = right_width > width_threshold

        # Decision logic
        if left_trigger and right_trigger:
            obstacle = True
            if left_width > right_width:
                turn_strength = (left_width - width_threshold) * k_turn
                drives = np.array([1.0 + turn_strength, 1.0 - turn_strength])
            else:
                turn_strength = (right_width - width_threshold) * k_turn
                drives = np.array([1.0 - turn_strength, 1.0 + turn_strength])

        elif left_trigger:
            obstacle = True
            turn_strength = (left_width - width_threshold) * k_turn
            drives = np.array([1.0 + turn_strength, 1.0 - turn_strength])

        elif right_trigger:
            obstacle = True
            turn_strength = (right_width - width_threshold) * k_turn
            drives = np.array([1.0 - turn_strength, 1.0 + turn_strength])

        return drives, left_width, right_width, obstacle, left_seg, right_seg


    ###############
    def draw_width(self, img, segment, color=(0, 0, 255)):
        if segment is None:
            return img
        y, x_min, x_max = segment
        img_out = img.copy()
        cv2.line(img_out, (x_min, y), (x_max, y), color, 2)
        return img_out


    def write_raw_vision_video(self, raw_vision):
        images = np.asarray(raw_vision)  # (2, H, W, 3)

        # Filtered green images
        left_green = self.filter_green(images[0])
        right_green = self.filter_green(images[1])
        # print("size : ", left_green.shape)

        # Above skyline + green filtered
        left_processed, skyline_L = self.full_process(images[0])
        right_processed, skyline_R = self.full_process(images[1])

        # Original images with skyline
        left_original = self.draw_skyline(images[0], skyline_L)
        right_original = self.draw_skyline(images[1], skyline_R)

        drives, left_count, right_count, obstacle, left_seg, right_seg = self.avoid_obstacle(
            left_processed,
            right_processed, skyline_L, skyline_R
        )

        left_original = self.draw_width(left_original, left_seg)
        right_original = self.draw_width(right_original, right_seg)

        top = np.concatenate([left_original, right_original], axis=1)
        middle = np.concatenate([left_processed, right_processed], axis=1)
        bottom = np.concatenate([left_green, right_green], axis=1)

        frame = np.concatenate([top, middle, bottom], axis=0)

        # RGB -> BGR for OpenCV video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape
        panel_width = 450
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

        text1 = f"L drive: {drives[0]:.2f} | R drive: {drives[1]:.2f}"
        text2 = f"L green: {left_count:.2f} | R green: {right_count:.2f}"
        text3 = f"Obstacle: {obstacle}"

        cv2.putText(panel, text1, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(panel, text2, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(panel, text3, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame = np.concatenate([frame, panel], axis=1)

        if self.raw_video_writer is None:
            h, w, _ = frame.shape
            self.raw_frame_size = (w, h)

            self.raw_video_writer = cv2.VideoWriter(
                "raw_vision.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                self.raw_frame_size
            )

        self.raw_video_writer.write(frame)
