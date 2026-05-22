import numpy as np
import cv2
from miniproject.simulation import MiniprojectSimulation
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression



class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep,
                                                    convergence_coefs=np.ones(6) * 50,
                                                    intrinsic_freqs=12*np.ones(6), 
                                                    intrinsic_amps=np.ones(6) * 4)

        self.prev_vision = None
        self.step_count = 0
    

        self.video_writer = None
        self.frame_size = None
        self.raw_video_writer = None
        self.raw_frame_size = None
        self.odor_drives = np.ones(2)
        self.obstacle_drives = np.ones(2)
        self.drives = [0,0]
        self.obstacle = False
        self.head_position = []

        # --- stuck mode ---
        self.is_stuck = False
        self.stuck_count = 0
        self.stuck_duration = 8000  # Nb of step for the stuck movement

        self.immobile_count = 0
        self.immobile_threshold = 1700  # Nb of step to enter to stuck mode 
        
        self.avoid_drives = None

    def step(self, sim: MiniprojectSimulation):

        #Store head position 
        self.head_position.append(self.get_head_position(sim))
        stuck = self.check_stuck()
        
        ###Olfaction###
        olfaction = sim.get_olfaction(sim.fly.name)
        olfaction_smooth = None

        if olfaction_smooth is None:
            olfaction_smooth = olfaction
        else:
            alpha = 0.001
            olfaction_smooth = (1-alpha) * olfaction_smooth + alpha * olfaction
        
        odor_drives = odor_intensity_to_control_signal(olfaction_smooth)

        ###Obstacle Avoidance### 
        if self.step_count % 200 == 0:
    
            raw_vision = sim.get_raw_vision(sim.fly.name)
            # self.plot_raw_vision(raw_vision)
            self.write_raw_vision_video(raw_vision, stuck)
            images = np.asarray(raw_vision)  # (2, H, W, 3)

            
            left_vis, skyline_L = self.full_process(images[0])
            right_vis, skyline_R = self.full_process(images[1])
            crop_left_processed = self.keep_middle_left(left_vis,threshold=2/5)
            crop_right_processed = self.keep_middle_right(right_vis,threshold=3/5)
            self.obstacle_drives, left_count, right_count, self.obstacle, _, _ = self.avoid_obstacle(crop_left_processed, 
                                                                                                     crop_right_processed, 
                                                                                                     skyline_L, 
                                                                                                     skyline_R,
                                                                                                     width_threshold_min=45,
                                                                                                     width_threshold_max=80)
            
            self.avoid_drives = dragonfly_avoidance_drives(raw_vision)

        ###Determine if there is a Dragonlfly
        if (self.avoid_drives!=None):
            print("Dragonfly")
            self.drives = self.avoid_drives
        ### Prioritize dragonfly avoidance on obstacle avoidance
        elif(self.obstacle):
            self.drives = self.obstacle_drives
        else:
            self.drives = odor_drives

        if stuck :
            self.immobile_count += 1
        else:
            self.immobile_count = 0

        if ((self.immobile_count >= self.immobile_threshold) and not(self.is_stuck)):
            self.is_stuck = True
            self.stuck_count = 0
            self.immobile_count = 0
            
            if(odor_drives[0]>odor_drives[1]):
                    self.stuck_drives = np.array([1.3, -1.3])
            else:
                self.stuck_drives = np.array([-1.3, 1.3])
            
        #if stuck, reverse the control signal and then turn around itself to try to get unstuck
        if self.is_stuck:

            self.stuck_count += 1
            self.drives = self.stuck_drives

            if(self.stuck_count < (self.stuck_duration*(3.0/5.0))):
                self.drives = np.array([-1.0, -1.0])
            else :
                self.drives = self.stuck_drives
                
            if self.stuck_count >= self.stuck_duration:
                # Reset the stuck count and switch back to normal mode
                self.stuck_count = 0
                self.is_stuck = False
                self.immobile_count = 0
                print(f"Fin Stuck Mode au step {self.step_count}. Et c'est reparti pour un touuuur")
        
        
        #If stuck all negative drives
        if self.is_stuck:
            min_drive = -2.0
        else:
            min_drive = 0.0

        self.drives = np.clip(self.drives, min_drive, 2.0) #clip for safety ?

        self.step_count += 1
        joint_angles, adhesion = self.turning_controller.step(self.drives)
        return joint_angles, adhesion



    # ==============================  


    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print("Saved video: optic_flow.mp4")
        if self.raw_video_writer is not None:
            self.raw_video_writer.release()
            print("Saved video: raw_vision")

   ##### Our Functions #####


    def check_stuck(self, N=1000, threshold=0.30):
        """
        Calculate the cumulative movement and determined if the fly is stuck
        """

        if len(self.head_position) < N:
            return False

        recent = np.array(self.head_position[-N:])

        if recent.shape[0] < 2:
            return False

        step_movements = np.diff(recent, axis=0)
        step_distances = np.linalg.norm(step_movements, axis=1)
        cumulative_movement = np.sum(step_distances)
        stuck = cumulative_movement < threshold

        return stuck




    ########### Vision ###########
       
    def keep_middle_right(self, img,threshold=1/3):
        """
        Keep only the left third of the right vision (center of the vision)
        """
        
        w = img.shape[1]
        cutoff = int(w * threshold)
        left_third = img[:, :cutoff, ...]  
        return left_third
    
    def keep_middle_left(self, img,threshold=2/3):
        """
        Keep only the right third of the left vision (center of the vision)
        """
        w = img.shape[1]
        cutoff = int(w * threshold)          
        right_third = img[:, cutoff:, ...]
        return right_third


    def filter_green(self, img, g_min=100, b_max=50, r_max=120):
        # Ensure uint8
        img = img.astype(np.uint8)

        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]

        mask = (g > g_min) & (b < b_max) #& (r < r_max)

        filtered = np.zeros_like(img)
        filtered[mask] = img[mask]

        return filtered

    
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

        sky_mask = ((S < 120) & (V > 80)).astype(np.uint8) * 255

        kernel = np.ones((7, 7), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

        points = []

        for x in range(w):
            ys = np.where(sky_mask[:, x] > 0)[0]
            if len(ys) > 0:
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

        y_line = y_line - 50 
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
    
    def draw_width(self, img, segment, color=(0, 0, 255)):
        if segment is None:
            return img
        y, x_min, x_max = segment
        img_out = img.copy()
        cv2.line(img_out, (x_min, y), (x_max, y), color, 2)
        return img_out
            

    ########### Avoid Obstacle ###########

    def avoid_obstacle(self, left_img, right_img, skyline_L, skyline_R,
                   width_threshold_min=40, width_threshold_max=80, k_turn=0.055): 
        """
        Deter
        
        """

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
        left_trigger = width_threshold_min < left_width < width_threshold_max
        right_trigger = width_threshold_min < right_width < width_threshold_max

        # Decision logic
        if left_trigger and right_trigger:
            obstacle = True
            if left_width > right_width:
                turn_strength = (left_width - width_threshold_min) * k_turn
                drives = np.array([1.0 + turn_strength, 1.0 - turn_strength])
            else:
                turn_strength = (right_width - width_threshold_min) * k_turn
                drives = np.array([1.0 - turn_strength, 1.0 + turn_strength])

        elif left_trigger:
            obstacle = True
            turn_strength = (left_width - width_threshold_min) * k_turn
            drives = np.array([1.0 + turn_strength, 1.0 - turn_strength])

        elif right_trigger:
            obstacle = True
            turn_strength = (right_width - width_threshold_min) * k_turn
            drives = np.array([1.0 - turn_strength, 1.0 + turn_strength])

        return drives, left_width, right_width, obstacle, left_seg, right_seg


    
   ########### Debug Video Creation ###########

    def write_raw_vision_video(self, raw_vision, stuck):

        images = np.asarray(raw_vision)  # (2, H, W, 3)
        #Crop vision to only keep center view for obstacle detection 
        crop_left = self.keep_middle_left(images[0],threshold=2/5)
        crop_right = self.keep_middle_right(images[1],threshold=3/5)

        # Filtered green images
        left_green = self.filter_green(crop_left)
        right_green = self.filter_green(crop_right)

        # Above skyline + green filtered
        left_processed, skyline_L = self.full_process(images[0])
        right_processed, skyline_R = self.full_process(images[1])

        crop_left_processed = self.keep_middle_left(left_processed,threshold=2/5)
        crop_right_processed = self.keep_middle_right(right_processed,threshold=3/5)


        # Original images with skyline
        left_original = self.draw_skyline(crop_left, skyline_L)
        right_original = self.draw_skyline(crop_right, skyline_R)

        drives, left_count, right_count, obstacle, left_seg, right_seg = self.avoid_obstacle(
            crop_left_processed,
             crop_right_processed, skyline_L, skyline_R
        )
        drives = np.clip(drives, -1.0, 2.5)
        left_original = self.draw_width(left_original, left_seg)
        right_original = self.draw_width(right_original, right_seg)

        top = np.concatenate([left_original, right_original], axis=1)
        middle = np.concatenate([crop_left_processed, crop_right_processed], axis=1)
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
        text4 = f"Stuck: {stuck}"
        text5 = f"is_stuck: {self.is_stuck}"
        text6 = f"immobile counts: {self.immobile_count}"

        cv2.putText(panel, text1, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(panel, text2, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(panel, text3, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(panel, text4, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(panel, text5, (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(panel, text6, (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame = np.concatenate([frame, panel], axis=1)

        if self.raw_video_writer is None:
            h, w, _ = frame.shape
            self.raw_frame_size = (w, h)

            self.raw_video_writer = cv2.VideoWriter(
                "raw_vision.webm",
                cv2.VideoWriter_fourcc(*"VP09"),
                30,
                self.raw_frame_size
            )

        self.raw_video_writer.write(frame)





    ########### Position of the Fly ###########

    def get_head_position(self, sim: MiniprojectSimulation):
        """
        Get the position of the head of the fly in the 3D space

        """

        body_segments = sim.fly.get_bodysegs_order()

        head_index = next(i for i, seg in enumerate(body_segments) if seg.name == 'c_head')

        all_positions = sim.get_body_positions(sim.fly.name)

        head_position = all_positions[head_index]

        return head_position
    
    
    ########### Plot Function ###########


    def plot_head_trajectory(self, sim: MiniprojectSimulation):
        
        if not self.head_position:
            print("Error Nothing to plot.")
            return

        positions = np.array(self.head_position)


        x_coords = positions[:, 0]  
        y_coords = positions[:, 1]  
        
        plt.figure(figsize=(8, 6))
    
        plt.plot(x_coords, y_coords, label="Head Trajectory", color="blue", linewidth=2)
        
        banana_xy = sim.world.banana_xy
        fly_xy = np.array([x_coords[-1], y_coords[-1]])
        print("final dist",np.linalg.norm(fly_xy - banana_xy))

        # Marque le point de départ et d'arrivée
        plt.scatter(x_coords[0], y_coords[0], color="green", edgecolors="black", s=100, label="Start", zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color="red", edgecolors="black", s=100, label="Finished", zorder=5)
        plt.scatter(banana_xy[0], banana_xy[1], color="yellow", edgecolors="black", s=100, label="Banana", zorder=5)
        # Habillage du graphique
        plt.title("Trajectory of the fly's head")
        plt.xlabel("Position X (mm)")
        plt.ylabel("Position Y (mm)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.axis("equal")  
        
        plt.show()


########### Odor ###########

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
    modulation_amount = np.abs(effective_bias_norm) * 1.0
    control_signal[side_to_modulate] -= modulation_amount
    return control_signal

########### Dragonfly ###########

def detect_red(raw_vision, red_ratio=0.6):
    """Red pixel fraction and horizontal center per eye.
    Returns [(area_L, xcenter_L), (area_R, xcenter_R)].
    xcenter is normalized [0, 1]. For the left eye, small = anterior.
    For the right eye, large = anterior."""
    results = []
    for eye_img in raw_vision:
        r = eye_img[..., 0].astype(np.float32)
        g = eye_img[..., 1].astype(np.float32)
        b = eye_img[..., 2].astype(np.float32)
        is_red = (r / (r + g + b + 1e-6)) > red_ratio
        area = is_red.mean()
        if area > 0:
            _, cols = np.where(is_red)
            xcenter = cols.mean() / eye_img.shape[1]
        else:
            xcenter = 0.5
        results.append((area, xcenter))
    return results


def dragonfly_avoidance_drives(raw_vision, looming_thr=0.0001, open_loop_thr=0.01):
    (area_l, xc_l), (area_r, xc_r) = detect_red(raw_vision)
    total = area_l + area_r

    if total < looming_thr:
        return None                        # no dragonfly → normal nav

    if total > open_loop_thr:
        return np.array([1.0, 1.0])        # open-loop → full forward, it overshoots behind us

    # Looming: turn to keep dragonfly at ~90° (xcenter ≈ 0.5 in the dominant eye)
    # Pick the eye that sees more red
    if area_l >= area_r:
        xcenter = xc_l
    else:
        xcenter = xc_r

    # Both eyes share the same rule:
    #   xcenter < 0.5 → dragonfly toward anterior → turn right (slow right)
    #   xcenter > 0.5 → dragonfly toward posterior → turn left (slow left)
    #   xcenter ≈ 0.5 → already perpendicular → full forward
    drives = np.array([1.0, 1.0])
    deviation = abs(xcenter - 0.5) * 2     # 0 = centered, 1 = at edge
    if deviation < 0.15:
        return drives                      # already roughly perpendicular

    turn_strength = np.clip(1 - deviation * 2, 0.3, 1.0)
    if xcenter < 0.5:
        return np.array([1.0, 0.3])          # slow right → turn right
    else:
        return np.array([0.3, 1.0])         # slow left → turn left

