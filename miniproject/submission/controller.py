import numpy as np
import cv2
from miniproject.simulation import MiniprojectSimulation
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression



#CPG Parameters 
wave_phase_biases = np.array(
    [
        [0, 1, 2, 3, 4, 5],
        [5, 0, 1, 2, 3, 4],
        [4, 5, 0, 1, 2, 3],
        [3, 4, 5, 0, 1, 2],
        [2, 3, 4, 5, 0, 1],
        [1, 2, 3, 4, 5, 0],
    ]
) * (2 * np.pi / 6)

wave_coupling_weights = (wave_phase_biases > 0).astype(float) * 10.0


class Controller:
    def __init__(self, sim: MiniprojectSimulation):

        from flygym.examples.locomotion import TurningController

        #self.turning_controller = TurningController(sim.timestep, intrinsic_freqs=np.ones(6) * 20, intrinsic_amps=np.ones(6) * 4)
        self.turning_controller = TurningController(sim.timestep, 
                                                    intrinsic_freqs=np.ones(6) * 12,
                                                    phase_biases=wave_phase_biases,
                                                    coupling_weights=wave_coupling_weights,
                                                    convergence_coefs=np.ones(6) * 50
                                                    )
        #self.turning_controller = TurningController(sim.timestep)

        self.prev_vision = None
        self.step_count = 0
        #Video recording
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
        self.stuck_duration = 9000  # Nombre de steps pendant lesquels on recule (arbitraire as fuck)

        self.immobile_count = 0
        self.immobile_threshold = 1700  # Nombre de steps immobiles max avant de trigger le stuck mode
        # self.window_size = 500 # fenêtre glissante
        # self.movement_threshold = 0.05  # Distance min (en mm) pour considérer que la mouche a bougé

    def step(self, sim: MiniprojectSimulation):
        self.head_position.append(self.get_head_position(sim))

        stuck = self.check_stuck()
        
        olfaction = sim.get_olfaction(sim.fly.name)
        olfaction_smooth = None

        if olfaction_smooth is None:
            olfaction_smooth = olfaction
        else:
            alpha = 0.001
            olfaction_smooth = (1-alpha) * olfaction_smooth + alpha * olfaction
        
        odor_drives = odor_intensity_to_control_signal(olfaction_smooth)

        

        if self.step_count % 200 == 0:
            #print("vision")
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

        if(self.obstacle):
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
                    self.stuck_drives = np.array([1.8, -1.8])
            else:
                self.stuck_drives = np.array([-1.0, 1.0])
            
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
        
        # # print("odor drive : ", odor_drives, "  obstacle drive : ", obstacle_drives)
        # # drives = obstacle_drives*0 + odor_drives*1.0

        if self.is_stuck: 
            min_drive = -2.0 #1.0
        else:
            min_drive = 0.0

        self.drives = np.clip(self.drives, min_drive, 2.0) #clip for safety ?
        #self.drives = np.clip(self.drives, -1.0, 2.5) #clip for safety ?


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

    ###########

    # def check_stuck(self, N=1000, threshold=0.5):
    #     # Need enough history
    #     if len(self.head_position) < N:
    #         return False

    #     recent = np.array(self.head_position[-N:])

    #     if recent.shape[0] < 2:
    #         return False

    #     # Net displacement: distance from the start of the window to the end
    #     net_displacement = np.linalg.norm(recent[-1] - recent[0])

    #     # Consider stuck if the net distance covered over N steps is below threshold
    #     stuck = net_displacement < threshold

    #     return stuck


    def check_stuck(self, N=1000, threshold=0.30):
        # Need enough history
        if len(self.head_position) < N:
            return False

        recent = np.array(self.head_position[-N:])

        if recent.shape[0] < 2:
            return False

        # Compute displacement between consecutive positions
        step_movements = np.diff(recent, axis=0)

        # Distance traveled at each step
        step_distances = np.linalg.norm(step_movements, axis=1)

        # Total movement over the window
        cumulative_movement = np.sum(step_distances)

        # Consider stuck if total movement is below threshold
        stuck = cumulative_movement < threshold

        return stuck

    def keep_top(self, img):
        """
        Keep only the top third of the image.
        """
        h = img.shape[0]
        cutoff = int(h * 0.33)
        top_third = img[: cutoff, ...]  # keep rows from 0 to h/3
        return top_third
    
    def keep_middle_right(self, img,threshold=1/3):
        """
        Keep only the left third of the right vision
        """
        
        w = img.shape[1]
        cutoff = int(w * threshold)
        left_third = img[:, :cutoff, ...]  
        return left_third
    
    def keep_middle_left(self, img,threshold=2/3):
        """
        Keep only the right third of the left vision
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
        


    ###############

    def avoid_obstacle(self, left_img, right_img, skyline_L, skyline_R,
                   width_threshold_min=40, width_threshold_max=80, k_turn=0.055): #vid 27 : 100, 0.01 #vid 28 : 100, 0.035 #before at 0.055

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


    ###############
    def draw_width(self, img, segment, color=(0, 0, 255)):
        if segment is None:
            return img
        y, x_min, x_max = segment
        img_out = img.copy()
        cv2.line(img_out, (x_min, y), (x_max, y), color, 2)
        return img_out


    def write_raw_vision_video(self, raw_vision, stuck):
        images = np.asarray(raw_vision)  # (2, H, W, 3)
       
        crop_left = self.keep_middle_left(images[0],threshold=2/5)
        crop_right = self.keep_middle_right(images[1],threshold=3/5)
        # Filtered green images
        left_green = self.filter_green(crop_left)
        right_green = self.filter_green(crop_right)
        # print("size : ", left_green.shape)

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


    def get_head_position(self, sim: MiniprojectSimulation):
        # 1. Tu récupères l'ordre des segments de la mouche
        body_segments = sim.fly.get_bodysegs_order()

        # 2. Tu trouves l'indice qui correspond à la tête ('c_head')
        head_index = next(i for i, seg in enumerate(body_segments) if seg.name == 'c_head')

        # 3. Tu récupères toutes les positions 3D en utilisant le NOM de la mouche (str)
        all_positions = sim.get_body_positions(sim.fly.name)

        # 4. Tu extrais la position de la tête grâce à son indice
        head_position = all_positions[head_index]

        return head_position
    

    def plot_head_trajectory(self, sim: MiniprojectSimulation):
        
        # Si la liste est vide (la simulation n'a pas tourné), on évite le crash
        if not self.head_position:
            print("Erreur : Aucun historique de position à tracer.")
            return

        # On convertit la liste en array NumPy pour extraire facilement les colonnes
        positions = np.array(self.head_position)


        x_coords = positions[:, 0]  # Première colonne = X
        y_coords = positions[:, 1]  # Deuxième colonne = Y
        
        plt.figure(figsize=(8, 6))
    
        # Trace la ligne de la trajectoire
        plt.plot(x_coords, y_coords, label="Trajectoire de la tête", color="blue", linewidth=2)
        #banana [-19.36779711 -23.66539834]
        
        #banana_xy = np.array([30.07098571432481 , -6.076987186520356])
        #banana_xy = np.array([30.07098571432481 , -6.076987186520356]) #seed 67
        #banana_xy = np.array([-10.355681357019149 , 28.906774050637956])  #seed 777
        banana_xy = sim.world.banana_xy
        fly_xy = np.array([x_coords[-1], y_coords[-1]])
        print("final dist",np.linalg.norm(fly_xy - banana_xy))

        # Marque le point de départ et d'arrivée
        plt.scatter(x_coords[0], y_coords[0], color="green", edgecolors="black", s=100, label="Départ", zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color="red", edgecolors="black", s=100, label="Arrivée", zorder=5)
        plt.scatter(banana_xy[0], banana_xy[1], color="yellow", edgecolors="black", s=100, label="Banana", zorder=5)
        # Habillage du graphique
        plt.title("Trajectoire de la tête de la mouche (Plan X-Y)")
        plt.xlabel("Position X (mm)")
        plt.ylabel("Position Y (mm)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.axis("equal")  # Très important pour ne pas déformer les virages de la mouche !
        
        plt.show()


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