'''
import numpy as np
from miniproject.simulation import MiniprojectSimulation

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        print("iniiiit")

        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        # get other observations as needed
        drives = np.array([1.0, 1.0])  # replace with your control logic
        print("stepping of drives : ", drives)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

'''

import numpy as np
import cv2
from miniproject.simulation import MiniprojectSimulation
import matplotlib.pyplot as plt

from flygym.vision.retina import Retina
retina = Retina()
ommatidia_id_map = retina.ommatidia_id_map

optic_flow_kws = dict(
    flow=None,
    pyr_scale=0.5,
    levels=2,
    winsize=3,
    iterations=2,
    poly_n=5,
    poly_sigma=1.1,
    flags=0,
)

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
        left_vis = self.full_process(images[0])
        right_vis = self.full_process(images[1])
        left_vis, right_vis = self.keep_cropped(left_vis, right_vis)
        obstacle_drives, left_count, right_count, obstacle = self.avoid_obstacle(left_vis, right_vis)

        if(obstacle):
            drives = obstacle_drives
        else:
            drives = odor_drives
        
        # print("odor drive : ", odor_drives, "  obstacle drive : ", obstacle_drives)
        # drives = obstacle_drives*0 + odor_drives*1.0
        drives = np.clip(drives, 0.2, 1.8) #clip for safety ?

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

    # ==============================
    # VIDEO
    # ==============================
    def flow_to_image(self, flow):
        vmax = np.max(np.abs(flow)) + 1e-6
        flow_norm = (flow / vmax + 1) / 2
        flow_img = (flow_norm * 255).astype(np.uint8)
        flow_img = cv2.applyColorMap(flow_img, cv2.COLORMAP_TURBO)
        return flow_img

    def write_video_frame(self, vision, flow):

        # build vision frame (top)
        # vision_frame = np.concatenate(vision, axis=1)
        # vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_GRAY2BGR)

        # build flow frame (bottom)
        frame = np.concatenate(
            [self.flow_to_image(f) for f in flow],
            axis=1
        )

        # stack vertically
        # frame = np.vstack([vision_frame, flow_frame])

        # init writer once
        if self.video_writer is None:
            h, w, _ = frame.shape
            self.frame_size = (w, h)

            self.video_writer = cv2.VideoWriter(
                "optic_flow.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                self.frame_size
            )

        self.video_writer.write(frame)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print("Saved video: optic_flow.mp4")
        if self.raw_video_writer is not None:
            self.raw_video_writer.release()
            print("Saved video: raw_vision.mp4")

    ###########

    def keep_cropped(self, img_L, img_R):
        h, w = img_L.shape[:2]
        h_cutoff = int(h * 0.33)
        # w_left = int(w * (1/6))
        # w_right = int(w * (5/6))
        w_cutoff = int(w*0.4)

        left_cropped = img_L[:h_cutoff, w_cutoff:]
        right_cropped = img_R[:h_cutoff, :w_cutoff]

        # cropped = img[:h_cutoff, w_left:w_right]
        return left_cropped, right_cropped

    def filter_green(self, img, g_min=170, b_max=50, r_max=120):
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
    
    def full_process(self, img, g_min=170, b_max=50, r_max=120):
        filtered = self.filter_green(img, g_min, b_max, r_max)
        return filtered 
    

    def avoid_obstacle(self, left_img, right_img, threshold=2600, k_turn=0.00005): #22900 : doesn't detect

        drives = np.array([1.0, 1.0])
        obstacle = False

        # left_count = np.count_nonzero(left_img)
        # right_count = np.count_nonzero(right_img)

        left_count = np.count_nonzero(np.any(left_img > 0, axis=-1))
        right_count = np.count_nonzero(np.any(right_img > 0, axis=-1))

        if(left_count > threshold):
            turn_strength = left_count*k_turn
            obstacle = True
            drives = np.array([
                1.0 + turn_strength,  # left
                1.0 - turn_strength   # right
            ])

        elif (right_count > threshold):
            if(right_count > left_count):
                turn_strength = right_count*k_turn
                obstacle = True
                drives = np.array([
                    1.0 - turn_strength,  # left
                    1.0 + turn_strength   # right
                ])

        return drives, left_count, right_count, obstacle



    ###############

    # ==============================
    # Optic flow
    # ==============================
    def compute_optic_flow_x(self, pre_imgs, post_imgs):
                
        return np.array(
            [
                cv2.calcOpticalFlowFarneback(pre_img, post_img, **optic_flow_kws)[..., 0]
                for pre_img, post_img in zip(pre_imgs, post_imgs)
            ]
        )
    
    def preprocess_raw_vision(self, raw_vision):
        images = np.asarray(raw_vision)  # (2, 512, 450, 3)

        processed = []

        for img in images:
            # RGB -> grayscale
            img = self.filter_green(img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Resize for faster optical flow
            gray = cv2.resize(gray, (128, 128))

            processed.append(gray)

        return np.array(processed)
    
    def plot_raw_vision(self, raw_vision):

        images = np.asarray(raw_vision)

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(images[0])
        plt.title("Left eye")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(images[1])
        plt.title("Right eye")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


    def pad_to_width(self, img, target_w):
        h, w = img.shape[:2]
        pad_total = target_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        return np.pad(
            img,
            ((0, 0), (pad_left, pad_right), (0, 0)),
            mode='constant'
        )
    
    def write_raw_vision_video(self, raw_vision):
        images = np.asarray(raw_vision)  # (2, H, W, 3)

        # left = images[0]
        # right = images[1]
        left = self.filter_green(images[0])
        right = self.filter_green(images[1])
        left_2 = images[0]
        right_2 = images[1]
        left_3 = self.full_process(images[0])
        right_3 = self.full_process(images[1])
        left_3, right_3 = self.keep_cropped(left_3, right_3)

        drives, left_count, right_count, _ = self.avoid_obstacle(left_3, right_3)


        top = np.concatenate([left_2, right_2], axis=1)   # original
        bottom = np.concatenate([left, right], axis=1)    # filtered
        bottom_2 = np.concatenate([left_3, right_3], axis=1)   # original
        bottom_2 = self.pad_to_width(bottom_2, top.shape[1])

        frame = np.concatenate([top, bottom_2, bottom], axis=0)

        # OpenCV expects BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape
        panel_width = 600
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)  # black panel
        # text = f"L: {drives[0]:.2f}  R: {drives[1]:.2f}"
        text = f"L: {drives[0]:.2f} ({left_count})  R: {drives[1]:.2f} ({right_count})"
        cv2.putText(
            panel,
            text,
            (10, 50),  # position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),  # green text
            2,
            cv2.LINE_AA
        )

        frame = np.concatenate([frame, panel], axis=1)

        # init writer once
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