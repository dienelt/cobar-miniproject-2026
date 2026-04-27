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

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

        self.prev_vision = None

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        ommatidia_readouts = sim.get_ommatidia_readouts(sim.fly.name)
        curr_vision = self.preprocess_fly_vision(ommatidia_readouts)
        # get other observations as needed
        # drives = np.array([1.0, 1.0])  # replace with your control logic

        if self.prev_vision is not None:
            flow = self.compute_optic_flow_x(self.prev_vision, curr_vision)
            mid = flow.shape[-1] // 2
            left_flow = np.mean(flow[..., :mid])
            right_flow = np.mean(flow[..., mid:])

            #turning rule: more flow = closer obstacle -> turn away
            turn = right_flow - left_flow

            # print("left flow : ", left_flow, "right flow : ", right_flow, "turn : ", turn)

            # #To test to be more smooth maybe
            # turn = 0.8 * turn + 0.2 * self.prev_turn
            # self.prev_turn = turn

            drives = np.array([
                1.0 - turn,  #left
                1.0 + turn   #right
            ])
            drives = np.clip(drives, 0.2, 1.5) #clip for safety ?

        else:
            drives = np.array([1.0, 1.0])

        self.prev_vision = curr_vision

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
    
######Vision :
    def crop_hex_to_rect(self, visual_input): #Doing image processing with hexagonal images is challenging, so we will resize the fly's vision to a rectangular image
        """Extract a rectangular crop from the hexagonal ommatidium layout."""
        rows = [np.unique(row) for row in ommatidia_id_map]
        max_width = max(len(row) for row in rows)
        rows = np.array([row for row in rows if len(row) == max_width])[:, 1:] - 1
        cols = [np.unique(col) for col in rows.T]
        min_height = min(len(col) for col in cols)
        cols = [col[:min_height] for col in cols]
        rows = np.array(cols).T
        return visual_input.max(-1)[..., rows]
    
    def preprocess_fly_vision(self, ommatidia_readouts):
        # return the result as np.uint8 image with values between 0 and 255
        images = self.crop_hex_to_rect(ommatidia_readouts)[:, :5]
        images = (images * 255).astype(np.uint8)
        return images
    
    # perform optic flow
    def compute_optic_flow_x(self, pre_imgs, post_imgs):
                
        return np.array(
            [
                cv2.calcOpticalFlowFarneback(pre_img, post_img, **optic_flow_kws)[..., 0]
                for pre_img, post_img in zip(pre_imgs, post_imgs)
            ]
        )

