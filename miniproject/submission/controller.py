import numpy as np
from miniproject.simulation import MiniprojectSimulation


tetrapod_phase_biases = np.array(
    [
        [0, 1, 2, 2, 0, 1],
        [2, 0, 1, 1, 2, 0],
        [1, 2, 0, 0, 1, 2],
        [1, 2, 0, 0, 1, 2],
        [0, 1, 2, 2, 0, 1],
        [2, 0, 1, 1, 2, 0],
    ]
) * (2 * np.pi / 3)

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        #self.turning_controller = TurningController(sim.timestep)
        self.turning_controller = TurningController(sim.timestep, phase_biases=tetrapod_phase_biases)

    # --- Movement ---

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        # get other observations as needed
        drives = np.array([0.5, 0.5])  # replace with your control logic

        # --- Test temporaire ---
        # proportional steering gain
        steering_gain = 0.5
        base_speed = 0.8

        angle_error = 0.5  # EXEMPLE A CHANGER

        delta = steering_gain * angle_error # compute angle error from olfaction and vision
        drive_l = base_speed - delta
        drive_r = base_speed + delta
        drives = np.clip(np.array([drive_l, drive_r]), 0.3, 1.5) # ensure drives are within valid range

        # --- Stuck case ---
        # if stuck, apply a different control strategy (e.g., reverse or turn in place
        
        # --- Odor plume following ---
        # use olfaction to follow the odor plume towards the source

        # --- Avoid obstacles ---
        # if obstacle detected, modify drives to avoid collision

        # --- Avoid dragonfly ---
        # if dragonfly detected, modify drives to avoid it

        # --- Smoothing and control ---
        # Empêche les changements trop brusques qui feraient basculer la mouche sur une bosse ou un obstacle, en limitant les variations de gain d'une étape à l'autre.


        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
    
    # --- Visual information ---

    # --- Odour --- 

