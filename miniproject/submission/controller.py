import numpy as np
from miniproject.simulation import MiniprojectSimulation

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

        # --- mode coincé ---
        self.is_stuck = False
        self.stuck_count = 0
        self.stuck_duration = 20000  # Nombre de steps pendant lesquels on recule (arbitraire as fuck)

    # --- Movement ---

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        # get other observations as needed
        drives = np.array([0.5, 0.5])  # replace with your control logic

        # --- Test temporaire ---
        # proportional steering gain
        # steering_gain = 0.5
        # base_speed = 0.8

        # angle_error = 0.5  # EXEMPLE A CHANGER

        # delta = steering_gain * angle_error # compute angle error from olfaction and vision
        # drive_l = base_speed - delta
        # drive_r = base_speed + delta
        # drives = np.clip(np.array([drive_l, drive_r]), 0.3, 1.5) # ensure drives are within valid range

        # Normal mode :
        # --- Odor plume following ---
        # use olfaction to follow the odor plume towards the source
        
        olfaction_smooth = None

        if olfaction_smooth is None:
            olfaction_smooth = olfaction
        else:
            alpha = 0.001
            olfaction_smooth = (1-alpha) * olfaction_smooth + alpha * olfaction
        
        drives = odor_intensity_to_control_signal(olfaction_smooth)

        # --- Turn around itself ---
        # if plume behind the fly, make it turn around itself instead of moving forward in a big circle
        
    
        # Stuck mode :
        # --- Stuck case ---
        # if stuck, reverse the control signal and then turn around itself to try to get unstuck
        if self.is_stuck:
            # Reverse the control signal to try to get unstuck
            drives = -drives
            # Increment the stuck count
            self.stuck_count += 1
            # Check if we've been stuck for long enough to try turning around
            if self.stuck_count >= self.stuck_duration:
                # Reset the stuck count and switch back to normal mode
                self.stuck_count = 0
                self.is_stuck = False
        
        # --- Avoid obstacles ---
        # if obstacle detected, modify drives to avoid collision

        # --- Avoid dragonfly ---
        # if dragonfly detected, modify drives to avoid it


        # --- Smoothing and control ---
        # Empêche les changements trop brusques qui feraient basculer la mouche sur une bosse ou un obstacle, en limitant les variations de gain d'une étape à l'autre.

        # End conditions :
        # if close to source, stop        
       
       
        #drives = np.array([1, 1])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
    
    # --- Visual information ---

    # --- Odour --- 

