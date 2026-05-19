import numpy as np
from miniproject.simulation import MiniprojectSimulation


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).ravel()
    if quat.shape[-1] != 4:
        raise ValueError("Antenna quaternion qpos must have 4 elements")
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def antenna_qpos_to_heading_deg(qpos: np.ndarray) -> float:
    R = quaternion_to_rotation_matrix(qpos)
    local_axis = np.array([1.0, 0.0, 0.0])
    world_axis = R @ local_axis
    return np.rad2deg(np.arctan2(world_axis[1], world_axis[0]))


def estimate_wind_angle_from_antennae(antenna_data: dict[str, dict[str, np.ndarray]]) -> float:
    headings = []
    for side in ["l", "r"]:
        qpos = antenna_data[side]["qpos"]
        headings.append(antenna_qpos_to_heading_deg(qpos))

    angles = np.deg2rad(headings)
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    return np.rad2deg(mean_angle)


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

        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        antenna_pos = sim.get_antenna_data(sim.fly.name)
        wind_angle_est = estimate_wind_angle_from_antennae(antenna_pos)
        self.wind_angle_estimate = wind_angle_est
        # get other observations as needed
        olfaction_smooth = None


        if olfaction_smooth is None:
            olfaction_smooth = olfaction
        else:
            alpha = 0.001
            olfaction_smooth = (1-alpha) * olfaction_smooth + alpha * olfaction
        
        drives = 0*odor_intensity_to_control_signal(olfaction_smooth)
       
       
       
        #drives = np.array([1, 1])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
