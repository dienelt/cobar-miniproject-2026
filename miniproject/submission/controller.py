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


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.olfaction_smooth = None

    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)
        if self.olfaction_smooth is None:
            self.olfaction_smooth = olfaction
        else:
            self.olfaction_smooth = 0.9995 * self.olfaction_smooth + 0.0005 * olfaction
        odor_drives = odor_intensity_to_control_signal(self.olfaction_smooth)

        raw_vision = sim.get_raw_vision(sim.fly.name)
        avoid_drives = dragonfly_avoidance_drives(raw_vision)

        if avoid_drives is not None:
            drives = avoid_drives       # override during attack
        else:
            drives = odor_drives        # normal navigation

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion