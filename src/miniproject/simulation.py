import cv2
import numpy as np
from flygym.utils.math import Rotation3D
from flygym.simulation import Simulation
from miniproject.fly import create_fly
from miniproject.arena.terrain import RollingHills
from miniproject.arena.banana import BananaSliceMixin
from miniproject.arena.grass import GrassMixin
from miniproject.arena.dragonfly import DragonFlyMixin
from miniproject.arena.sky import SkyMixin


def sample_polar(r_range, theta_range, rng):
    r = rng.uniform(*r_range)
    theta = rng.uniform(*theta_range)
    return r * np.array((np.cos(theta), np.sin(theta)))


def _circ(
    img: np.ndarray,
    xy: tuple[float, float],
    r: float,
    value: bool,
    xmin: float,
    ymin: float,
    res: float,
    outer=False,
):
    """Draw a circle on a 2D image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw on.
    xy : tuple[float, float]
        The center of the circle.
    r : float
        The radius of the circle.
    value : bool
        The value to set the pixels to.
    xmin : float
        The minimum x value of the grid.
    ymin : float
        The minimum y value of the grid.
    res : float
        The resolution of the grid.
    outer : bool, optional
        If True, draw the outer circle. Otherwise, draw a filled circle.

    Returns
    -------
    None
    """
    center = ((np.asarray(xy) - (xmin, ymin)) / res).astype(int)
    radius = int(r / res) + 1 if outer else int(r / res)
    color = bool(value)
    thickness = 1 if outer else -1
    cv2.circle(img, center, radius, color, thickness)


def get_grass_positions(
    target_position: tuple[float, float],
    target_clearance_radius: float,
    grass_clearance_radius: float,
    fly_clearance_radius: float,
    rng: np.random.Generator,
    res: float = 0.05,
):
    """Generate random grass positions.

    Parameters
    ----------
    target_position : tuple[float, float]
        The target x and y coordinates.
    target_clearance_radius : float
        The radius of the area around the target that should be clear of grass.
    grass_clearance_radius : float
        The radius of the area around each grass that should be clear of other grass.
    fly_clearance_radius : float
        The radius of the area around the fly that should be clear of grass.
    rng : np.random.Generator
        The random number generator.
    res : float, optional
        The resolution of the grid. Default is 0.05.

    Returns
    -------
    np.ndarray
        The positions of the grass in the form of [[x1, y1], [x2, y2], ...].
    """
    target_position = np.asarray(target_position)
    distance = np.linalg.norm(target_position)
    xmin = ymin = -distance
    xmax = ymax = distance
    n_cols = int((xmax - xmin) / res)
    n_rows = int((ymax - ymin) / res)
    im1 = np.zeros((n_rows, n_cols), dtype=np.uint8)
    im2 = np.zeros((n_rows, n_cols), dtype=np.uint8)

    _circ(im1, (0, 0), distance, 1, xmin, ymin, res)
    _circ(im1, (0, 0), fly_clearance_radius, 0, xmin, ymin, res)
    _circ(im1, target_position, target_clearance_radius, 0, xmin, ymin, res)

    grass_xy = [target_position / 2]
    _circ(im1, grass_xy[0], grass_clearance_radius, 0, xmin, ymin, res)
    _circ(im2, grass_xy[0], grass_clearance_radius, 1, xmin, ymin, res, outer=True)

    while True:
        argwhere = np.argwhere(im1 & im2)
        try:
            p = argwhere[rng.choice(len(argwhere)), ::-1] * res + (xmin, ymin)
        except ValueError:
            break
        grass_xy.append(p)
        _circ(im1, p, grass_clearance_radius, 0, xmin, ymin, res)
        _circ(im2, p, grass_clearance_radius, 1, xmin, ymin, res, outer=True)

    return np.array(grass_xy)


class MiniprojectWorld(
    BananaSliceMixin, DragonFlyMixin, GrassMixin, SkyMixin, RollingHills
):
    pass


class MiniprojectSimulation(Simulation):
    def __init__(
        self,
        level,
        seed=0,
        back_cam=True,
        top_cam=True,
        camera_res=(512, 512),
    ):
        self.enable_terrain = level in (1, 2, 3, 4)
        self.enable_grass = level in (2, 3, 4)
        self.enable_wind = level in (3, 4)
        self.enable_dragonfly = level in (4,)
        rng = np.random.default_rng(seed)
        self.rng = rng

        fly = create_fly()
        cams = []

        if back_cam:
            cams.append(
                fly.add_tracking_camera(
                    name="backcam",
                    mode="fixed",
                    pos_offset=(-15, 0, 10),
                    rotation=Rotation3D("euler", (1.2, 0, -np.pi / 2)),
                    fovy=50,
                )
            )

        amplitude = 8 if self.enable_terrain else np.finfo(float).eps
        world = MiniprojectWorld(amplitude=amplitude, rng=rng)
        banana_xy = sample_polar((29, 31), (-np.pi, np.pi), rng)
        world.add_banana_slice(pos=banana_xy)

        if self.enable_grass:
            grass_positions = get_grass_positions(
                target_position=banana_xy,
                target_clearance_radius=8.0,
                grass_clearance_radius=6.0,
                fly_clearance_radius=8.0,
                rng=rng,
            )
            for xy in grass_positions:
                world.add_grass_blade((*xy, world.get_height(*xy)))

        if self.enable_dragonfly:
            world.add_dragonfly()

        world.add_fly(
            fly,
            spawn_position=(0, 0, world.get_height(0, 0) + 0.5),
            spawn_rotation=Rotation3D("quat", (1, 0, 0, 0)),
        )

        if top_cam:
            cams.append(
                world.add_camera(
                    name="birdeyecam",
                    pos=(0, 0, 60),
                    fovy=60,
                    projection="orthographic",
                )
            )

        super().__init__(world)
        self.fly = fly
        self.world = world

        if self.enable_dragonfly:
            self._init_dragonfly_controller()

        self.set_renderer(
            cams,
            camera_res=camera_res,
            stabilized_cam_indices=[],
        )
        self.reset()

        for _ in range(2000):
            self.step()

    def _init_dragonfly_controller(self):
        self._fly_name = self.fly.name
        fly_body_segments = self.fly.get_bodysegs_order()
        self._thorax_idx = next(
            (i for i, seg in enumerate(fly_body_segments) if seg.name == "c_thorax"), 0
        )
        self._thorax_body_id = self._internal_bodyids_by_fly[self._fly_name][
            self._thorax_idx
        ]

        self._dragonfly_dt = self.mj_model.opt.timestep
        self._dragonfly_cruise_height = 11.0
        self._dragonfly_hit_height_offset = 0.35
        self._dragonfly_rest_pos = np.array([0.0, 0.0, 80.0])
        self._dragonfly_rest_yaw = np.pi

        self._dragonfly_approach_vel = 50.0
        self._dragonfly_approach_start_radius = 20.0
        self._dragonfly_overshoot_dist = 5.0
        self._dragonfly_approach_angles = np.array([np.pi / 4, 3 * np.pi / 4])

        looming_lambda = 1.0
        self._dragonfly_p_no_looming = np.exp(-looming_lambda * self._dragonfly_dt)
        interception_time = (
            self._dragonfly_approach_start_radius / self._dragonfly_approach_vel
        )
        self._dragonfly_n_interception_steps = max(
            1, int(interception_time / self._dragonfly_dt)
        )
        self._dragonfly_n_overshoot_steps = max(
            1,
            int(
                (self._dragonfly_overshoot_dist / self._dragonfly_approach_vel)
                / self._dragonfly_dt
            ),
        )
        self._dragonfly_trajectory = np.zeros(
            (
                self._dragonfly_n_interception_steps
                + self._dragonfly_n_overshoot_steps,
                2,
            )
        )
        self._dragonfly_z_trajectory = np.full(
            len(self._dragonfly_trajectory),
            self._dragonfly_cruise_height,
            dtype=float,
        )

        self._dragonfly_vel_buffer_size = 500
        self._dragonfly_velocities = np.full((self._dragonfly_vel_buffer_size, 2), np.nan)
        self._dragonfly_velocities_idx = 0
        self._dragonfly_prev_fly_xy = None

        self._dragonfly_is_looming = False
        self._dragonfly_traj_advancement = 0
        self._dragonfly_attack_yaw = np.pi
        self._set_dragonfly_rest_pose()

    def _get_fly_state(self):
        thorax_pos = self.get_body_positions(self._fly_name)[self._thorax_idx].copy()
        fly_xy = thorax_pos[:2]
        fly_z = thorax_pos[2]
        xmat = self.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        heading_vec = xmat[:2, 0]
        return fly_xy, fly_z, heading_vec

    def _update_dragonfly_velocity_buffer(self, fly_xy):
        if self._dragonfly_prev_fly_xy is None:
            fly_vel = np.zeros(2, dtype=float)
        else:
            fly_vel = (fly_xy - self._dragonfly_prev_fly_xy) / self._dragonfly_dt
        self._dragonfly_prev_fly_xy = fly_xy
        self._dragonfly_velocities[
            self._dragonfly_velocities_idx % self._dragonfly_vel_buffer_size
        ] = fly_vel
        self._dragonfly_velocities_idx += 1

    def _get_mean_fly_velocity(self):
        mean_vel = np.nanmean(self._dragonfly_velocities, axis=0)
        if np.isnan(mean_vel).any():
            return np.zeros(2, dtype=float)
        return mean_vel

    def _should_trigger_dragonfly(self):
        return (
            self.rng.uniform() > self._dragonfly_p_no_looming
            and not self._dragonfly_is_looming
        )

    def _set_dragonfly_rest_pose(self):
        self.world.set_dragonfly_pose(
            self,
            self._dragonfly_rest_pos,
            (self._dragonfly_rest_yaw, 0.0, 0.0),
        )

    def _start_dragonfly_attack(self, fly_xy, fly_z, fly_heading_vec):
        fly_heading = np.arctan2(fly_heading_vec[1], fly_heading_vec[0])
        approach_side = self.rng.choice([-1, 1])
        rel_angles = self._dragonfly_approach_angles * approach_side + fly_heading
        if approach_side == -1:
            rel_angles = rel_angles[::-1]

        start_angle = self.rng.uniform(low=rel_angles[0], high=rel_angles[1])
        mean_fly_vel = self._get_mean_fly_velocity()
        interception_pos = (
            fly_xy
            + mean_fly_vel
            * self._dragonfly_n_interception_steps
            * self._dragonfly_dt
        )
        direction = np.array([np.cos(start_angle), np.sin(start_angle)])
        start_pos = (
            interception_pos + self._dragonfly_approach_start_radius * direction
        )
        end_pos = interception_pos - self._dragonfly_overshoot_dist * direction

        self._dragonfly_trajectory = np.linspace(
            start_pos, end_pos, len(self._dragonfly_trajectory)
        )
        hit_z = fly_z + self._dragonfly_hit_height_offset
        start_z = max(self._dragonfly_cruise_height, hit_z + 2.5)
        self._dragonfly_z_trajectory = np.linspace(
            start_z, hit_z, len(self._dragonfly_trajectory)
        )
        self._dragonfly_attack_yaw = start_angle + np.pi
        self._dragonfly_traj_advancement = 0
        self._dragonfly_is_looming = True

    def _step_dragonfly(self):
        fly_xy, fly_z, fly_heading_vec = self._get_fly_state()
        self._update_dragonfly_velocity_buffer(fly_xy)

        if self._should_trigger_dragonfly():
            self._start_dragonfly_attack(fly_xy, fly_z, fly_heading_vec)

        if self._dragonfly_is_looming:
            i = self._dragonfly_traj_advancement
            xy = self._dragonfly_trajectory[i]
            z = self._dragonfly_z_trajectory[i]
            self.world.set_dragonfly_pose(
                self,
                (xy[0], xy[1], z),
                (self._dragonfly_attack_yaw, 0.0, 0.0),
            )
            self._dragonfly_traj_advancement += 1
            if self._dragonfly_traj_advancement >= len(self._dragonfly_trajectory):
                self._dragonfly_is_looming = False
                self._set_dragonfly_rest_pose()
        else:
            self._set_dragonfly_rest_pose()

    def set_wind(self, magnitude, angle_deg):
        angle = np.deg2rad(angle_deg)
        wind = np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), 0])
        self.mj_model.opt.wind[:] = wind
        self.world.flow_velocity[:2] = wind[:2]

    def step(self):
        if self.enable_wind:
            if self._curr_step % 1000 == 0 and self._curr_step >= 2000:
                angle_deg = self.rng.uniform(0, 360)
                self.set_wind(magnitude=50000, angle_deg=angle_deg)
        if self.enable_dragonfly:
            self._step_dragonfly()
        super().step()
