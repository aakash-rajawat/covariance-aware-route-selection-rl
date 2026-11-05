"""MuJoCo physics backend for the converted bot_ws robot and maze.

This backend is independent of the simplified RL environment.

It does not access or modify the original ROS workspace.
"""

from pathlib import Path
import math

import mujoco
import numpy as np


class BotWSMuJoCoBackend:
    """Differential-drive robot using the converted bot_ws MJCF."""

    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.14

    LINEAR_ACCELERATION = 0.8
    LINEAR_DECELERATION = 0.4
    ANGULAR_ACCELERATION = 1.5

    CONTROLLER_PERIOD = 0.01
    DEFAULT_DECISION_PERIOD = 0.2

    DEFAULT_HEIGHT = 0.0335

    def __init__(self, scene_path=None):

        if scene_path is None:
            project_root = Path(__file__).resolve().parents[2]

            scene_path = (
                project_root
                / "generated"
                / "bot_ws_mujoco_collision_only"
                / "scene_implicitfast_torque020.xml"
            )

        self.scene_path = Path(scene_path).resolve()

        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)

        self.model = mujoco.MjModel.from_xml_path(
            str(self.scene_path)
        )

        self.data = mujoco.MjData(self.model)

        self.dt = float(self.model.opt.timestep)

        self._validate_model()

        controller_ratio = self.CONTROLLER_PERIOD / self.dt
        self.controller_substeps = round(controller_ratio)

        if not math.isclose(
            controller_ratio,
            self.controller_substeps,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Controller period must be an integer "
                "multiple of the physics timestep."
            )

        self._physics_steps = 0
        self._limited_v = 0.0
        self._limited_omega = 0.0

        self.reset()

    def _joint_id(self, name):

        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise ValueError(f"Missing joint: {name}")

        return joint_id

    def _validate_model(self):

        model = self.model

        if model.nu != 2:
            raise ValueError(
                f"Expected 2 wheel actuators, found {model.nu}"
            )

        if (
            model.opt.integrator
            != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        ):
            raise ValueError(
                "Expected the ImplicitFast integrator."
            )

        if not math.isclose(self.dt, 0.001, abs_tol=1e-10):
            raise ValueError(
                f"Unexpected physics timestep: {self.dt}"
            )

        free_joints = [
            i for i in range(model.njnt)
            if model.jnt_type[i]
            == mujoco.mjtJoint.mjJNT_FREE
        ]

        if len(free_joints) != 1:
            raise ValueError(
                "Expected exactly one floating-base joint."
            )

        self.free_joint_id = free_joints[0]

        free_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.free_joint_id,
        )

        if free_name != "robot_free_joint":
            raise ValueError(
                f"Unexpected floating joint: {free_name}"
            )

        self.free_qpos_address = int(
            model.jnt_qposadr[self.free_joint_id]
        )

        self.root_body_id = int(
            model.jnt_bodyid[self.free_joint_id]
        )

        self.right_joint_id = self._joint_id(
            "base_rw_joint"
        )

        self.left_joint_id = self._joint_id(
            "base_lw_joint"
        )

        self.right_dof = int(
            model.jnt_dofadr[self.right_joint_id]
        )

        self.left_dof = int(
            model.jnt_dofadr[self.left_joint_id]
        )

        actuator_ids = {}

        for i in range(model.nu):

            if (
                model.actuator_trntype[i]
                != mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise ValueError(
                    "Expected joint-transmission wheel actuators."
                )

            joint_id = int(model.actuator_trnid[i, 0])

            joint_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            actuator_ids[joint_name] = i

            if not model.actuator_forcelimited[i]:
                raise ValueError(
                    "Wheel actuator must be force limited."
                )

            if not np.allclose(
                model.actuator_forcerange[i],
                [-0.2, 0.2],
                atol=1e-8,
            ):
                raise ValueError(
                    "Unexpected wheel actuator force limits."
                )

            if not model.actuator_ctrllimited[i]:
                raise ValueError(
                    "Wheel actuator must have control limits."
                )

        expected = {
            "base_rw_joint",
            "base_lw_joint",
        }

        if set(actuator_ids) != expected:
            raise ValueError(
                f"Unexpected actuator mapping: {actuator_ids}"
            )

        self.right_actuator = actuator_ids[
            "base_rw_joint"
        ]

        self.left_actuator = actuator_ids[
            "base_lw_joint"
        ]

        self.robot_body_ids = set()

        for body_id in range(1, model.nbody):

            current = body_id

            while current > 0:

                if current == self.root_body_id:
                    self.robot_body_ids.add(body_id)
                    break

                current = int(
                    model.body_parentid[current]
                )

    def reset(
        self,
        pose=(1.0, 1.0, 0.0),
        height=DEFAULT_HEIGHT,
    ):
        """Reset physics, robot pose, and controller state."""

        x, y, yaw = map(float, pose)
        height = float(height)

        if not np.all(
            np.isfinite([x, y, yaw, height])
        ):
            raise ValueError("Reset pose must be finite.")

        mujoco.mj_resetData(self.model, self.data)

        qadr = self.free_qpos_address

        self.data.qpos[qadr:qadr + 3] = [
            x, y, height
        ]

        self.data.qpos[qadr + 3:qadr + 7] = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self._physics_steps = 0
        self._limited_v = 0.0
        self._limited_omega = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._check_finite()

        return self.pose()

    def pose(self):
        """Return the simulated ground-truth planar pose."""

        qadr = self.free_qpos_address

        qpos = self.data.qpos

        x = float(qpos[qadr])
        y = float(qpos[qadr + 1])

        w, qx, qy, qz = qpos[
            qadr + 3:qadr + 7
        ]

        yaw = math.atan2(
            2.0 * (w * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        return np.array(
            [x, y, yaw],
            dtype=np.float64,
        )

    def wheel_velocities(self):
        """Return right and left wheel angular velocities."""

        return np.array(
            [
                self.data.qvel[self.right_dof],
                self.data.qvel[self.left_dof],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _limit_velocity(
        current,
        target,
        acceleration,
        deceleration,
        dt,
    ):
        """Rate-limit one controller velocity component."""

        if current * target < 0.0:
            limit = deceleration

        elif abs(target) > abs(current):
            limit = acceleration

        else:
            limit = deceleration

        maximum_change = limit * dt

        change = np.clip(
            target - current,
            -maximum_change,
            maximum_change,
        )

        return float(current + change)

    def _controller_update(self, v, omega):

        self._limited_v = self._limit_velocity(
            self._limited_v,
            v,
            self.LINEAR_ACCELERATION,
            self.LINEAR_DECELERATION,
            self.CONTROLLER_PERIOD,
        )

        self._limited_omega = self._limit_velocity(
            self._limited_omega,
            omega,
            self.ANGULAR_ACCELERATION,
            self.ANGULAR_ACCELERATION,
            self.CONTROLLER_PERIOD,
        )

        half_track = self.WHEEL_SEPARATION / 2.0

        right_target = (
            self._limited_v
            + half_track * self._limited_omega
        ) / self.WHEEL_RADIUS

        left_target = (
            self._limited_v
            - half_track * self._limited_omega
        ) / self.WHEEL_RADIUS

        targets = (
            (self.right_actuator, right_target),
            (self.left_actuator, left_target),
        )

        for actuator_id, target in targets:

            low, high = (
                self.model.actuator_ctrlrange[actuator_id]
            )

            self.data.ctrl[actuator_id] = np.clip(
                target,
                low,
                high,
            )

    def _check_finite(self):

        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.ctrl,
        )

        if not all(
            np.all(np.isfinite(array))
            for array in arrays
        ):
            raise FloatingPointError(
                "Non-finite MuJoCo simulation state."
            )

    def advance(self, v, omega, duration=None):
        """Advance physics under a body-velocity command.

        The acceleration limiter and controller clock persist
        across consecutive calls.
        """

        v = float(v)
        omega = float(omega)

        if not np.all(np.isfinite([v, omega])):
            raise ValueError(
                "Velocity commands must be finite."
            )

        if duration is None:
            duration = self.DEFAULT_DECISION_PERIOD

        duration = float(duration)

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                "Duration must be positive and finite."
            )

        exact_steps = duration / self.dt
        steps = round(exact_steps)

        if not math.isclose(
            exact_steps,
            steps,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "Duration must be a multiple of "
                "the physics timestep."
            )

        for _ in range(steps):

            if (
                self._physics_steps
                % self.controller_substeps
                == 0
            ):
                self._controller_update(v, omega)

            mujoco.mj_step(self.model, self.data)

            self._physics_steps += 1

            if (
                self._physics_steps
                % self.controller_substeps
                == 0
            ):
                self._check_finite()

        self._check_finite()

        return self.pose()

    def wall_contact(self):
        """Detect robot contact with non-plane world geometry.

        Normal ground-plane contact is excluded.
        Robot self-contact is excluded.
        """

        model = self.model
        data = self.data

        for i in range(data.ncon):

            contact = data.contact[i]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])

            robot1 = body1 in self.robot_body_ids
            robot2 = body2 in self.robot_body_ids

            if robot1 and body2 == 0:
                world_geom = geom2

            elif robot2 and body1 == 0:
                world_geom = geom1

            else:
                continue

            if (
                model.geom_type[world_geom]
                == mujoco.mjtGeom.mjGEOM_PLANE
            ):
                continue

            return True

        return False

    @property
    def simulation_time(self):
        return float(self.data.time)

    @property
    def limited_command(self):
        return (
            self._limited_v,
            self._limited_omega,
        )
