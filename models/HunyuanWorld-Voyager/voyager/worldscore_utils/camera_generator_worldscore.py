# WorldScore/models/HunyuanWorld-Voyager/voyager/worldscore_utils/camera_generator_worldscore.py

import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class WorldScoreCameraGenerator:
    """
    完全复现WorldScore的相机生成逻辑
    支持11种相机命令 + 插值
    """

    def __init__(
            self,
            camera_speed: float = 1.0,
            rotation_range_theta: float = 30.0,  # 度数
            total_frames: int = 49
    ):
        self.camera_speed = camera_speed
        self.rotation_range_theta = rotation_range_theta
        self.total_frames = total_frames

        # 相机初始状态（单位矩阵）
        self.init_camera = np.eye(4, dtype=np.float32)

    def generate_camera_trajectory(
            self,
            command: str,
            start_camera: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        生成完整的相机轨迹

        Args:
            command: WorldScore相机命令
            start_camera: [4, 4] 起始相机矩阵（w2c格式）

        Returns:
            keyframe_cameras: 关键帧相机（起始+结束）
            interpolated_cameras: 插值后的完整轨迹（49帧）
        """
        if start_camera is None:
            start_camera = self.init_camera.copy()

        # 生成结束帧
        end_camera = self._apply_camera_command(command, start_camera)

        keyframe_cameras = [start_camera, end_camera]

        # 插值
        interpolated_cameras = self._interpolate_cameras(
            start_camera, end_camera, self.total_frames
        )

        return keyframe_cameras, interpolated_cameras

    def _apply_camera_command(
            self,
            command: str,
            camera: np.ndarray
    ) -> np.ndarray:
        """
        应用单个相机命令，返回新的相机矩阵
        """
        camera = camera.copy()

        # 提取R和T (w2c格式)
        R_w2c = camera[:3, :3]
        T_w2c = camera[:3, 3]

        if command == "push_in":
            # 向前推进（在相机坐标系中沿+Z移动）
            T_w2c[2] -= self.camera_speed * 1.0

        elif command == "pull_out":
            # 向后拉出
            T_w2c[2] += self.camera_speed * 1.0

        elif command == "move_left":
            # 左移（在相机坐标系中沿-X移动）
            T_w2c[0] -= self.camera_speed * 0.5

        elif command == "move_right":
            # 右移
            T_w2c[0] += self.camera_speed * 0.5

        elif command == "orbit_left":
            # 绕场景中心向左旋转
            camera = self._orbit(camera, -self.rotation_range_theta)
            return camera

        elif command == "orbit_right":
            # 绕场景中心向右旋转
            camera = self._orbit(camera, self.rotation_range_theta)
            return camera

        elif command == "pan_left":
            # 相机原地左转（旋转朝向）
            theta_rad = np.deg2rad(self.rotation_range_theta)
            R_pan = self._rotation_matrix_y(theta_rad)
            R_w2c = R_w2c @ R_pan

        elif command == "pan_right":
            # 相机原地右转
            theta_rad = np.deg2rad(-self.rotation_range_theta)
            R_pan = self._rotation_matrix_y(theta_rad)
            R_w2c = R_w2c @ R_pan

        elif command == "pull_left":
            # 组合命令：左移 + 拉出 + 左转
            T_w2c[0] -= self.camera_speed * 0.3  # 左移
            T_w2c[2] += self.camera_speed * 0.3  # 拉出
            theta_rad = np.deg2rad(self.rotation_range_theta)
            R_pan = self._rotation_matrix_y(theta_rad)
            R_w2c = R_w2c @ R_pan  # 左转

        elif command == "pull_right":
            # 组合命令：右移 + 拉出 + 右转
            T_w2c[0] += self.camera_speed * 0.3  # 右移
            T_w2c[2] += self.camera_speed * 0.3  # 拉出
            theta_rad = np.deg2rad(-self.rotation_range_theta)
            R_pan = self._rotation_matrix_y(theta_rad)
            R_w2c = R_w2c @ R_pan  # 右转

        elif command == "fixed":
            # 固定不动，直接返回
            return camera

        else:
            raise ValueError(f"Unsupported camera command: {command}")

        # 重新组装矩阵
        camera[:3, :3] = R_w2c
        camera[:3, 3] = T_w2c

        return camera

    def _orbit(self, camera: np.ndarray, theta_deg: float) -> np.ndarray:
        """
        绕场景中心旋转（Orbit）

        实现逻辑（参考WorldScore）：
        1. 将相机移动到旋转中心
        2. 绕Y轴旋转
        3. 移回原位置
        """
        camera = camera.copy()
        R_w2c = camera[:3, :3]
        T_w2c = camera[:3, 3]

        # 计算旋转半径（当前深度）
        radius = self.camera_speed
        z = T_w2c[2]

        # 平移到旋转中心
        T_to_center = T_w2c.copy()
        T_to_center[2] = radius - z

        # 旋转矩阵
        theta_rad = np.deg2rad(theta_deg)
        R_orbit = self._rotation_matrix_y(theta_rad)

        # 应用变换
        # 1. 移到中心
        cam_centered = np.eye(4)
        cam_centered[:3, :3] = R_w2c
        cam_centered[:3, 3] = T_to_center

        # 2. 旋转
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R_orbit
        cam_rotated = R_4x4 @ cam_centered

        # 3. 移回
        T_back = T_w2c.copy()
        T_back[2] = z - radius
        cam_back = np.eye(4)
        cam_back[:3, :3] = np.eye(3)
        cam_back[:3, 3] = T_back

        result = cam_rotated @ cam_back

        return result

    def _rotation_matrix_y(self, theta: float) -> np.ndarray:
        """绕Y轴旋转矩阵（右手系）"""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            [cos_t,  0, -sin_t],
            [0,      1,  0],
            [sin_t,  0,  cos_t]
        ], dtype=np.float32)

    def _interpolate_cameras(
            self,
            cam1: np.ndarray,
            cam2: np.ndarray,
            num_frames: int
    ) -> List[np.ndarray]:
        """
        在两个相机之间进行插值（Slerp for rotation + Linear for translation）

        返回包含起始和结束帧的完整序列
        """
        R1 = cam1[:3, :3]
        R2 = cam2[:3, :3]
        T1 = cam1[:3, 3]
        T2 = cam2[:3, 3]

        # 转为四元数
        quat1 = R.from_matrix(R1).as_quat()
        quat2 = R.from_matrix(R2).as_quat()

        # 插值
        cameras = []
        for i in range(num_frames):
            t = i / (num_frames - 1)

            # Slerp for rotation
            slerp = Slerp([0, 1], R.from_quat([quat1, quat2]))
            R_interp = slerp([t]).as_matrix()[0]

            # Linear for translation
            T_interp = T1 + t * (T2 - T1)

            # 组装
            cam_interp = np.eye(4, dtype=np.float32)
            cam_interp[:3, :3] = R_interp
            cam_interp[:3, 3] = T_interp

            cameras.append(cam_interp)

        return cameras


# 测试代码
if __name__ == "__main__":
    gen = WorldScoreCameraGenerator()

    # 测试所有命令
    commands = [
        "push_in", "pull_out", "move_left", "move_right",
        "orbit_left", "orbit_right", "pan_left", "pan_right",
        "pull_left", "pull_right", "fixed"
    ]

    for cmd in commands:
        keyframes, trajectory = gen.generate_camera_trajectory(cmd)
        print(f"{cmd}: Generated {len(trajectory)} frames")
        print(f"  Start T: {keyframes[0][:3, 3]}")
        print(f"  End T:   {keyframes[1][:3, 3]}\n")
