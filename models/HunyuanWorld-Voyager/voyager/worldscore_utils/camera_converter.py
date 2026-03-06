# WorldScore/models/HunyuanWorld-Voyager/voyager/worldscore_utils/camera_converter.py

import numpy as np
import torch
from typing import Union, List

class CameraCoordinateConverter:
    """
    坐标系转换工具类

    WorldScore使用Blender坐标系（右手系）：
        +X = Right
        +Y = Up
        +Z = Backward (指向相机后方)
        保存格式：c2w (camera-to-world) 4x4矩阵

    Voyager使用OpenCV坐标系（右手系）：
        +X = Right
        +Y = Down
        +Z = Forward (光轴方向)
        使用格式：w2c (world-to-camera) 4x4矩阵
    """

    @staticmethod
    def blender_to_opencv_matrix():
        """
        Blender → OpenCV 坐标系转换矩阵

        转换关系：
        OpenCV_X =  Blender_X
        OpenCV_Y = -Blender_Y (翻转)
        OpenCV_Z = -Blender_Z (翻转)
        """
        return np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ], dtype=np.float32)

    @staticmethod
    def worldscore_c2w_to_voyager_w2c(
            c2w_blender: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        WorldScore的c2w (Blender) → Voyager的w2c (OpenCV)

        步骤：
        1. Blender c2w → OpenCV c2w (坐标系转换)
        2. OpenCV c2w → OpenCV w2c (求逆)

        Args:
            c2w_blender: [4, 4] WorldScore的相机矩阵 (Blender c2w)

        Returns:
            w2c_opencv: [4, 4] Voyager的相机矩阵 (OpenCV w2c)
        """
        if isinstance(c2w_blender, torch.Tensor):
            c2w_blender = c2w_blender.cpu().numpy()

        c2w_blender = c2w_blender.astype(np.float32)

        # 转换矩阵
        T_b2o = CameraCoordinateConverter.blender_to_opencv_matrix()

        # Blender c2w → OpenCV c2w
        # T_o2o = T_b2o @ T_b2w @ T_b2o^T
        c2w_opencv = T_b2o @ c2w_blender @ T_b2o.T

        # OpenCV c2w → OpenCV w2c
        w2c_opencv = np.linalg.inv(c2w_opencv)

        return w2c_opencv

    @staticmethod
    def voyager_w2c_to_worldscore_c2w(
            w2c_opencv: np.ndarray
    ) -> np.ndarray:
        """
        Voyager的w2c (OpenCV) → WorldScore的c2w (Blender)

        逆向转换，用于保存结果
        """
        w2c_opencv = w2c_opencv.astype(np.float32)

        # OpenCV w2c → OpenCV c2w
        c2w_opencv = np.linalg.inv(w2c_opencv)

        # 转换矩阵
        T_b2o = CameraCoordinateConverter.blender_to_opencv_matrix()

        # OpenCV c2w → Blender c2w
        # T_b2w = T_b2o^T @ T_o2w @ T_b2o
        c2w_blender = T_b2o.T @ c2w_opencv @ T_b2o

        return c2w_blender

    @staticmethod
    def convert_camera_list(
            cameras: List[Union[np.ndarray, torch.Tensor]],
            from_format: str = "worldscore",
            to_format: str = "voyager"
    ) -> List[np.ndarray]:
        """
        批量转换相机列表

        Args:
            cameras: 相机矩阵列表
            from_format: 'worldscore' (Blender c2w) 或 'voyager' (OpenCV w2c)
            to_format: 'voyager' (OpenCV w2c) 或 'worldscore' (Blender c2w)
        """
        converted = []

        for cam in cameras:
            if from_format == "worldscore" and to_format == "voyager":
                converted.append(
                    CameraCoordinateConverter.worldscore_c2w_to_voyager_w2c(cam)
                )
            elif from_format == "voyager" and to_format == "worldscore":
                converted.append(
                    CameraCoordinateConverter.voyager_w2c_to_worldscore_c2w(cam)
                )
            else:
                raise ValueError(f"Unsupported conversion: {from_format} → {to_format}")

        return converted


# 便捷函数
def ws2voy(c2w_blender):
    """WorldScore c2w → Voyager w2c"""
    return CameraCoordinateConverter.worldscore_c2w_to_voyager_w2c(c2w_blender)

def voy2ws(w2c_opencv):
    """Voyager w2c → WorldScore c2w"""
    return CameraCoordinateConverter.voyager_w2c_to_worldscore_c2w(w2c_opencv)
