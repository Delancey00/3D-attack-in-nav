import random   
import numpy as np
from typing import Tuple, Union, List, Any
import mitsuba as mi


#从矩阵中提取相机位置的半径
def extract_camera_radius_from_matrix(camera_matrix: Any, center_pos: Tuple[float, float, float]) -> float:
    """
    从相机变换矩阵中提取到中心点的半径距离
    
    Args:
        camera_matrix: 相机变换矩阵，支持多种类型（DrJit Transform4f, Matrix4f, numpy array等）
        center_pos: 中心点位置 (x, y, z)
    
    Returns:
        float: 相机位置到中心点的水平距离（半径）
    
    Raises:
        ValueError: 当无法从矩阵中提取位置信息时
    """
    try:
        # 提取相机位置 (变换矩阵的平移部分)
        if hasattr(camera_matrix, 'matrix'):
            matrix = camera_matrix.matrix
            # 处理DrJit类型，需要先转换为numpy再转float
            try:
                cam_x = float(matrix[0, 3].numpy()) if hasattr(matrix[0, 3], 'numpy') else float(matrix[0, 3])
                cam_y = float(matrix[1, 3].numpy()) if hasattr(matrix[1, 3], 'numpy') else float(matrix[1, 3])
            except:
                # 如果上述方法失败，尝试直接访问数据
                cam_x = matrix[0, 3].data if hasattr(matrix[0, 3], 'data') else matrix[0, 3]
                cam_y = matrix[1, 3].data if hasattr(matrix[1, 3], 'data') else matrix[1, 3]
                cam_x = float(cam_x)
                cam_y = float(cam_y)
        else:
            # 处理普通矩阵类型
            try:
                cam_x = float(camera_matrix[0, 3].numpy()) if hasattr(camera_matrix[0, 3], 'numpy') else float(camera_matrix[0, 3])
                cam_y = float(camera_matrix[1, 3].numpy()) if hasattr(camera_matrix[1, 3], 'numpy') else float(camera_matrix[1, 3])
            except:
                cam_x = camera_matrix[0, 3].data if hasattr(camera_matrix[0, 3], 'data') else camera_matrix[0, 3]
                cam_y = camera_matrix[1, 3].data if hasattr(camera_matrix[1, 3], 'data') else camera_matrix[1, 3]
                cam_x = float(cam_x)
                cam_y = float(cam_y)
        
        # 计算相机位置到中心点的水平距离
        center_x, center_y = center_pos[0], center_pos[1]
        radius = np.sqrt((cam_x - center_x)**2 + (cam_y - center_y)**2)
        return radius
        
    except Exception as e:
        raise ValueError(f"无法从相机矩阵中提取位置信息: {str(e)}")


def get_smart_perturbation_params(cam_idx: int, 
                                 camera_positions: List[Any], 
                                 center_pos: Tuple[float, float, float]) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float]]:
    """
    智能半径扰动：基于具体半径值优化的方案
    
    Args:
        cam_idx: 当前相机索引
        camera_positions: 相机位置列表
        center_pos: 中心点位置 (x, y, z)
    
    Returns:
        Tuple包含:
        - pitch_perturbation: 俯仰角扰动（度）
        - yaw_perturbation: 航向角扰动（度） 
        - closest_radius: 最接近的预设半径
        - pitch_range: 俯仰角扰动范围
        - yaw_range: 航向角扰动范围
    """
    # 半径到角度扰动的映射表
    radius_perturbation_map = {
        0.75: {'yaw': (15, 20), 'pitch': (0, 0),'position_xz': (0.05, 0.10) },   # 最内圈，最大扰动
        1.00: {'yaw': (15, 22), 'pitch': (0, 0),'position_xz': (0.08, 0.15)},   # 内圈
        1.25: {'yaw': (15, 24), 'pitch': (0, 0),'position_xz': (0.10, 0.18)},   # 中内圈
        1.50: {'yaw': (15, 20), 'pitch': (0, 0),'position_xz': (0.12, 0.20)},   # 中圈
        1.75: {'yaw': (10, 15.25), 'pitch': (0, 0),'position_xz': (0.15, 0.25)}, # 中外圈
        2.00: {'yaw': (15, 25), 'pitch': (0, 0),'position_xz': (0.18, 0.30)},   # 外圈
        2.25: {'yaw': (15, 25), 'pitch': (0, 0),'position_xz': (0.20, 0.35)},   # 远外圈
        2.50: {'yaw': (15, 25), 'pitch': (0, 0),'position_xz': (0.25, 0.40)}    # 最外圈，最小扰动
    }
    # radius_perturbation_map = {
    #     0.75: {'yaw': (5, 8), 'pitch': (0, 0),'position_xz': (0.10, 0.15) },   # 最内圈，最大扰动
    #     1.00: {'yaw': (8, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 内圈
    #     1.25: {'yaw': (12, 15), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中内圈
    #     1.50: {'yaw': (15, 20), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中圈
    #     1.75: {'yaw': (10, 15.25), 'pitch': (0, 0),'position_xz': (0.20, 0.25)}, # 中外圈
    #     2.00: {'yaw': (15, 25), 'pitch': (0, 0),'position_xz': (0.20, 0.25)},   # 外圈
    #     2.25: {'yaw': (15, 23), 'pitch': (0, 0),'position_xz': (0.20, 0.25)},   # 远外圈
    #     2.50: {'yaw': (15, 20), 'pitch': (0, 0),'position_xz': (0.20, 0.25)}    # 最外圈，最小扰动
    # }
    # radius_perturbation_map = {
    #    # 0.25: {'yaw': (5, 8), 'pitch': (0, 0),'position_xz': (0.20, 0.25) },   # 最内圈，最大扰动
    #     0.3: {'yaw': (8, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 内圈
    #     0.5: {'yaw': (10, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中内圈
    #     0.7: {'yaw': (10, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中圈
    #     0.9: {'yaw': (5, 8), 'pitch': (0, 0),'position_xz': (0.20, 0.25)}, # 中外圈  廁所
    # }
    # radius_perturbation_map = {
    #    # 0.25: {'yaw': (5, 8), 'pitch': (0, 0),'position_xz': (0.20, 0.25) },   # 最内圈，最大扰动
    #     1: {'yaw': (8, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 内圈
    #     1.25: {'yaw': (10, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中内圈
    #     1.5: {'yaw': (10, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中圈
    #     1.75: {'yaw': (5, 8), 'pitch': (0, 0),'position_xz': (0.20, 0.25)}, # 中外圈  bed
    # }
    # radius_perturbation_map = {
    #     0.1: {'yaw': (8, 10), 'pitch': (0, 0),'position_xz': (0.20, 0.25) },   # 最内圈，最大扰动
    #     0.25: {'yaw': (8, 10), 'pitch': (0, 0),'position_xz': (0.20, 0.25) },   # 最内圈，最大扰动
    #     0.5: {'yaw': (8, 10), 'pitch': (0, 0),'position_xz': (0.20, 0.25) },   # 最内圈，最大扰动
    #     0.75: {'yaw': (8, 10), 'pitch': (0, 0),'position_xz': (0.10, 0.15) },   # 最内圈，最大扰动
    #     1.00: {'yaw': (8, 12), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 内圈
    #     1.25: {'yaw': (12, 15), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中内圈
    #     1.50: {'yaw': (15, 20), 'pitch': (0, 0),'position_xz': (0.15, 0.20)},   # 中圈
    #     1.75: {'yaw': (10, 15.25), 'pitch': (0, 0),'position_xz': (0.20, 0.25)}, # 中外圈
    #     2.00: {'yaw': (15, 25), 'pitch': (0, 0),'position_xz': (0.20, 0.25)},   # 外圈
    #     2.25: {'yaw': (15, 23), 'pitch': (0, 0),'position_xz': (0.20, 0.25)},   # 远外圈
    #     2.50: {'yaw': (15, 20), 'pitch': (0, 0),'position_xz': (0.20, 0.25)}    # 最外圈，最小扰动plant
    # }
    
    # 获取当前相机的半径
    current_radius = extract_camera_radius_from_matrix(camera_positions[cam_idx], center_pos)
    
    # 找到最接近的预设半径
    closest_radius = min(radius_perturbation_map.keys(), 
                        key=lambda r: abs(r - current_radius))


    #获取对应的变换参数范围
    # 找到最接近的预设半径
    closest_radius = min(radius_perturbation_map.keys(), 
                        key=lambda r: abs(r - current_radius))
    
    # 获取对应的扰动参数
    params = radius_perturbation_map[closest_radius].copy()
    params['radius'] = closest_radius
    params['actual_radius'] = current_radius

    return params


def generate_perturbations(perturbation_params:dict) -> dict:
    """
    基于参数生成具体的扰动值
    
    Args:
        perturbation_params: 扰动参数字典
    
    Returns:
        dict: 包含具体扰动值的字典
    """
    perturbations = {}

    # 角度扰动
    pitch_range = perturbation_params['pitch']
    yaw_range = perturbation_params['yaw']
    
    perturbations['pitch'] = random.uniform(-pitch_range[1], pitch_range[1])
    
    # yaw扰动：双向扰动
    if random.choice([True, False]):
        perturbations['yaw'] = random.uniform(yaw_range[0], yaw_range[1])
    else:
        perturbations['yaw'] = random.uniform(-yaw_range[1], -yaw_range[0])

    # 位置扰动（只扰动XZ平面，保持Y轴高度不变）
    xz_range = perturbation_params['position_xz']
    
    # XZ平面的随机扰动（2D圆形区域内）
    angle = random.uniform(0, 2 * np.pi)
    radius = random.uniform(xz_range[0], xz_range[1])
    perturbations['pos_x'] = radius * np.cos(angle)
    perturbations['pos_z'] = radius * np.sin(angle)
    
    # Y轴扰动为0（保持原始高度不变）
    perturbations['pos_y'] = 0.0
    
    return perturbations


    
    

def create_combined_transform_matrix(pitch_deg: float, yaw_deg: float, 
                                   pos_offset: Tuple[float, float, float]) -> np.ndarray:
    """
    在CPU上计算旋转矩阵，减少GPU内存占用
    
    Args:
        pitch_deg: 俯仰角（度）
        yaw_deg: 航向角（度）
    
    Returns:
        numpy.ndarray: 4x4旋转矩阵
    """
    # 转换为弧度
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    
    # 计算三角函数值
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    
    # X轴旋转矩阵（俯仰）
    rot_x = np.array([
        [1.0,    0.0,     0.0, 0.0],
        [0.0, cos_p, -sin_p, 0.0],
        [0.0, sin_p,  cos_p, 0.0],
        [0.0,    0.0,     0.0, 1.0]
    ], dtype=np.float32)
    
    # Y轴旋转矩阵（航向）
    rot_y = np.array([
        [ cos_y, 0.0, sin_y, 0.0],
        [    0.0, 1.0,    0.0, 0.0],
        [-sin_y, 0.0, cos_y, 0.0],
        [    0.0, 0.0,    0.0, 1.0]
    ], dtype=np.float32)
    
    # 组合旋转（X * Y 顺序，与原代码保持一致）
    rotation_matrix = rot_x @ rot_y
    #添加位置偏移
    rotation_matrix[0, 3] = pos_offset[0]  # dx
    rotation_matrix[1, 3] = pos_offset[1]  # dy (应该为0)
    rotation_matrix[2, 3] = pos_offset[2]  # dz
    
    return rotation_matrix
    #return combined_rotation



def apply_camera_perturbation_with_position(original_transform: Any, 
                                            pitch_deg: float, 
                                            yaw_deg: float,
                                            pos_offset: Tuple[float, float, float]) -> Any:
    """
    对相机变换矩阵应用旋转扰动
    
    Args:
        original_transform: 原始相机变换矩阵
        pitch_deg: 俯仰角扰动（度）
        yaw_deg: 航向角扰动（度）
    
    Returns:
        扰动后的变换矩阵
        
    Note:
        这个函数封装了旋转矩阵计算和应用的完整过程，
        需要在有mitsuba环境的地方调用
    """
    # CPU计算旋转矩阵（减少GPU内存占用）
    perturbation_matrix = create_combined_transform_matrix(pitch_deg, yaw_deg, pos_offset)
    
    # 转换为Mitsuba格式并应用扰动
    # 注意：这里需要导入mitsuba
    try:
        perturbation_transform = mi.cuda_ad_rgb.Transform4f(
            mi.cuda_ad_rgb.Matrix4f(perturbation_matrix)
        )
        perturbed_transform = original_transform @ perturbation_transform
        return perturbed_transform
    except ImportError:
        raise ImportError("需要导入mitsuba库才能使用此函数")
    except Exception as e:
        raise RuntimeError(f"应用相机扰动失败: {str(e)}")

def batch_apply_smart_perturbations(camera_positions: List[Any], 
                                    center_pos: Tuple[float, float, float],
                                    enable_position_perturbation: bool = True,
                                    position_scale: float = 1.0,
                                    seed: int = None) -> List[Tuple[Any, dict]]:
    """
    批量对相机位置应用智能扰动
    
    Args:
        camera_positions: 相机位置列表
        center_pos: 中心点位置
        enable_position_perturbation: 是否启用位置扰动（仅XZ平面）
        position_scale: 位置扰动缩放因子
        seed: 随机种子，用于可重复性
    
    Returns:
        List[Tuple]: 每个元素包含(扰动后的变换矩阵, 扰动信息字典)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    results = []
    
    for i, camera_pos in enumerate(camera_positions):
        
        # # 获取智能扰动参数
        # pitch_pert, yaw_pert, radius, pitch_range, yaw_range = get_smart_radius_perturbation(
        #     i, camera_positions, center_pos
        # )
        # # 应用扰动
        # perturbed_matrix = apply_camera_perturbation(camera_pos, pitch_pert, yaw_pert)
        # 获取智能扰动参数
        params = get_smart_perturbation_params(i, camera_positions, center_pos)
        # 生成具体扰动值
        perturbations = generate_perturbations(params)

        # 缩放位置扰动
        if enable_position_perturbation:
            pos_offset = (
                perturbations['pos_x'] * position_scale,
                perturbations['pos_y'],  # 始终为0
                perturbations['pos_z'] * position_scale
            )
        else:
            pos_offset = (0.0, 0.0, 0.0)

        # 应用扰动
        perturbed_matrix = apply_camera_perturbation_with_position(
            camera_pos, 
            perturbations['pitch'], 
            perturbations['yaw'],
            pos_offset
        )
       
        
        # 扰动信息
        perturbation_info = {
            'camera_idx': i,
            'radius': params['radius'],
            'actual_radius': params['actual_radius'],
            'pitch_perturbation': perturbations['pitch'],
            'yaw_perturbation': perturbations['yaw'],
            'position_offset': pos_offset,
            'params': params,
            'enable_position_perturbation': enable_position_perturbation
        }
        
        results.append((perturbed_matrix, perturbation_info))
    
    return results



def example_usage():
    """
    使用示例
    """
    print("=== Enhanced Camera Perturbation Example ===")
    
    center_pos = (7.84387, -3.82026, 0.88)
    
    # 创建模拟的相机变换矩阵
    mock_camera_positions = []
    for i in range(3):
        transform = np.eye(4, dtype=np.float32)
        radius = 1.0 + i * 0.5
        angle = i * np.pi / 3
        transform[0, 3] = center_pos[0] + radius * np.cos(angle)  # x
        transform[1, 3] = center_pos[1] + radius * np.sin(angle) # y 
        transform[2, 3] = center_pos[2]  # z(height)
        mock_camera_positions.append(transform)
    
    # 测试智能扰动参数
    for i in range(len(mock_camera_positions)):
        params = get_smart_perturbation_params(i, mock_camera_positions, center_pos)
        perturbations = generate_perturbations(params)
        
        print(f"Camera {i}:")
        print(f"  Radius: {params['radius']} (actual: {params['actual_radius']:.3f})")
        print(f"  Angle perturbations: pitch={perturbations['pitch']:.2f}°, yaw={perturbations['yaw']:.2f}°")
        print(f"  Position offsets (XZ only): x={perturbations['pos_x']:.3f}, z={perturbations['pos_z']:.3f}, y=0.0 (fixed)")
        print()


if __name__ == "__main__":
    example_usage()
