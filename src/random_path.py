import random
import numpy as np
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置日志记录器
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # 您可以取消注释此行以获得更详细的调试输出

class TrajectoryGenerator:
    """
    根据真实的相机布局动态生成平滑的攻击轨迹。

    工作流程:
    1. 初始化: 接收来自 dt2.py 的 `camera_info`，包含所有相机的位置、半径和角度。
    2. 动态分层: 根据相机的半径（使用一个容差范围）将它们自动分组到不同的“层”。
    3. 构建图: 根据新的连接规则创建一个表示相机连通性的图。
       - 同层内: 任意两个相机都可互相访问。
       - 相邻层间: 一个相机可以访问其径向对齐的相机及其左右邻居。
    4. 轨迹搜索: 从最外层开始，使用深度优先搜索（DFS）寻找一条逐步向内、
       访问所有层、且长度符合要求的路径。
    """

    def __init__(self, camera_info: dict, radius_tolerance=0.1, angle_tolerance=4.5):
        """
        使用真实的相机信息初始化轨迹生成器。

        Args:
            camera_info (dict): 从 dt2.py 的 `extract_camera_info_from_scene` 函数获取的字典。
            radius_tolerance (float): 用于将相机划分到同一层的半径容差。
            angle_tolerance (float): 用于判断相机是否“径向对齐”的角度容差（单位：度）。
        """
        self.camera_info = camera_info
        self.radius_tolerance = radius_tolerance
        self.angle_tolerance = angle_tolerance
        
        self.layers = {}
        self.graph = defaultdict(list)
        self.is_initialized = False

        if self.camera_info:
            try:
                self._process_camera_data()
                self._build_graph()
                self.is_initialized = True
                logger.info(f"轨迹生成器初始化成功，识别出 {len(self.layers)} 个相机层。")
            except Exception as e:
                logger.error(f"轨迹生成器初始化失败: {e}", exc_info=True)
        else:
            logger.warning("未提供相机信息，轨迹生成器无法初始化。")

    def _process_camera_data(self):
        """
        将原始相机数据处理并分层。
        """
        if not self.camera_info.get('camera_radii'):
            raise ValueError("相机信息中缺少 'camera_radii'。")

        # 1. 提取并打包相机属性
        cameras = []
        for i in range(self.camera_info['camera_count']):
            cameras.append({
                'id': i,
                'radius': self.camera_info['camera_radii'][i],
                'angle': self.camera_info['camera_angles'][i]
            })

        # 2. 按半径排序，准备分层
        cameras.sort(key=lambda c: c['radius'])

        # 3. 根据半径容差进行分层
        if not cameras:
            return

        # 使用第一个相机初始化第一层
        self.layers = {0: [cameras[0]]}
        current_layer_idx = 0
        
        for i in range(1, len(cameras)):
            # 如果当前相机的半径与该层第一个相机的半径差在容差内，则属于同一层
            if abs(cameras[i]['radius'] - self.layers[current_layer_idx][0]['radius']) <= self.radius_tolerance:
                self.layers[current_layer_idx].append(cameras[i])
            else:
                # 否则，创建新的一层
                current_layer_idx += 1
                self.layers[current_layer_idx] = [cameras[i]]
        
        # 4. 对每一层内的相机按角度排序，方便查找邻居
        for layer_idx in self.layers:
            self.layers[layer_idx].sort(key=lambda c: c['angle'])
            logger.debug(f"层 {layer_idx} (半径约 {self.layers[layer_idx][0]['radius']:.2f}): "
                         f"包含 {len(self.layers[layer_idx])} 个相机。")

    def _build_graph(self):
        """
        根据分层结果和连接规则构建图。
        """
        for layer_idx, cameras_in_layer in self.layers.items():
            # 规则1: 同一层内相机完全连通
            camera_ids_in_layer = [c['id'] for c in cameras_in_layer]
            for cam_id in camera_ids_in_layer:
                # 添加除自身外的所有同层相机为邻居
                neighbors = [other_id for other_id in camera_ids_in_layer if other_id != cam_id]
                self.graph[cam_id].extend(neighbors)

            # 规则2: 与相邻层连接
            for i, cam in enumerate(cameras_in_layer):
                # 查找相邻层（更内或更外）
                for adjacent_layer_idx in [layer_idx - 1, layer_idx + 1]:
                    if adjacent_layer_idx in self.layers:
                        cameras_in_adjacent_layer = self.layers[adjacent_layer_idx]
                        
                        # 寻找径向对齐的相机
                        best_match = None
                        min_angle_diff = float('inf')
                        for adj_cam in cameras_in_adjacent_layer:
                            angle_diff = abs(adj_cam['angle'] - cam['angle'])
                            # 处理角度环绕问题 (e.g., 359度 vs 1度)
                            angle_diff = min(angle_diff, 360 - angle_diff)
                            if angle_diff < min_angle_diff:
                                min_angle_diff = angle_diff
                                best_match = adj_cam
                        
                        # 如果找到了在容差内的径向对齐相机
                        if best_match and min_angle_diff <= self.angle_tolerance:
                            # 添加径向对齐的相机为邻居
                            self.graph[cam['id']].append(best_match['id'])
                            
                            # 找到它在自己层内的索引
                            best_match_idx = cameras_in_adjacent_layer.index(best_match)
                            
                            # 添加其左邻居（考虑环形）
                            left_neighbor_idx = (best_match_idx - 1) % len(cameras_in_adjacent_layer)
                            self.graph[cam['id']].append(cameras_in_adjacent_layer[left_neighbor_idx]['id'])
                            
                            # 添加其右邻居（考虑环形）
                            right_neighbor_idx = (best_match_idx + 1) % len(cameras_in_adjacent_layer)
                            self.graph[cam['id']].append(cameras_in_adjacent_layer[right_neighbor_idx]['id'])

        # 去重
        for cam_id in self.graph:
            self.graph[cam_id] = sorted(list(set(self.graph[cam_id])))

    def generate_trajectory(self, target_points=10, verbose=False):
        """
        生成一条满足新规则的轨迹。
        【优化版：使用贪心策略替代DFS，避免组合爆炸】
        """
        if not self.is_initialized:
            logger.error("生成器未正确初始化，无法生成轨迹。")
            return None, None

        num_layers = len(self.layers)
        min_points = num_layers
        max_points = 10

        # 尝试几次以找到有效路径
        for _ in range(10): # 最多尝试10次
            # 1. 从最外层随机选择一个相机作为起点
            outer_layer_idx = num_layers - 1
            start_camera = random.choice(self.layers[outer_layer_idx])
            
            path = [start_camera['id']]
            visited_layers = {outer_layer_idx}
            current_id = start_camera['id']

            # 2. 贪心构建路径，直到达到最大长度或无法前进
            while len(path) < max_points:
                current_layer_idx = self._get_camera_layer_idx(current_id)
                
                # 获取所有合法的邻居
                potential_neighbors = []
                for neighbor_id in self.graph[current_id]:
                    if neighbor_id not in path: # 避免回头
                        neighbor_layer_idx = self._get_camera_layer_idx(neighbor_id)
                        # 必须向内或同层移动
                        if neighbor_layer_idx <= current_layer_idx:
                            potential_neighbors.append(neighbor_id)
                
                if not potential_neighbors:
                    break # 没有路可走了

                # 3. 贪心选择策略：优先选择能解锁新图层的邻居
                unvisited_layer_neighbors = []
                for nid in potential_neighbors:
                    if self._get_camera_layer_idx(nid) not in visited_layers:
                        unvisited_layer_neighbors.append(nid)
                
                if unvisited_layer_neighbors:
                    # 如果有能解锁新图层的邻居，从中随机选一个
                    next_id = random.choice(unvisited_layer_neighbors)
                else:
                    # 否则，在所有合法邻居中随机选一个
                    next_id = random.choice(potential_neighbors)

                # 更新路径
                path.append(next_id)
                visited_layers.add(self._get_camera_layer_idx(next_id))
                current_id = next_id

            # 4. 检查生成的路径是否满足条件
            if len(visited_layers) == num_layers and min_points <= len(path) <= max_points:
                # 找到一条合法路径，立即返回
                chosen_path_indices = path
                chosen_path_points = [self._get_camera_by_id(cam_id) for cam_id in chosen_path_indices]
                if verbose:
                    logger.info(f"成功生成轨迹 (贪心策略)，长度为 {len(chosen_path_indices)}。")
                    logger.info(f"轨迹路径（相机索引）: {chosen_path_indices}")
                return chosen_path_points, chosen_path_indices

        # 如果尝试多次后仍然失败
        logger.warning("未能找到满足所有条件的轨迹路径 (贪心策略)。")
        return None, None

    def _get_camera_layer_idx(self, camera_id):
        """辅助函数：根据相机ID查找其所在的层索引。"""
        for layer_idx, cameras in self.layers.items():
            for cam in cameras:
                if cam['id'] == camera_id:
                    return layer_idx
        return -1

    def _get_camera_by_id(self, camera_id):
        """辅助函数：根据相机ID查找完整的相机信息字典。"""
        for layer_idx, cameras in self.layers.items():
            for cam in cameras:
                if cam['id'] == camera_id:
                    return cam
        return None

    # 这个函数现在是主入口，旧的函数可以删除
    def generate_trajectory_for_dt2_integration_with_actual_cameras(self, target_points=10, verbose=False):
        """
        为与 dt2.py 集成而设计的主轨迹生成函数。
        """
        return self.generate_trajectory(target_points=target_points, verbose=verbose)

    def visualize_trajectory(self, trajectory_indices, save_path):
        """
        将相机布局和生成的轨迹可视化，并保存为图像。

        Args:
            trajectory_indices (list): 生成的轨迹中的相机索引列表。
            save_path (str): 可视化图像的保存路径 (e.g., "trajectory.png")。
        """
        if not self.is_initialized:
            logger.error("生成器未初始化，无法进行可视化。")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        def get_pos_from_transform(transform_matrix):
            """辅助函数：从变换矩阵中提取 (x, y, z) 坐标。"""
            if hasattr(transform_matrix, 'matrix'):
                m = transform_matrix.matrix
            else:
                m = transform_matrix
            
            import drjit as dr
            # 从变换矩阵中分离出不带梯度的 Dr.Jit 浮点数
            x_dr = dr.detach(m[0, 3])
            y_dr = dr.detach(m[1, 3])
            z_dr = dr.detach(m[2, 3])
            
            # 使用索引 [0] 从 Dr.Jit 标量中提取出 Python 内置的 float
            return x_dr[0], y_dr[0], z_dr[0]

        # 1. 绘制所有相机点
        all_cam_x = []
        all_cam_y = []
        for i in range(self.camera_info['camera_count']):
            transform = self.camera_info['camera_positions'][i]
            x, y, _ = get_pos_from_transform(transform)
            all_cam_x.append(x)
            all_cam_y.append(y)
            # 在每个点旁边标注其ID
            ax.text(x, y + 0.1, str(i), color="grey", fontsize=8)

        ax.scatter(all_cam_x, all_cam_y, c='lightblue', label='All Cameras', s=50)

        # 2. 绘制轨迹路径
        traj_x = []
        traj_y = []
        for cam_id in trajectory_indices:
            transform = self.camera_info['camera_positions'][cam_id]
            x, y, _ = get_pos_from_transform(transform)
            traj_x.append(x)
            traj_y.append(y)

        # 用红线连接轨迹点
        ax.plot(traj_x, traj_y, 'r-', label='Trajectory Path', marker='o', markersize=8)

        # 标记起点和终点
        if traj_x:
            ax.scatter(traj_x[0], traj_y[0], c='green', s=150, label='Start', zorder=5, alpha=0.8)
            ax.scatter(traj_x[-1], traj_y[-1], c='purple', s=150, label='End', zorder=5, alpha=0.8)

        # 3. 设置图表样式
        ax.set_title('Trajectory Visualization')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # 4. 保存图像
        try:
            plt.savefig(save_path, dpi=300)
            logger.info(f"轨迹可视化图像已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存可视化图像失败: {e}")
        finally:
            plt.close(fig) # 关闭图形，释放内存