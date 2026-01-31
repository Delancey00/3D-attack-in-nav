import os, PIL, csv
import gc
import numpy as np
import torch as ch
from torchvision.io import read_image
import mitsuba as mi
import drjit as dr
import time
from omegaconf import DictConfig
import logging

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
from detectron2.data.detection_utils import *
 
import traceback

from detectron2.data.datasets import register_coco_instances
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from random_path import TrajectoryGenerator
from camera_rotation import (
    get_smart_perturbation_params,  # 新增
    generate_perturbations,         # 新增  
    apply_camera_perturbation_with_position,  # 新增
    batch_apply_smart_perturbations,  # 新增
    extract_camera_radius_from_matrix,
)

# #9.2添加随机旋转角度
# import random   
# import numpy as np

register_coco_instances("my_classes", {}, "path/to/your.json", "path/to/images")
MetadataCatalog.get("my_classes").thing_classes = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor', 'fireplace', 'bathtub', 'mirror']
MetadataCatalog.get("my_classes").thing_dataset_id_to_contiguous_id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
MetadataCatalog.get("my_classes").thing_colors=[[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], 
                                                    [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30]]
def dt2_input(image_path:str)->dict:
    """
    Construct a Detectron2-friendly input for an image
    """
    input = {}
    filename = image_path
    adv_image = read_image(image_path, format="RGB")
    adv_image_tensor = ch.as_tensor(np.ascontiguousarray(adv_image.transpose(2, 0, 1)))
    height = adv_image_tensor.shape[1]
    width = adv_image_tensor.shape[2]
    instances = Instances(image_size=(height,width))
    instances.gt_classes = ch.Tensor([2])
    instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))
    input['image'] = adv_image_tensor    
    input['filename'] = filename
    input['height'] = height
    input['width'] = width
    input['instances'] = instances
    return input

def save_adv_image_preds(model \
    , dt2_config \
    , input \
    , instance_mask_thresh=0.7 \
    , target:int=None \
    , untarget:int=None
    , is_targeted:bool=True \
    , format="RGB" \
    , path:str=None):
    """
    Helper fn to save the predictions on an adversarial image
    attacked_image:ch.Tensor An attacked image
    instance_mask_thresh:float threshold pred boxes on confidence score
    path:str where to save image
    """ 
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dt2")
    model.train = False
    model.training = False

    model.proposal_generator.training = False
    model.roi_heads.training = False    
    with ch.no_grad():
        # 【核心修改在这里】
        # 直接使用函数参数'input'进行预测，而不是重新渲染。9.16
        adv_outputs = model([input])
        
        # 后续代码使用'input'中的图像数据进行可视化
        perturbed_image = input['image'].data.permute((1,2,0)).detach().cpu().numpy()
        pbi = ch.tensor(perturbed_image, requires_grad=False).detach().cpu().numpy()
        if format=="BGR":
            pbi = pbi[:, :, ::-1]
        v = Visualizer(pbi, MetadataCatalog.get("my_classes"),scale=1.0)
        instances = adv_outputs[0]['instances']
        categories = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor', 'fireplace', 'bathtub', 'mirror']
        things = np.array(categories)

        predicted_classes = things[instances.pred_classes.cpu().numpy().tolist()] 
        mask = instances.scores > instance_mask_thresh
        instances = instances[mask]
        out = v.draw_instance_predictions(instances.to("cpu"))
        target_pred_exists = target in instances.pred_classes.cpu().numpy().tolist()
        untarget_pred_not_exists = untarget not in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True  
    PIL.Image.fromarray(pred).save(path)
    if is_targeted and target_pred_exists:
        return True
    elif (not is_targeted) and (untarget_pred_not_exists):
        return True
    return False

def use_provided_cam_position(scene_file: str, sensor_key:str) -> np.array:
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)
    sensors = []
    for key in p.keys():
        if key.startswith(sensor_key) and key.endswith('to_world'):
            sensor = p[key]
            sensors.append(sensor)
    return np.array(sensors)

def generate_cube_scene_cam_positions() -> np.array:#9.26这个函数压根没用到
    """
    Load a mesh and use its vertices as camera positions
    e.g.,  Load a half-icosphere and separate the vertices by their height above target object
    each strata of vertices forms a 'ring' around the object. place cameras in a ring around the object
    and return camera positions (world_transform())
    """
    from mitsuba import ScalarTransform4f as T    
    def load_sensor_at_position(x,y,z):  
        origin = mi.ScalarPoint3f([x,y,z])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T().look_at(
                origin=origin,
                target=mi.ScalarPoint3f([0, -0.5, 0]),
                up=mi.ScalarPoint3f([0, 1, 0])
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
    sphere = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_mid.ply"
        },
    })
    sphere_outer = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_outer.ply"
        },
    })  
    sphere_inner = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_inner.ply"
        },
    })        
    ip = mi.traverse(sphere)
    ipv = np.array(ip["sphere.vertex_positions"])
    ipv  = np.reshape(ipv,(int(len(ipv)/3),3))    

    outer_sphere_ip = mi.traverse(sphere_outer)
    outer_sphere_ipv = np.array(outer_sphere_ip["sphere.vertex_positions"])
    outer_sphere_ipv = np.reshape(outer_sphere_ipv,(int(len(outer_sphere_ipv)/3),3))    

    inner_sphere_ip = mi.traverse(sphere_inner)
    inner_sphere_ipv = np.array(inner_sphere_ip["sphere.vertex_positions"])    
    inner_sphere_ipv = np.reshape(inner_sphere_ipv,(int(len(inner_sphere_ipv)/3),3))       
    # strata = np.array(list(set(np.round(ipv[:,1],3))))  
    # strata_2_cams =  ipv[np.where(np.round(ipv,3)[:,1] == strata[2])]    
    # strata_1_cams = ipv[np.where(np.round(ipv,3)[:,1] == strata[1])]    
    ipv_f = ipv[np.where(ipv[:,0] > 0)]
    outer_sphere_ipv_f = outer_sphere_ipv[np.where(outer_sphere_ipv[:,0] > 0)]
    inner_sphere_ipv_f = inner_sphere_ipv[np.where(inner_sphere_ipv[:,0] > 0)]
    cam_pos_ring = np.concatenate((ipv_f, outer_sphere_ipv_f, inner_sphere_ipv_f))
    positions = np.array([load_sensor_at_position(p[0], p[1], p[2]).world_transform() for p in cam_pos_ring])
    return positions

def gen_cam_positions(z,r,size) -> np.ndarray:
    """
    Generates # cam positions of length (size) in a circle of radius (r) 
    at the given latitude (z) on a sphere.  Think of the z value as the height above/below the object
    you want to render.  

    The sphere is centered at the origin (0,0,0) in the scene.  
    """
    if z > r:
        raise Exception("z value must be less than or equal to the radius of the sphere")
    lat_r = np.sqrt(r**2 - z**2)  # find latitude circle radius
    num_points = np.arange(1,size+1)
    angles = np.array([(2 * np.pi * p / size) for p in num_points])
    vertices = np.array([np.array([np.cos(a)*lat_r, z, np.sin(a)*lat_r]) for a in angles])
    return vertices

def load_sensor_at_position(x,y,z):  
    
    from mitsuba import ScalarTransform4f as T   
    origin = mi.ScalarPoint3f([x,y,z])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T().look_at(
            origin=origin,
            #target=[0, -0.5, 0],
            target=mi.ScalarPoint3f([0, 0, 0]),
            up=mi.ScalarPoint3f([0, 1, 0])
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })
#相机位置生成函数
def generate_cam_positions_for_lats(lats=[], r=None, size=None, reps_per_position=1):
    """
    Wrapper function to allow generation of camera angles for any list of arbitrary latitudes
    Note that the latitudes must be some z value within the pos/neg value of the radius in the sphere:
    so: {z | -r <= z <= r}
    """
    all_pos = gen_cam_positions(lats[0], r, size)
    for i in range(1,len(lats)):
        p = gen_cam_positions(lats[i], r, size)
        all_pos = np.concatenate((all_pos, p), axis=0)
      
    positions = np.array([load_sensor_at_position(p[0], p[1], p[2]).world_transform() for p in all_pos])
    positions = np.repeat(positions, reps_per_position)
    return positions    


#9.11调试半径代码
# 将这个调试函数添加到你的代码中，在optim_batch函数开始处调用
def debug_camera_positions(camera_positions, center_pos, sample_size=10):
    """
    调试相机位置，检查半径分布
    
    Args:
        camera_positions: 相机位置数组
        center_pos: 中心位置
        sample_size: 采样检查的相机数量
    """
    print("=== 相机位置调试信息 ===")
    print(f"中心位置: {center_pos}")
    print(f"总相机数量: {len(camera_positions)}")
    
    # 采样检查前sample_size个相机
    sample_indices = range(min(sample_size, len(camera_positions)))
    radius_distribution = {}
    
    for i in sample_indices:
        try:
            # 提取相机位置
            if hasattr(camera_positions[i], 'matrix'):
                matrix = camera_positions[i].matrix
                try:
                    cam_x = float(matrix[0, 3].numpy()) if hasattr(matrix[0, 3], 'numpy') else float(matrix[0, 3])
                    cam_y = float(matrix[1, 3].numpy()) if hasattr(matrix[1, 3], 'numpy') else float(matrix[1, 3])
                    cam_z = float(matrix[2, 3].numpy()) if hasattr(matrix[2, 3], 'numpy') else float(matrix[2, 3])
                except:
                    cam_x = matrix[0, 3].data if hasattr(matrix[0, 3], 'data') else matrix[0, 3]
                    cam_y = matrix[1, 3].data if hasattr(matrix[1, 3], 'data') else matrix[1, 3]
                    cam_z = matrix[2, 3].data if hasattr(matrix[2, 3], 'data') else matrix[2, 3]
                    cam_x, cam_y, cam_z = float(cam_x), float(cam_y), float(cam_z)
            else:
                try:
                    cam_x = float(camera_positions[i][0, 3].numpy()) if hasattr(camera_positions[i][0, 3], 'numpy') else float(camera_positions[i][0, 3])
                    cam_y = float(camera_positions[i][1, 3].numpy()) if hasattr(camera_positions[i][1, 3], 'numpy') else float(camera_positions[i][1, 3])
                    cam_z = float(camera_positions[i][2, 3].numpy()) if hasattr(camera_positions[i][2, 3], 'numpy') else float(camera_positions[i][2, 3])
                except:
                    cam_x = camera_positions[i][0, 3].data if hasattr(camera_positions[i][0, 3], 'data') else camera_positions[i][0, 3]
                    cam_y = camera_positions[i][1, 3].data if hasattr(camera_positions[i][1, 3], 'data') else camera_positions[i][1, 3]
                    cam_z = camera_positions[i][2, 3].data if hasattr(camera_positions[i][2, 3], 'data') else camera_positions[i][2, 3]
                    cam_x, cam_y, cam_z = float(cam_x), float(cam_y), float(cam_z)
            
            # 计算到中心的距离
            center_x, center_y, center_z = center_pos[0], center_pos[1], center_pos[2]
            
            # XZ平面距离（应该是主要的半径）
            radius_xz = np.sqrt((cam_x - center_x)**2 + (cam_z - center_z)**2)
            # XY平面距离（用于对比）
            radius_xy = np.sqrt((cam_x - center_x)**2 + (cam_y - center_y)**2)
            # 3D距离
            radius_3d = np.sqrt((cam_x - center_x)**2 + (cam_y - center_y)**2 + (cam_z - center_z)**2)
            
            print(f"相机{i:2d}: 位置=({cam_x:7.3f}, {cam_y:7.3f}, {cam_z:7.3f}), "
                       f"XZ半径={radius_xz:.3f}, XY半径={radius_xy:.3f}, 3D半径={radius_3d:.3f}")
            
            # 统计半径分布
            radius_rounded = round(radius_xz, 1)
            radius_distribution[radius_rounded] = radius_distribution.get(radius_rounded, 0) + 1
            
        except Exception as e:
            print(f"无法提取相机{i}的位置: {str(e)}")
    
    print("--- 半径分布统计 ---")
    for radius, count in sorted(radius_distribution.items()):
        print(f"半径 {radius}: {count} 个相机")
    
    print("=== 调试信息结束 ===\n")


# 同时添加一个检查智能扰动参数的函数
def debug_perturbation_mapping(camera_positions, center_pos, sample_size=10):
    """
    调试智能扰动参数映射
    """
    print("=== 扰动参数映射调试 ===")
    
    sample_indices = range(min(sample_size, len(camera_positions)))
    
    for i in sample_indices:
        try:
            # 计算实际半径
            actual_radius = extract_camera_radius_from_matrix(camera_positions[i], center_pos)
            
            # 获取扰动参数
            params = get_smart_perturbation_params(i, camera_positions, center_pos)
            
            print(f"相机{i:2d}: 实际半径={actual_radius:.3f}, "
                       f"映射到={params['radius']}, "
                       f"XZ扰动范围={params['position_xz']}")
            
        except Exception as e:
            print(f"相机{i}扰动参数获取失败: {str(e)}")
    
    print("=== 扰动参数调试结束 ===\n")


# #9.16添加显存监测函数
# import torch
# import gc
# import sys
# import traceback

# def print_gpu_usage(label=""):
#     """打印当前GPU显存使用情况"""
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated() / 1024**3  # GB
#         reserved = torch.cuda.memory_reserved() / 1024**3    # GB
#         print(f"[{label}] GPU显存 - 已分配: {allocated:.3f}GB, 已保留: {reserved:.3f}GB")
#     else:
#         print(f"[{label}] CUDA不可用")

# def print_variable_sizes(locals_dict, min_size_mb=1):
#     """打印变量占用的显存大小"""
#     print("=== 变量显存占用 ===")
#     variables = []
    
#     for name, obj in locals_dict.items():
#         size_mb = 0
#         var_info = ""
        
#         # PyTorch张量
#         if isinstance(obj, torch.Tensor):
#             if obj.is_cuda:
#                 size_mb = obj.element_size() * obj.numel() / (1024**2)
#                 var_info = f"Tensor{list(obj.shape)} on {obj.device}"
        
#         # 张量列表
#         elif isinstance(obj, list) and obj and isinstance(obj[0], torch.Tensor):
#             total_size = 0
#             for tensor in obj:
#                 if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
#                     total_size += tensor.element_size() * tensor.numel()
#             size_mb = total_size / (1024**2)
#             var_info = f"List[{len(obj)} tensors]"
        
#         # DrJit张量（如果有的话）
#         elif hasattr(obj, 'array') and hasattr(obj, 'shape'):
#             try:
#                 size_mb = sys.getsizeof(obj) / (1024**2)  # 粗略估计
#                 var_info = f"DrJit array"
#             except:
#                 pass
        
#         if size_mb >= min_size_mb:
#             variables.append((name, size_mb, var_info))
    
#     # 按大小排序
#     variables.sort(key=lambda x: x[1], reverse=True)
    
#     for name, size_mb, info in variables:
#         print(f"  {name}: {size_mb:.2f}MB ({info})")
    
#     if not variables:
#         print("  没有找到大于1MB的GPU变量")
#     print("=" * 30)

# def monitor_function(func):
#     """装饰器：监控函数的显存使用"""
#     def wrapper(*args, **kwargs):
#         print(f"\n--- 执行 {func.__name__} ---")
#         print_gpu_usage("开始前")
        
#         # 重置峰值统计
#         if torch.cuda.is_available():
#             torch.cuda.reset_peak_memory_stats()
        
#         try:
#             result = func(*args, **kwargs)
#             print_gpu_usage("完成后")
            
#             if torch.cuda.is_available():
#                 peak = torch.cuda.max_memory_allocated() / 1024**3
#                 print(f"峰值显存: {peak:.3f}GB")
            
#             return result
#         except Exception as e:
#             print_gpu_usage("异常时")
#             print(f"错误: {e}")
#             traceback.print_exc()
#             raise
    
#     return wrapper

# def detailed_tensor_info(tensor, name="tensor"):
#     """详细显示张量信息"""
#     if isinstance(tensor, torch.Tensor):
#         size_mb = tensor.element_size() * tensor.numel() / (1024**2)
#         print(f"{name}:")
#         print(f"  形状: {tensor.shape}")
#         print(f"  数据类型: {tensor.dtype}")
#         print(f"  设备: {tensor.device}")
#         print(f"  显存占用: {size_mb:.2f}MB")
#         print(f"  是否需要梯度: {tensor.requires_grad}")
#     else:
#         print(f"{name}: 不是张量")

# def clean_gpu_memory():
#     """清理GPU显存"""
#     print("清理GPU显存...")
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     print_gpu_usage("清理后")





def attack_dt2(cfg:DictConfig) -> None:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", default=1)
    DEVICE = "cuda:0"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dt2")
    batch_size = cfg.attack.batch_size
    eps = cfg.attack.eps
    eps_step =  cfg.attack.eps_step
    targeted =  cfg.attack.targeted
    target_class = cfg.attack.target_idx
    target_string = cfg.attack_class
    untargeted_class = cfg.attack.untarget_idx
    untargeted_string = cfg.untargeted_class
    iters = cfg.attack.iters
    spp = cfg.attack.samples_per_pixel
    multi_pass_rendering = cfg.attack.multi_pass_rendering
    multi_pass_spp_divisor = cfg.attack.multi_pass_spp_divisor
    scene_file = cfg.scene.path
    param_keys = cfg.scene.target_param_keys
    sensor_key = cfg.scene.sensor_key
    score_thresh = cfg.model.score_thresh_test
    weights_file = cfg.model.weights_file 
    model_config = cfg.model.config
    randomize_sensors = cfg.scenario.randomize_positions 
    scene_file_dir = os.path.dirname(scene_file)
    tex_paths = cfg.scene.textures
    multicam = cfg.multicam
    tmp_perturbation_path = os.path.join(f"{scene_file_dir}",f"textures/{target_string}_tex","tmp_perturbations")
    if os.path.exists(tmp_perturbation_path) == False:
        os.makedirs(tmp_perturbation_path)
    render_path = os.path.join(f"renders",f"{target_string}")
    if os.path.exists(render_path) == False:
        os.makedirs(render_path)
    preds_path = os.path.join("preds",f"{target_string}")
    if os.path.exists(preds_path) == False:
        os.makedirs(preds_path)    
    if multi_pass_rendering:
        logger.info(f"Using multi-pass rendering with {spp//multi_pass_spp_divisor} passes")
    mi.set_variant('cuda_ad_rgb')
    for tex in tex_paths:
        mitsuba_tex = mi.load_dict({
            'type': 'bitmap',
            'id': 'heightmap_texture',
            'filename': tex,
            'raw': True
        })
        mt = mi.traverse(mitsuba_tex)
    # FIXME - allow variant to be set in the configuration.
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)    
    k = param_keys
    keep_keys = [k for k in param_keys]
    k1 = f'{sensor_key}.to_world'
    k2 = f'{sensor_key}.film.size'
    keep_keys.append(k1)
    p.keep(keep_keys)
    p.update()
    orig_texs = []
    moves_matrices = use_provided_cam_position(scene_file=scene_file, sensor_key=sensor_key)
    if randomize_sensors:
        np.random.shuffle(moves_matrices)
    # load pre-trained robust faster-rcnn model
    dt2_config = get_cfg()
    dt2_config.merge_from_file(model_config)
    dt2_config.MODEL.WEIGHTS = weights_file
    dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    # FIXME - Get GPU Device form environment variable.
    dt2_config.MODEL.DEVICE = DEVICE
    model = build_model(dt2_config)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_config.MODEL.WEIGHTS)
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True




    #9.9新添加函数
    def apply_camera_rotation_unified(camera_positions, cam_idx, center_pos,enable_rotation=True, enable_position_perturbation=True,position_scale=1.0, system_name=""):
        """
        统一的相机旋转函数，供随机系统和轨迹系统共享使用
        复用已有的 camera_rotation.py 中的函数，最大化内存效率
        
        Args:
            camera_positions: 相机位置数组
            cam_idx: 相机索引
            center_pos: 中心位置 (x, y, z) 坐标元组
            enable_rotation: 是否启用旋转，默认True
            enable_position_perturbation: 是否启用位置扰动（仅XZ平面），默认True
            position_scale: 位置扰动缩放因子，默认1.0
            system_name: 系统名称，用于日志区分 ("随机" 或 "轨迹")
        
        Returns:
            perturbed_transform: 旋转后的相机变换矩阵
            rotation_info: 旋转信息字典
        
        Memory optimizations:
            - 复用 camera_rotation.py 中的所有函数
            - 避免重复的变换矩阵计算
            - 统一的错误处理和日志格式
        """
        if not enable_rotation and not enable_position_perturbation:
            # 不启用旋转时，直接返回原始变换
            if isinstance(camera_positions[cam_idx], mi.cuda_ad_rgb.Transform4f):
                original_transform = camera_positions[cam_idx]
            else:
                original_transform = mi.cuda_ad_rgb.Matrix4f(camera_positions[cam_idx])
                
            return original_transform, {
                "pitch": 0.0, "yaw": 0.0, "radius": 0.0,
                "position_offset": (0.0, 0.0, 0.0),
                "pitch_range": (0, 0), "yaw_range": (0, 0)
            }
        
        try:
            # 获取原始相机变换矩阵
            if isinstance(camera_positions[cam_idx], mi.cuda_ad_rgb.Transform4f):
                original_transform = camera_positions[cam_idx]
            else:
                original_transform = mi.cuda_ad_rgb.Matrix4f(camera_positions[cam_idx])
            
            # 使用新的智能扰动参数获取函数
            perturbation_params = get_smart_perturbation_params(cam_idx, camera_positions, center_pos)

            # 生成具体的扰动值
            perturbations = generate_perturbations(perturbation_params)
            
            # 构建角度扰动
            if enable_rotation:
                # 只旋转航向角，俯仰角固定为0（与用户需求一致）
                pitch_deg = 0.0  # 强制设为0，确保只旋转航向角
                yaw_deg = float(perturbations['yaw'])
            else:
                pitch_deg = 0.0
                yaw_deg = 0.0
            
            # 构建位置扰动
            if enable_position_perturbation:
                pos_offset = (
                    perturbations['pos_x'] * position_scale,
                    0.0,  # Y轴始终不变
                    perturbations['pos_z'] * position_scale
                )
            else:
                pos_offset = (0.0, 0.0, 0.0)

            # 应用组合扰动（角度 + 位置）
            perturbed_transform = apply_camera_perturbation_with_position(
                original_transform, pitch_deg, yaw_deg, pos_offset
            )
            
            # # 只旋转航偏角，俯仰角固定为0（与用户需求一致）
            # pitch_deg = 0.0  # 强制设为0，确保只旋转航偏角
            # yaw_deg = float(yaw_perturbation)
            
            # # 使用 camera_rotation.py 中的扰动应用函数
            # perturbed_transform = apply_camera_perturbation(original_transform, pitch_deg, yaw_deg)
            
            # 构建旋转信息字典
            perturbation_info = {
                "pitch": pitch_deg,
                "yaw": yaw_deg, 
                "radius": perturbation_params['radius'],
                "actual_radius": perturbation_params['actual_radius'],
                "position_offset": pos_offset,
                "pitch_range": perturbation_params['pitch'],
                "yaw_range": perturbation_params['yaw'],
                "position_xz_range": perturbation_params['position_xz'],
                "system": system_name,
                "enable_rotation": enable_rotation,
                "enable_position_perturbation": enable_position_perturbation
            }
            
            return perturbed_transform, perturbation_info
        
        except Exception as e:
            logger.error(f"{system_name}系统相机{cam_idx}扰动失败: {str(e)}")
            # 失败时返回原始变换
            return original_transform, {
                "pitch": 0.0, "yaw": 0.0, "radius": 0.0, 
                "position_offset": (0.0, 0.0, 0.0),
                "pitch_range": (0, 0), "yaw_range": (0, 0), 
                "error": str(e)
            }

    #9.14 新增函数从场景文件中提取相机信息
    def extract_camera_info_from_scene(scene_file: str, sensor_key: str, center_pos: tuple):
        """
        从Mitsuba场景文件中提取实际相机的布局信息
        
        功能说明：
        - 读取XML场景文件中的所有相机传感器
        - 计算每个相机相对于目标中心的位置、角度、半径
        - 返回完整的相机布局信息供轨迹生成器使用
        
        参数说明：
        scene_file (str): 场景XML文件的完整路径，例如 "scenes/scene.xml"
        sensor_key (str): 相机传感器的键名前缀，例如 "sensor" 
        center_pos (tuple): 目标物体的中心坐标 (x, y, z)，例如 (7.84387, -0.88, -3.82026)
        
        返回值：
        dict: 包含以下键的字典
            - 'center': 中心位置坐标 
            - 'camera_positions': 所有相机的变换矩阵列表
            - 'camera_count': 相机总数
            - 'camera_angles': 每个相机的角度列表（度）
            - 'camera_radii': 每个相机到中心的半径列表
            - 'height': 相机高度（假设所有相机同一高度）
            - 'average_radius': 平均半径
        
        使用时机：在创建轨迹生成器之前调用
        """
        
        # 加载Mitsuba场景
        scene = mi.load_file(scene_file)
        p = mi.traverse(scene)  # 遍历场景参数
        
        camera_positions = []
        camera_info = {
            'center': center_pos,
            'camera_positions': [],
            'camera_count': 0,
            'camera_angles': [],
            'camera_radii': [],
            'height': None,
            'average_radius': 0.0
        }
        
        # 提取所有相机的变换矩阵
        # 在Mitsuba场景中，相机的变换信息存储在 "传感器名.to_world" 键中
        logger.info("开始提取相机变换矩阵...")
        for key in p.keys():
            if key.startswith(sensor_key) and key.endswith('to_world'):
                sensor_matrix = p[key]
                camera_positions.append(sensor_matrix)
                logger.debug(f"找到相机: {key}")
        
        if not camera_positions:
            raise Exception(f"错误：未找到匹配的传感器，请检查sensor_key: {sensor_key}")
        
        camera_info['camera_count'] = len(camera_positions)
        camera_info['camera_positions'] = camera_positions
        
        # 分析每个相机的几何位置
        logger.info(f"分析 {len(camera_positions)} 个相机的几何布局...")
        for i, cam_matrix in enumerate(camera_positions):
            # 从变换矩阵中提取位置坐标
            # Mitsuba的变换矩阵是4x4矩阵，位置信息在第4列（索引3）的前3行
            try:
                if hasattr(cam_matrix, 'matrix'):
                    matrix = cam_matrix.matrix
                    try:
                        # 对于DrJit类型，先转换为numpy或python类型
                        if hasattr(matrix[0, 3], 'numpy'):
                            cam_x = float(matrix[0, 3].numpy())
                            cam_y = float(matrix[1, 3].numpy())
                            cam_z = float(matrix[2, 3].numpy())
                        elif hasattr(matrix[0, 3], 'data'):
                            cam_x = float(matrix[0, 3].data)
                            cam_y = float(matrix[1, 3].data)
                            cam_z = float(matrix[2, 3].data)
                        else:
                            # 尝试直接访问底层数值
                            import drjit as dr
                            cam_x = float(dr.detach(matrix[0, 3]))
                            cam_y = float(dr.detach(matrix[1, 3]))
                            cam_z = float(dr.detach(matrix[2, 3]))
                    except Exception as drjit_error:
                        logger.warning(f"DrJit转换失败，尝试其他方法: {str(drjit_error)}")
                        # 备用方法：转换为字符串再转换为float
                        cam_x = float(str(matrix[0, 3]).split()[0])
                        cam_y = float(str(matrix[1, 3]).split()[0])  
                        cam_z = float(str(matrix[2, 3]).split()[0])
                else:
                    # 处理其他矩阵格式
                    if hasattr(cam_matrix[0, 3], 'numpy'):
                        cam_x = float(cam_matrix[0, 3].numpy())
                        cam_y = float(cam_matrix[1, 3].numpy())
                        cam_z = float(cam_matrix[2, 3].numpy())
                    else:
                        import drjit as dr
                        cam_x = float(dr.detach(cam_matrix[0, 3]))
                        cam_y = float(dr.detach(cam_matrix[1, 3]))
                        cam_z = float(dr.detach(cam_matrix[2, 3]))
            except Exception as e:
                # 提供更详细的调试信息
                logger.debug(f"相机 {i} 矩阵类型: {type(cam_matrix)}")
                if hasattr(cam_matrix, 'matrix'):
                    logger.debug(f"矩阵元素类型: {type(cam_matrix.matrix[0, 3])}")
                    logger.debug(f"矩阵元素值: {cam_matrix.matrix[0, 3]}")
                continue
            
            # 计算到中心的距离和角度
            center_x, center_y, center_z = center_pos
            
            # 计算XY平面上的半径（假设相机围绕Z轴排列）
            radius_xy = np.sqrt((cam_x - center_x)**2 + (cam_y - center_y)**2)
            
            # 计算XY平面上的角度（以X轴正方向为0度，逆时针为正）
            angle_rad = np.arctan2(cam_y - center_y, cam_x - center_x)
            angle_deg = np.degrees(angle_rad)
            # 将角度标准化到 [0, 360) 范围
            angle_deg = angle_deg % 360
            
            camera_info['camera_radii'].append(radius_xy)
            camera_info['camera_angles'].append(angle_deg)
            
            # 记录高度（假设所有相机在同一高度）
            if camera_info['height'] is None:
                camera_info['height'] = cam_z
            
            logger.debug(f"相机 {i}: 位置({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}), "
                        f"半径={radius_xy:.3f}, 角度={angle_deg:.1f}°")
        
        # 计算统计信息
        if camera_info['camera_radii']:
            camera_info['average_radius'] = np.mean(camera_info['camera_radii'])
        
        logger.info(f"相机信息提取完成:")
        logger.info(f"  - 相机数量: {camera_info['camera_count']}")
        logger.info(f"  - 平均半径: {camera_info['average_radius']:.3f}")
        logger.info(f"  - 相机高度: {camera_info['height']:.3f}")
        logger.info(f"  - 角度范围: {min(camera_info['camera_angles']):.1f}° ~ {max(camera_info['camera_angles']):.1f}°")
        
        return camera_info

    def optim_batch(scene, params, opt, orig_texs, non_diff_params, batch_size, camera_positions, spp, k, label, unlabel, iters, alpha, epsilon, targeted=False):
        #9.16函数开始时检查
        #print_gpu_usage("optim_batch 开始")
        #center_pos = (7.84387, -3.82026,0.88)tv的中心坐标
        #center_pos = (3.0431, 2.4604,0.88) toilet的中心坐标
        #center_pos  = (8.8631,2.3345, 0.88) # bed的中心坐标
        center_pos = (9.7831,-2.80345, 0.88)#plant的中心坐标
        # 尝试从场景文件提取实际相机信息
        camera_info = None
        try:
            # 注意：这里需要访问场景文件路径和传感器键
            # scene_file 和 sensor_key 变量在attack_dt2函数开头已定义
            camera_info = extract_camera_info_from_scene(scene_file, sensor_key, center_pos)
            logger.info(" 成功提取实际相机布局信息")#9.16
            #print_gpu_usage("提取相机信息后")
        except Exception as e:
            logger.warning(f" 无法提取相机信息，将使用默认参数: {str(e)}")
            logger.warning("这可能是因为场景文件路径错误或传感器键名不匹配")
            camera_info = None
    
        # # 创建轨迹生成器
        # # 如果camera_info为None，TrajectoryGenerator会自动使用默认参数
        # trajectory_generator = TrajectoryGenerator(camera_info=camera_info,radius_tolerance=0.1, angle_tolerance=4.5)
        # #print_gpu_usage("创建轨迹生成器后")#9.16

        # if camera_info is not None:
        #     logger.info(f"✓ 轨迹生成器初始化完成，使用实际相机参数")
        #     logger.info(f"  - 检测到 {camera_info['camera_count']} 个相机")
        #     logger.info(f"  - 平均半径: {camera_info['average_radius']:.2f}")
        # else:
        #     logger.info("✓ 轨迹生成器初始化完成，使用默认参数")

        # #创建轨迹生成器实例
        # #trajectory_generator = TrajectoryGenerator()
        # #logger.info(f"轨迹生成器已初始化")
        if targeted:
            assert(label is not None)
        print('len(camera_positions)=',len(camera_positions))
        #assert(batch_size <= len(camera_positions))
        success = False
        # wrapper function that models the input image and returns the loss
        # TODO - 2 model input should accept a batch
        #定义损失函数
        @dr.wrap_ad(source='drjit', target='torch')
        def model_input(x, target, gt_boxes):
            losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]
            target_loss_idx = [0,1,2,3]  # Target only `loss_cls` loss
            loss_weights = [0.7,0.1,0.1,0.1]  # Updated loss weights
            x = ch.permute(x, (0, 3, 1, 2)).requires_grad_()
            x.retain_grad()
            height = x.shape[2]
            width = x.shape[3]
            instances = Instances(image_size=(height, width))
            instances.gt_classes = target.long()
            inputs = []
            for i in range(x.shape[0]):
                instances.gt_boxes = Boxes(ch.tensor([gt_boxes[i]]))
                # print(instances.gt_boxes)
                input = {'image': x[i], 'filename': '', 'height': height, 'width': width, 'instances': instances}
                inputs.append(input)
            with EventStorage(0) as storage:
                losses = model(inputs) 
                # 更新损失权重
                # print('losses=',losses)
                # print('losses_name=',losses_name)
                # print('target_loss_idx=',target_loss_idx)
                # print('loss_weights=',loss_weights)
                weighted_losses = [losses[losses_name[tgt_idx]] * weight for tgt_idx, weight in zip(target_loss_idx, loss_weights)]
                loss = sum(weighted_losses).requires_grad_()
            del x
            return loss
        #8.27新添加一个函数
        def _safe_get_instances(outputs):
            """
            兼容几种可能的返回结构，统一拿到 Instances：
            - [ { 'instances': Instances, ... } ]        ← 正常推理（eval）
            - [ [ { 'instances': Instances, ... } ] ]    ← 某些包裹一层 list 的情况
            - { 'instances': Instances, ... }            ← 少见（训练/错误模式）
            """
            # 先把 outputs 展平成列表
            o = outputs
            if isinstance(o, dict) and 'instances' in o:
                return o['instances']
            if isinstance(o, list):
                # 展平一层
                while isinstance(o, list) and len(o) > 0:
                    if isinstance(o[0], dict) and 'instances' in o[0]:
                        return o[0]['instances']
                    elif isinstance(o[0], list):
                        o = o[0]
                    else:
                        break
            raise TypeError(f"无法从模型输出中提取 instances，outputs 结构为: {type(outputs)} -> {type(outputs[0]) if isinstance(outputs,list) and outputs else 'N/A'}")

        # #添加轨迹梯度函数
        # def compute_trajectory_gradient(trajectory_camera_indices, current_params, current_orig_texs, camera_positions, camera_idx, target_class, untargeted_class, targeted, label, unlabel, shared_diff_params, shared_non_diff_params, shared_opt, center_pos):
        #     # —— 这里是 compute_trajectory_gradient 的函数开头 —— 
        #     if (not isinstance(trajectory_camera_indices, (list, tuple)) 
        #         or len(trajectory_camera_indices) == 0):
        #         raise ValueError("trajectory_camera_indices 应为非空的 list[int]。")

        #     # 若误把“轨迹点列表 list[dict]”传进来了，直接报清晰错误
        #     if isinstance(trajectory_camera_indices[0], dict):
        #         bad_keys = list(trajectory_camera_indices[0].keys())
        #         raise TypeError(
        #             f"收到的是轨迹点列表(list[dict])，而不是相机索引列表。示例键：{bad_keys}。"
        #             "请传入 TrajectoryGenerator.generate_trajectory_for_dt2_integration 返回的第二个值（相机索引列表）。"
        #         )

            
        #     """
        #     计算轨迹系统的梯度
        #     参数：
        #         trajectory_camera_indices: 轨迹相机索引列表
        #         current_params: 当前场景参数
        #         current_orig_texs: 原始纹理参数
        #     返回：
        #         trajectory_gradient: 轨迹平均梯度，如果失败返回None
        #     """
        #     logger.info(f"开始计算轨迹梯度，轨迹包含{len(trajectory_camera_indices)}个相机")
        #     # 【修改】复用共享参数，不再重复创建
        #     traj_diff_params = shared_diff_params
        #     traj_non_diff_params = shared_non_diff_params  
        #     traj_opt = shared_opt


        #     #启用梯度计算
        #     for i,k in enumerate(param_keys):
        #         dr.enable_grad(current_orig_texs[i])
        #         dr.enable_grad(traj_opt[k])
        #         traj_opt[k].set_label_(f"{k}_traj_bitmap")
            
        #     N_traj, H_traj, W_traj, C_traj = 1, traj_non_diff_params[k2][0], traj_non_diff_params[k2][1], 3
        #     traj_gradients = []
        #     successful_trajectory_detections = 0
            
        #     #遍历轨迹上每个相机
        #     for step, traj_cam_idx in enumerate(trajectory_camera_indices):
        #         logger.info(f"处理轨迹中的第{step+1}/{len(trajectory_camera_indices)}个相机(相机索引：{traj_cam_idx})")

        #         #设置相机位置
        #         perturbed_transform, rotation_info = apply_camera_rotation_unified(
        #         camera_positions, traj_cam_idx, center_pos, enable_rotation=True, enable_position_perturbation=True, position_scale=1.0, system_name="轨迹" )

        #         # 设置旋转后的轨迹相机位置
        #         traj_non_diff_params[k1].matrix = perturbed_transform.matrix
        #         traj_non_diff_params.update()

        #         # 轨迹系统的日志输出
        #         logger.info(f"轨迹相机{traj_cam_idx} - 半径: {rotation_info['radius']:.2f}, "
        #                 f"航偏角扰动: {rotation_info['yaw']:.2f}°, "
        #                 f"位置偏移: ({rotation_info['position_offset'][0]:.3f}, {rotation_info['position_offset'][2]:.3f}), "
        #                 f"系统: {rotation_info['system']}")
        #         current_params.update(traj_opt) 
                
        #         #渲染
        #         prb_integrator = mi.load_dict({'type': 'prb'})
        #         if multi_pass_rendering:
        #             # achieve the affect of rendering at a high sample-per-pixel (spp) value 
        #             # by rendering multiple times at a lower spp and averaging the results
        #             # render_passes = 16 # TODO - make this a config param
        #             mini_pass_spp = spp//multi_pass_spp_divisor#计算每次渲染的采样率
        #             render_passes = mini_pass_spp# #计算渲染次数
        #             mini_pass_renders = dr.empty(dr.cuda.ad.Float, render_passes * H * W * C)
        #             for i in range(render_passes):
        #                 #9.16
        #                 # if i == 0 and processed_cameras <= 2:  # 只在前几个相机的第一次渲染时检查
        #                 #     print_gpu_usage(f"多次渲染第{i+1}次")
        #                 seed = np.random.randint(0,1000)+i
        #                 img_i =  mi.render(scene, params=current_params, spp=mini_pass_spp, sensor=camera_idx[traj_cam_idx], seed=seed, integrator=prb_integrator)
        #                 s_index = i * (H * W * C)#当前的渲染结果的起始索引
        #                 e_index = (i+1) * (H * W * C)
        #                 mini_pass_index = dr.arange(dr.cuda.ad.UInt, s_index, e_index)
        #                 img_i = dr.ravel(img_i)#将渲染结果展平为一维张量
        #                 dr.scatter(mini_pass_renders, img_i, mini_pass_index)
        #             @dr.wrap_ad(source='drjit', target='torch')
        #             def stack_imgs(imgs):#用于将多次渲染的结果堆叠在一起
        #                 imgs = imgs.reshape((render_passes, H, W, C))
        #                 imgs = ch.mean(imgs,axis=0)#对重塑后的张量沿着第一个维度进行平均
        #                 return imgs
        #             mini_pass_renders = dr.cuda.ad.TensorXf(mini_pass_renders, dr.shape(mini_pass_renders))
        #             traj_img = stack_imgs(mini_pass_renders)
        #         else: # dont use multi-pass rendering
        #             traj_img =  mi.render(scene, params=current_params, spp=spp, sensor=camera_idx[traj_cam_idx], seed=cam_idx+100, integrator=prb_integrator)
                
        #         #9.16
        #         # 渲染后检查
        #         # if processed_cameras <= 2:
        #         #     print_gpu_usage(f"相机{cam_idx}渲染后")
        #         #     detailed_tensor_info(img, f"渲染图像{cam_idx}")
                
        #         #traj_img = mi.render(scene, params=current_params,spp=spp,sensor=camera_idx[traj_cam_idx],seed=traj_cam_idx+100,integrator=prb_integrator)
        #         #保存轨迹渲染图像
        #         traj_img.set_label_(f"trajectory_image_{step:03d}")
        #         traj_rendered_img_path = os.path.join(render_path, f"trajectory_render_{step:03d}.png")
        #         mi.util.write_bitmap(traj_rendered_img_path, data=traj_img, write_async=False)

        #         #目标检测
        #         traj_rendered_img_input = dt2_input(traj_rendered_img_path)
        #         model.train = False
        #         model.training = False
        #         model.proposal_generator.training = False
        #         model.roi_heads.training = False
        #         traj_outputs = model([traj_rendered_img_input])
        #         traj_instances = _safe_get_instances(traj_outputs)  #8.27改写成了函数的形式

        #         #检查检测结果
        #         if targeted:
        #             traj_mask = (traj_instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST) & (traj_instances.pred_classes == target_class)
        #         else:
        #             traj_mask = (traj_instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST) & (traj_instances.pred_classes == untargeted_class)

        #         traj_filtered_instances = traj_instances[traj_mask]
        #         traj_detection_successful = len(traj_filtered_instances) > 0

        #         if traj_detection_successful:
        #             successful_trajectory_detections += 1
        #             traj_current_gt_box = traj_filtered_instances.pred_boxes.tensor[0].tolist()
        #             logger.info(f"轨迹检测成功，累积梯度")
                
        #             #计算梯度
        #             traj_imgs = dr.empty(dr.cuda.ad.Float, N_traj * H_traj * W_traj * C_traj)
        #             traj_img_flat = dr.ravel(traj_img)
        #             traj_start_index = 0
        #             traj_end_index = H_traj * W_traj * C_traj
        #             traj_index = dr.arange(dr.cuda.ad.UInt, traj_start_index, traj_end_index)
        #             dr.scatter(traj_imgs, traj_img_flat, traj_index)

        #             traj_imgs = dr.cuda.ad.TensorXf(dr.cuda.ad.Float(traj_imgs), shape=(N_traj, H_traj, W_traj, C_traj))
        #             if not dr.grad_enabled(traj_imgs):
        #                 dr.enable_grad(traj_imgs)

        #             #设置目标
        #             if targeted:
        #                 traj_target = dr.cuda.ad.TensorXf([label], shape=(1,))
        #             else:
        #                 traj_target = dr.cuda.ad.TensorXf([unlabel], shape=(1,))

        #             traj_detected_boxes = [traj_current_gt_box]

        #             #计算损失
                    
        #             # 重置模型状态
        #             model.train = True
        #             model.training = True
        #             model.proposal_generator.training = True
        #             model.roi_heads.training = True
        #             traj_loss = model_input(traj_imgs, traj_target, traj_detected_boxes)
        #             logger.info(f"轨迹损失值: {str(traj_loss.array[0])[0:7]}")
                    
        #             gc.collect()
        #             ch.cuda.empty_cache()
        #             # 反向传播
        #             dr.enable_grad(traj_loss)
        #             dr.backward(traj_loss)
                    
        #             # 获取梯度
        #             HH_traj, WW_traj = dr.shape(dr.grad(traj_opt[param_keys[0]]))[0], dr.shape(dr.grad(traj_opt[param_keys[0]]))[1]
        #             traj_grad = ch.Tensor(dr.grad(traj_opt[param_keys[0]]).array).view((HH_traj, WW_traj, C_traj))
        #             traj_gradients.append(traj_grad)
                    
        #             #9.16
        #             # 【在此处添加显存清理代码】
        #             # 清理本次循环中创建的、不再需要的大张量及其计算图
        #             del traj_img, traj_imgs, traj_loss, traj_grad, traj_outputs, traj_instances
        #             gc.collect()
        #             if ch.cuda.is_available():
        #                 ch.cuda.empty_cache()
        #             logger.debug(f"轨迹相机 {traj_cam_idx} 的显存已清理")
                    
        #             # 清理内存
        #             gc.collect()
        #             ch.cuda.empty_cache()
        #         else:
        #             #traj_current_gt_box = [0.0, 0.0, float(512), float(512)]
        #             logger.info(f"轨迹检测失败，跳过梯度累积（与随机系统保持一致）")

        #     # 计算平均轨迹梯度
        #     if traj_gradients:
        #         trajectory_gradient = sum(traj_gradients) / len(traj_gradients)
        #         logger.info(f"   -轨迹梯度计算完成:")
        #         logger.info(f"   - 处理相机数: {len(trajectory_camera_indices)}")
        #         logger.info(f"   - 成功检测数: {successful_trajectory_detections}")
        #         logger.info(f"   - 累积梯度数: {len(traj_gradients)}")
        #         return trajectory_gradient
        #     else:
        #         logger.warning(" 未获得任何轨迹梯度")
        #         return None
                


        #添加双梯度融合函数
        def fuse_and_update_texture(img, random_gradients, trajectory_gradient, current_params, current_orig_texs, update_count):
            """
            融合随机梯度和轨迹梯度，更新纹理
            参数：
                random_gradients: 随机系统累积的梯度列表
                trajectory_gradient: 轨迹系统的平均梯度
                current_params: 当前场景参数
                current_orig_texs: 原始纹理参数
                update_count: 更新次数（用于文件命名）
            返回：
                bool: 更新是否成功
            """
            logger.info("  开始融合双梯度并更新纹理...")

            try:
                # 计算随机系统平均梯度
                if random_gradients:
                    random_avg_gradient = sum(random_gradients) / len(random_gradients)
                    logger.info(f" 随机系统平均梯度已计算 (来自{len(random_gradients)}个梯度)")
                else:
                    random_avg_gradient = None
                    logger.warning(" 无可用随机梯度")
                
                # 检查可用梯度
                available_gradients = []
                gradient_sources = []
                
                if random_avg_gradient is not None:
                    available_gradients.append(random_avg_gradient)
                    gradient_sources.append("随机系统")
                
                if trajectory_gradient is not None:
                    available_gradients.append(trajectory_gradient)
                    gradient_sources.append("轨迹系统")
                
                if not available_gradients:
                    logger.error(" 没有可用梯度，跳过更新")
                    return False

                 # 梯度融合
                if len(available_gradients) == 2:
                    # 双梯度融合：等权重平均
                    fused_gradient = (available_gradients[0] + available_gradients[1]) / 2
                    logger.info(" 使用双梯度融合 (随机 + 轨迹) / 2")
                else:
                    # 单梯度使用
                    fused_gradient = available_gradients[0]
                    logger.info(f" 使用单一梯度: {gradient_sources[0]}")
                
                logger.info(f"融合梯度shape: {fused_gradient.shape}")

                # 纹理更新（使用原有算法）
                for i, k in enumerate(param_keys):
                    C = 3
                    HH, WW = fused_gradient.shape[0], fused_gradient.shape[1]
                    
                    tex = ch.Tensor(current_params[k].array).view((HH, WW, C))
                    _orig_tex = ch.Tensor(current_orig_texs[i].array).view((HH, WW, C))
                    
                    # 梯度归一化
                    l = len(fused_gradient.shape) - 1
                    g_norm = ch.norm(fused_gradient.view(fused_gradient.shape[0], -1), dim=1).view(-1, *([1]*l))
                    scaled_grad = fused_gradient / (g_norm + 1e-10)
                    
                    # 有目标攻击：最小化损失
                    if targeted:
                        scaled_grad = -scaled_grad
                    
                    # 梯度步骤
                    tex = tex + scaled_grad * alpha
                    delta = tex - _orig_tex
                    
                    # 投影到L2球
                    delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                    tex = _orig_tex + delta

                    # # convert back to mitsuba dtypes            
                    # tex = dr.cuda.ad.TensorXf(tex.to(DEVICE))
                    # # divide by average brightness
                    # scaled_img = img / dr.mean(dr.detach(img))
                    # tex = tex / dr.mean(scaled_img)         
                    # tex = dr.clamp(tex, 0, 1)
                    # current_params[k] = tex     
                    # dr.enable_grad(current_params[k])
                    # current_params.update()
                    # perturbed_tex = mi.Bitmap(current_params[k])
                    
                    # save_path = os.path.join(tmp_perturbation_path, f"{k}_dual_update_{update_count}.png")
                    # mi.util.write_bitmap(save_path, 
                    #                     data=perturbed_tex, write_async=False)
                    # logger.info(f"  更新后的纹理已保存至: {save_path}")
                
                    # gc.collect()
                    # ch.cuda.empty_cache()9.25注释
                    
                    # --- 核心修改：使用 dr.scatter 实现就地更新 ---9.22
                    # 1. 将PyTorch计算出的新纹理值转换为Dr.Jit张量
                    new_tex_values = dr.cuda.ad.TensorXf(tex.to(DEVICE))
                    
                    # 2. 获取优化器正在管理的原始纹理张量
                    original_tex_param = opt[k]
                    
                    # 3. 使用 dr.scatter 将新值写入原始张量
                    #    这会保持优化器与参数之间的连接
                    dr.scatter(
                        target=original_tex_param.array,
                        value=dr.clamp(new_tex_values.array, 0, 1),
                        index=dr.arange(dr.cuda.ad.UInt32, dr.width(original_tex_param))
                    )
                    
                    # 4. 更新场景以反映参数的变化
                    current_params.update()
                    
                    # 保存更新后的纹理
                    perturbed_tex = mi.Bitmap(current_params[k])
                    save_path = os.path.join(tmp_perturbation_path, f"{k}_dual_update_{update_count}.png")
                    mi.util.write_bitmap(save_path, 
                                        data=perturbed_tex, write_async=False)
                    logger.info(f"  更新后的纹理已保存至: {save_path}")
                
                logger.info(" 纹理更新完成")
                return True

            except Exception as e:
                logger.error(f"双梯度融合更新失败: {str(e)}")
                return False
                

            
        # #初始化场景参数9.22注释
        # params = mi.traverse(scene)
        # for k in param_keys:
        #     if isinstance(params[k], dr.cuda.ad.TensorXf):
        #         # use Float if dealing with just texture colors (not a texture map)
        #         orig_tex = dr.cuda.ad.TensorXf(params[k])
        #     elif isinstance(params[k], dr.cuda.ad.Float):
        #         orig_tex = dr.cuda.ad.Float(params[k])        
        #     else:
        #         raise Exception("Unrecognized Differentiable Parameter Data Type.  Should be one of dr.cuda.ad.Float or dr.cuda.ad.TensorXf")
        #     orig_tex.set_label_(f"{k}_orig_tex")
        #     orig_texs.append(orig_tex)
        # indicate sensors to use in producing the perturbation
        # e.g., [0,1,2,3] will use sensors 0-3 focus on Taxi/Cement Truck in 'intersection_taxi.xml'
        # sensor 10 is focused on stop sign.
        sensors = [0]
        # if iters % len(sensors) != 0:
        #     print("uneven amount of iterations provided for sensors! Some sensors will be used more than others\
        #         during attack")
        # if only one camera in the scene, then this idx will be repeated for each iter
        camera_idx = ch.Tensor(np.array(sensors)).repeat(len(camera_positions)).to(dtype=ch.uint8).numpy().tolist()
        # one matrix per camera position that we want to render from, equivalent to batch size
        # e.g., batch size of 5 = 5 required camera positions
        
        current_gt_box = [0.0, 0.0, float(512), float(512)]
        # current_gt_box = [428.0, 224.5, float(148), float(151)]
        #9.16 检查重要变量
        # print("=== 初始变量检查 ===")
        # print_variable_sizes(locals())
        #随机化相机处理顺序
        #skipped_camera_indices = [] 
        total_cameras = len(camera_positions)
        shuffled_camera_indices = list(range(total_cameras))
        np.random.shuffle(shuffled_camera_indices)
        print(f"Processing {total_cameras} cameras in shuffled order")
        #修改原来的梯度累积逻辑
        target_gradient_count = 5
        random_accumulate_grad = []#改为随机系统梯度累积
        successful_detection_count = 0#成功检测的视角数量
        processed_cameras = 0#已经处理的相机数量
        total_successful_views = 0#成功视角的总数量
        update_count = 0#更新次数

        logger.info(" 开始双梯度系统攻击...")
        logger.info(f"目标: 每累积{target_gradient_count}个随机成功检测梯度，立即生成轨迹并融合更新")



        # # keep 2 sets of parameters because we only need to differentiate wrt texture
        # diff_params = mi.traverse(scene)
        # non_diff_params = mi.traverse(scene)
        # diff_params.keep([k for k in param_keys])
        # non_diff_params.keep([k1,k2])
        # # optimizer is not used but necessary to instantiate to get gradients from diff rendering.
        # opt = mi.ad.Adam(lr=0.1, params=diff_params) 

        # for i,k in enumerate(param_keys):
        #     dr.enable_grad(orig_texs[i])#启用梯度计算进行纹理优化
        #     dr.enable_grad(opt[k])#启用梯度计算对场景中的参数进行优化
        #     opt[k].set_label_(f"{k}_bitmap")   9.22注释掉

        N, H, W, C = batch_size, non_diff_params[k2][0], non_diff_params[k2][1], 3    
        #center_pos = (7.84387, -3.82026,0.88 )  # 中心位置也移出循环
        #center_pos = (3.0431, 2.4604,0.88)#馬桶
        #center_pos = (8.8631,2.3345,0.88) # bed的中心坐标
        center_pos = (9.7831,-2.80345, 0.88)#plant的中心坐标
        # 主循环前检查9.16
        #print_gpu_usage("主循环开始前")
        #开始遍历所有的视角
        for cam_idx in shuffled_camera_indices:
            processed_cameras += 1
            # 每处理几个相机就检查一次9.16
            # if processed_cameras % 3 == 0:
            #     print(f"\n--- 处理进度: {processed_cameras}/{total_cameras} ---")
            #     print_gpu_usage(f"相机{processed_cameras}")
                
            #     # 检查当前变量占用
            #     print_variable_sizes(locals(), min_size_mb=5)  # 只显示5MB以上的变量
            logger.info(f" 随机处理相机 {processed_cameras}/{total_cameras} (cam_idx:{cam_idx})")
            #让我来输出一下相机的真实面目##
            # print(f"=== 随机系统相机 {cam_idx} 矩阵信息 ===")
            # if cam_idx < len(camera_positions):
            #     original_matrix = camera_positions[cam_idx]
            #     print(f"原始相机矩阵 [{cam_idx}]:")
            #     if hasattr(original_matrix, 'matrix'):
            #         print(f"  矩阵类型: {type(original_matrix.matrix)}")
            #         print(f"  矩阵内容:\n{original_matrix.matrix}")
            #     else:
            #         print(f"  矩阵类型: {type(original_matrix)}")
            #         print(f"  矩阵内容:\n{original_matrix}")
            # else:
            #     print(f"相机索引 {cam_idx} 超出范围!")

            #结束输出一下相机的真实面目##
            # 使用统一的相机旋转函数（随机系统）
            perturbed_transform, rotation_info = apply_camera_rotation_unified(
                camera_positions, cam_idx, center_pos, enable_rotation=True, enable_position_perturbation=True, position_scale=1.0, system_name="随机")

            # 渲染前检查（只对前几个相机详细监控）9.16 
            #if processed_cameras <= 2:
                # print_gpu_usage(f"相机{cam_idx}渲染前")
                # print("当前主要变量:")
                # if 'img' in locals():
                #     detailed_tensor_info(img, "渲染图像")
                # if random_accumulate_grad:
                #     total_grad_size = sum(g.element_size() * g.numel() for g in random_accumulate_grad) / (1024**2)
                #     print(f"累积梯度: {len(random_accumulate_grad)}个, 总大小: {total_grad_size:.2f}MB")
            
            
            # 设置旋转后的相机位置
            non_diff_params[k1].matrix = perturbed_transform.matrix
            non_diff_params.update()

            # 统一的日志输出
            print(f"随机相机 {cam_idx} - 半径: {rotation_info['radius']:.2f}, "
                    f"航偏角扰动: {rotation_info['yaw']:.2f}°, "
                    f"位置偏移: ({rotation_info['position_offset'][0]:.3f}, {rotation_info['position_offset'][2]:.3f}), "
                    f"航偏角范围: ±{rotation_info['yaw_range'][1]}°")  

            params.update(opt)

            imgs = dr.empty(dr.cuda.ad.Float, N * H * W * C)
            # gt_boxed
            detected_boxes = []
            #for b in range(0, batch_size):
            b=0
            gc.collect()
            ch.cuda.empty_cache()
                # EOT Strategy
                # set the camera position, render & attack
                # if cam_idx > len(sampled_camera_positions)-1:
                #     logger.info(f"Successfull detections on all {len(sampled_camera_positions)} positions.")
                #     break
                    #return
                # if batch_size > 1: # sample from random camera positions
                #     cam_idx = b
            #相机位置设置
            # print(f"Setting camera at position {cam_idx}")
            # print('camera_positions',camera_positions)
            # print('camera_positions[cam_idx]=',camera_positions[cam_idx])
            
            # print(f"Camera {cam_idx} - Original transform matrix:")
            # print(original_transform.matrix)
            # #输出渲染之前的图像
            # non_diff_params[k1].matrix = original_transform.matrix
            # non_diff_params.update()
            # params.update(opt)
            
            # # 渲染旋转之前的图像
            # prb_integrator = mi.load_dict({'type': 'prb'})
            # if multi_pass_rendering:
            #     # 多次渲染模式 - 旋转前
            #     mini_pass_spp = spp//multi_pass_spp_divisor
            #     render_passes = mini_pass_spp
            #     mini_pass_renders_before = dr.empty(dr.cuda.ad.Float, render_passes * H * W * C)
            #     for i in range(render_passes):
            #         seed = np.random.randint(0,1000)+i
            #         img_i_before = mi.render(scene, params=params, spp=mini_pass_spp, sensor=camera_idx[cam_idx], seed=seed, integrator=prb_integrator)
            #         s_index = i * (H * W * C)
            #         e_index = (i+1) * (H * W * C)
            #         mini_pass_index = dr.arange(dr.cuda.ad.UInt, s_index, e_index)
            #         img_i_before = dr.ravel(img_i_before)
            #         dr.scatter(mini_pass_renders_before, img_i_before, mini_pass_index)
                
            #     @dr.wrap_ad(source='drjit', target='torch')
            #     def stack_imgs_before(imgs):
            #         imgs = imgs.reshape((render_passes, H, W, C))
            #         imgs = ch.mean(imgs,axis=0)
            #         return imgs
            #     mini_pass_renders_before = dr.cuda.ad.TensorXf(mini_pass_renders_before, dr.shape(mini_pass_renders_before))
            #     img_before = stack_imgs_before(mini_pass_renders_before)
            # else:
            #     # 普通渲染模式 - 旋转前
            #     img_before = mi.render(scene, params=params, spp=spp, sensor=camera_idx[cam_idx], seed=cam_idx+1, integrator=prb_integrator)
            
            # # 保存旋转之前的图像
            # img_before.set_label_(f"image_before_b{cam_idx:03d}_s{b:03d}")
            # rendered_img_before_path = os.path.join(render_path, f"before-render_b{cam_idx:03d}.png")
            # mi.util.write_bitmap(rendered_img_before_path, data=img_before, write_async=False)
            # print(f"保存旋转前图像: {rendered_img_before_path}")
           
            
            # #添加随机角度
            # pitch_perturbation = random.uniform(-3.0, 3.0)  # degrees
            # yaw_perturbation = random.uniform(-3.0, 3.0)    # degrees
        
            # if isinstance(camera_positions[cam_idx], mi.cuda_ad_rgb.Transform4f):
            #     non_diff_params[k1].matrix = camera_positions[cam_idx].matrix
            # else:
            #     non_diff_params[k1].matrix = mi.cuda_ad_rgb.Matrix4f(camera_positions[cam_idx])
            
            #渲染
            prb_integrator = mi.load_dict({'type': 'prb'})
            if multi_pass_rendering:
                    # achieve the affect of rendering at a high sample-per-pixel (spp) value 
                    # by rendering multiple times at a lower spp and averaging the results
                    # render_passes = 16 # TODO - make this a config param
                mini_pass_spp = spp//multi_pass_spp_divisor#计算每次渲染的采样率
                render_passes = mini_pass_spp# #计算渲染次数
                mini_pass_renders = dr.empty(dr.cuda.ad.Float, render_passes * H * W * C)
                for i in range(render_passes):
                    seed = np.random.randint(0,1000)+i
                    img_i =  mi.render(scene, params=params, spp=mini_pass_spp, sensor=camera_idx[cam_idx], seed=seed, integrator=prb_integrator)
                    s_index = i * (H * W * C)#当前的渲染结果的起始索引
                    e_index = (i+1) * (H * W * C)
                    mini_pass_index = dr.arange(dr.cuda.ad.UInt, s_index, e_index)
                    img_i = dr.ravel(img_i)#将渲染结果展平为一维张量
                    dr.scatter(mini_pass_renders, img_i, mini_pass_index)
                @dr.wrap_ad(source='drjit', target='torch')
                def stack_imgs(imgs):#用于将多次渲染的结果堆叠在一起
                    imgs = imgs.reshape((render_passes, H, W, C))
                    imgs = ch.mean(imgs,axis=0)#对重塑后的张量沿着第一个维度进行平均
                    return imgs
                mini_pass_renders = dr.cuda.ad.TensorXf(mini_pass_renders, dr.shape(mini_pass_renders))
                img = stack_imgs(mini_pass_renders)
            else: # dont use multi-pass rendering
                img =  mi.render(scene, params=params, spp=spp, sensor=camera_idx[cam_idx], seed=cam_idx+1, integrator=prb_integrator)
            # img.set_label_(f"image_b{cam_idx:03d}_s{b:03d}")
            
            dr.set_label(img, f"image_b{cam_idx:03d}_s{b:03d}")
            rendered_img_path = os.path.join(render_path,f"render_after{cam_idx:03d}.png")
            mi.util.write_bitmap(rendered_img_path, data=img, write_async=False)
            print(f"保存旋转后图像: {rendered_img_path}")
            #########################################################################
            # gt_boxes对渲染生成的图像进行目标检测
            rendered_img_input = dt2_input(rendered_img_path)
            model.train = False
            model.training = False
            model.proposal_generator.training = False
            model.roi_heads.training = False   
            outputs = model([rendered_img_input])#推理
            instances = outputs[0]['instances']
            #print("000000",instances)
            if targeted:
                mask = (instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST) & (instances.pred_classes == target_class)
            else: 
                mask = (instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST) & (instances.pred_classes == untargeted_class)
            filtered_instances = instances[mask]
            # 只有当检测到目标时才进行梯度计算和累积
            detection_successful = len(filtered_instances) > 0
            if detection_successful:
                print(f"Targeted detected at camera{cam_idx}, accumulating gradient....")
                successful_detection_count += 1
                total_successful_views += 1
                #9.16
                # 梯度计算前的显存状态
                #pre_grad_memory = torch.cuda.memory_allocated() / 1024**3
                #更新当前的gt_box
                current_gt_box = filtered_instances.pred_boxes.tensor[0].tolist()
                detected_boxes.append(current_gt_box)
                #########################################################################
                img = dr.ravel(img)#将渲染结果展平为一维张量
                # dr.disable_grad(img)
                start_index = b * (H * W * C)
                end_index = (b+1) * (H * W * C)
                index = dr.arange(dr.cuda.ad.UInt, start_index, end_index)                
                dr.scatter(imgs, img, index)
                time.sleep(1.0)
                # Get and Vizualize DT2 Predictions from rendered image
                rendered_img_input = dt2_input(rendered_img_path)#从渲染图像路径加载输入数据
                success = save_adv_image_preds(model \
                    , dt2_config, input=rendered_img_input \
                    , instance_mask_thresh=dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST \
                    , target = label
                    , untarget = unlabel
                    , is_targeted = targeted
                    , path=os.path.join(preds_path,f'render_after{cam_idx:03d}.png'))
                if targeted:
                    target = dr.cuda.ad.TensorXf([label], shape=(1,))
                else:
                    target = dr.cuda.ad.TensorXf([unlabel], shape=(1,))
                    
                # if success:
                #     successful_views += 1
                    #########################################################################
                    # # If 20 consecutive iterations have been made and the attack has not been successful, skip this camera position and increase its iteration count
                    # if iter_counts[cam_idx] >= 20 and not success:
                    #     logger.info(f"Skipping camera position {cam_idx} after 20 iterations without success.")
                    #     skipped_camera_indices.append(cam_idx)
                    #     cam_idx += 1  # Skip the current camera position
                    #     continue
                    
                    # # Update the iteration count for the camera position
                    # iter_counts[cam_idx] += 1
                    #########################################################################
                #计算损失和梯度
                imgs = dr.cuda.ad.TensorXf(dr.cuda.ad.Float(imgs),shape=(N, H, W, C))
                if (dr.grad_enabled(imgs)==False):
                    dr.enable_grad(imgs)
                loss = model_input(imgs, target, detected_boxes)
                logger.info(f"随机相机编号：{cam_idx}     loss: {str(loss.array[0])[0:7]}")
                #9.16
                # 梯度计算后的显存变化
                # post_grad_memory = torch.cuda.memory_allocated() / 1024**3
                # grad_memory_diff = post_grad_memory - pre_grad_memory
                # logger.info(f"随机相机编号：{cam_idx} loss: {str(loss.array[0])[0:7]} "
                #         f"显存变化: +{grad_memory_diff:.3f}GB")
                # sensor_loss = f"[PASS {cfg.sysconfig.pass_idx}] sensor pos: {cam_idx+1}/{len(camera_positions)}, loss: {str(loss.array[0])[0:7]}"
                # logger.info(sensor_loss)
                gc.collect()
                ch.cuda.empty_cache()
                dr.enable_grad(loss)
                dr.backward(loss)
                #########################################################################
                # L-INFattack
                # grad = dr.grad(opt[k])
                # tex = opt[k]
                # eta = alpha * dr.sign(grad)
                # if targeted:
                #     eta = -eta
                # tex = tex + eta
                # eta = dr.clamp(tex - orig_tex, -epsilon, epsilon)
                # tex = orig_tex + eta
                # tex = dr.clamp(tex, 0, 1)
                #########################################################################
                #获取并累计梯度
                HH, WW  = dr.shape(dr.grad(opt[param_keys[0]]))[0], dr.shape(dr.grad(opt[param_keys[0]]))[1]
                grad = ch.Tensor(dr.grad(opt[param_keys[0]]).array).view((HH, WW, C))
                random_accumulate_grad.append(grad)#收集梯度
                #9.16清理显存
                del imgs, loss, grad, outputs, instances
                gc.collect()
                if ch.cuda.is_available():
                    ch.cuda.empty_cache()
                logger.debug(f"随机相机 {cam_idx} 的显存已清理")
                #9.16
                # 显示梯度累积情况
                # total_grad_memory = sum(g.element_size() * g.numel() for g in random_accumulate_grad) / (1024**2)
                # logger.info(f"已累积{len(random_accumulate_grad)}/{target_gradient_count}个随机梯度，"
                #         f"总显存: {total_grad_memory:.2f}MB")
                #sensor_loss = f"[len(accumulate_grad): {len(accumulate_grad)}, current ASR: {1-len(accumulate_grad)/(cam_idx+1)}, loss: {str(loss.array[0])[0:7]}]"
                logger.info(f"已累积{len(random_accumulate_grad)}/{target_gradient_count}个随机梯度")
                #累积到足够的梯度就对纹理进行更新
                if len(random_accumulate_grad) >= target_gradient_count:
                    update_count += 1
                    logger.info(f"\n=== 双系统更新 #{update_count} ===")
                    #9.16
                    # print_gpu_usage("双系统更新前")
                    # print_variable_sizes(locals(), min_size_mb=10)  # 显示10MB以上的变量
                    # #第一步 生成轨迹轨迹
                    # logger.info(" 第一步 生成轨迹轨迹....")
                    # trajectory, trajectory_camera_indices = trajectory_generator.generate_trajectory_for_dt2_integration_with_actual_cameras(
                    #     target_points=10,verbose= True
                    # )
                    # # 在这里添加可视化调用
                    # if trajectory_camera_indices:
                    #     vis_save_path = os.path.join(tmp_perturbation_path, f"trajectory_update_{update_count}.png")
                    #     trajectory_generator.visualize_trajectory(trajectory_camera_indices, vis_save_path)
                    # #8.27添加如下来确认传的轨迹相机下标是正确的
                    # # trajectory_gradient = None#8.27
                    # # if trajectory_camera_indices:
                    # #     trajectory_gradient = compute_trajectory_gradient(
                    # #         trajectory_camera_indices, params, orig_texs, camera_positions, camera_idx, target_class, untargeted_class, targeted, label, unlabel, diff_params, non_diff_params, opt
                    # #     )
                    # if trajectory and trajectory_camera_indices:
                    #     logger.info(f" 生成轨迹成功，包含{len(trajectory_camera_indices)}个相机")
                    # else:
                    #     logger.warning(" 轨迹生成失败，跳过轨迹更新")
                    #     trajectory_camera_indices = None

                    # # #第二步 计算轨迹梯度
                    trajectory_gradient = None
                    # if trajectory_camera_indices:
                    #     logger.info("步骤二：计算轨迹梯度....")
                    #     trajectory_gradient = compute_trajectory_gradient(
                    #         trajectory_camera_indices, params, orig_texs, camera_positions, camera_idx, target_class, untargeted_class, targeted, label, unlabel, diff_params, non_diff_params, opt, center_pos
                    #     )
                    # else:
                    #     logger.info("步骤二: 无轨迹，跳过轨迹梯度计算")

                    #第三步 融合梯度并更新纹理
                    logger.info("步骤三：融合梯度并更新纹理....")
                    fusion_success = fuse_and_update_texture(img,
                        random_accumulate_grad, trajectory_gradient, params, orig_texs, update_count
                    )

                    if fusion_success:
                        logger.info(" 双系统纹理更新成功")
                    else:
                        logger.error(" 双系统纹理更新失败，跳过本次更新")
                    
                    #第四步  重置随机系统梯度累积
                    random_accumulate_grad = []
                    successful_detection_count = 0

                    current_asr = 1-(total_successful_views / processed_cameras)
                    logger.info("="*80)
                    logger.info(f"双系统更新完成 - 总体ASR: {current_asr:.3f} ")
                    logger.info("="*80)
                    
                    gc.collect()
                    ch.cuda.empty_cache()

            else:
                logger.info(f" 随机相机{cam_idx}检测失败，跳过")
        
        #处理完所有相机后，检查是否有剩余的梯度需要更新
        if len(random_accumulate_grad) > 0:
            update_count += 1
            logger.info("="*80)
            logger.info(f"最终双系统更新 #{update_count}")
            logger.info(f"处理剩余{len(random_accumulate_grad)}个随机梯度")
            logger.info("="*80)

        
            # # 生成最终轨迹
            # try:
            #     trajectory, trajectory_camera_indices = trajectory_generator.generate_trajectory_for_dt2_integration(
            #         target_points=8, verbose=True
            #     )
            #     if trajectory and trajectory_camera_indices:
            #         trajectory_gradient = compute_trajectory_gradient(
            #             trajectory_camera_indices, params, orig_texs,
            #             camera_positions, camera_idx, target_class, untargeted_class, targeted, label, unlabel, diff_params, non_diff_params, opt,center_pos
            #         )
            #     else:
            #         trajectory_gradient = None
            # except:
            #     trajectory_gradient = None
            
            #最终融合更新
            trajectory_gradient = None
            fusion_success = fuse_and_update_texture(img,
                random_accumulate_grad, trajectory_gradient, params, orig_texs, update_count
            )
            if fusion_success:
                logger.info(" 最终双系统更新成功")
            else:
                logger.error(" 最终双系统更新失败")

            #9.16清理代码
            # 【在此处添加清理代码】
            # 清理在最终更新步骤中创建的临时梯度变量
            if 'trajectory_gradient' in locals() and trajectory_gradient is not None:
                del trajectory_gradient
            # random_accumulate_grad 自身也会被函数结束回收，但显式清空更安全
            random_accumulate_grad.clear() 
            gc.collect()
            if ch.cuda.is_available():
                ch.cuda.empty_cache()
            logger.debug("最终更新步骤的临时梯度已清理")
        #最终统计
        final_asr = 1-(total_successful_views / total_cameras)
        logger.info("="*80)
        logger.info(" 双梯度系统攻击完成")
        logger.info(f"总相机数: {total_cameras}")
        logger.info(f"成功检测数: {total_successful_views}")
        logger.info(f"最终ASR: {final_asr:.3f}")
        logger.info(f"总更新次数: {update_count}")
        logger.info("="*80)
        
        return scene
                    
        #             #计算平均梯度
        #             average_grad = sum(accumulate_grad) / len(accumulate_grad)
        #         #3d纹理的对抗攻击的算法
        #             for i, k in enumerate(param_keys):
        #                 C = 3 
        #                     #grad = ch.Tensor(dr.grad(opt[k]).array).view((HH, WW, C)# when use opacity
        #             #         grad=ch.Tensor(dr.grad(opt[k])[cam_idx].array).view((HH, WW, C))
        #             #         accumulate_grad += grad
        #             ##求平均
                    
                    
        #                 tex = ch.Tensor(opt[k].array).view((HH, WW, C))
        #                 _orig_tex = ch.Tensor(orig_texs[i].array).view((HH, WW, C))
        #                 l = len(grad.shape) -  1
        #                 g_norm = ch.norm(average_grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
        #                 scaled_grad = average_grad / (g_norm  + 1e-10)
        #             ## accumulate_grad += grad
        #                 if targeted:
        #                     scaled_grad = -scaled_grad # 有目标攻击，取最小化损失的方向
        #                 # step
        #                 tex = tex + scaled_grad * alpha
        #                 delta  = tex - _orig_tex
        #                 # project
        #                 delta =  delta.renorm(p=2, dim=0, maxnorm=epsilon)
        #                 tex = _orig_tex + delta

        #                 # convert back to mitsuba dtypes            
        #                 tex = dr.cuda.ad.TensorXf(tex.to(DEVICE))
        #                 # divide by average brightness
        #                 scaled_img = img / dr.mean(dr.detach(img))
        #                 tex = tex / dr.mean(scaled_img)         
        #                 tex = dr.clamp(tex, 0, 1)
        #                 params[k] = tex     
        #                 dr.enable_grad(params[k])
        #                 params.update()
        #                 perturbed_tex = mi.Bitmap(params[k])
                    
            
        #                 mi.util.write_bitmap(os.path.join(tmp_perturbation_path,f"{k}_{0}.png"), data=perturbed_tex, write_async=False)
                
        #             #重新梯度累积
        #             accumulate_grad = []
        #             successful_detection_count = 0
                    
        #             current_asr = total_successful_views /processed_cameras
        #             logger.info(f"Applied texture update. Overall ACC: {current_asr:.3f} ({total_successful_views}/{processed_cameras})")
        #             # time.sleep(0.2)
        #             # if isinstance(params[k], dr.cuda.ad.TensorXf):
        #             #     perturbed_tex = mi.Bitmap(params[k])
        #             #     mi.util.write_bitmap("perturbed_tex_map.png", data=perturbed_tex, write_async=False)
        #             #     logger.info(f"Skipped camera positions: {skipped_camera_indices}")
        #             #     #time.sleep(0.2) 
                    
        #             gc.collect()
        #             ch.cuda.empty_cache()
        #     else:
        #         # print(f"No target detected at camera {cam_idx}, skipping gradient accumulation")
        #         logger.info(f"Camera {cam_idx}: No detection, skipping")
        # #剩余不足五个成功检测
        # if len(accumulate_grad) > 0:
        #     update_count += 1
        #     print(f"\n=== Final Texture Update #{update_count} ===")
        #     print(f"Averaging gradients from remaining {len(accumulate_grad)} successful detections")
        #     average_grad = sum(accumulate_grad) /len(accumulate_grad)

        #     #3d纹理的对抗攻击的算法
        #     for i, k in enumerate(param_keys):
        #         C = 3 
        #                     #grad = ch.Tensor(dr.grad(opt[k]).array).view((HH, WW, C)# when use opacity
        #             #         grad=ch.Tensor(dr.grad(opt[k])[cam_idx].array).view((HH, WW, C))
        #             #         accumulate_grad += grad
        #             #     #求平均
                    
                    
        #         tex = ch.Tensor(opt[k].array).view((HH, WW, C))
        #         _orig_tex = ch.Tensor(orig_texs[i].array).view((HH, WW, C))
        #         l = len(grad.shape) -  1
        #         g_norm = ch.norm(average_grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
        #         scaled_grad = average_grad / (g_norm  + 1e-10)
        #     ## accumulate_grad += grad
        #         if targeted:
        #             scaled_grad = -scaled_grad # 有目标攻击，取最小化损失的方向
        #             # step
        #         tex = tex + scaled_grad * alpha
        #         delta  = tex - _orig_tex
        #         # project
        #         delta =  delta.renorm(p=2, dim=0, maxnorm=epsilon)
        #         tex = _orig_tex + delta

        #         # convert back to mitsuba dtypes            
        #         tex = dr.cuda.ad.TensorXf(tex.to(DEVICE))
        #         # divide by average brightness
        #         scaled_img = img / dr.mean(dr.detach(img))
        #         tex = tex / dr.mean(scaled_img)         
        #         tex = dr.clamp(tex, 0, 1)
        #         params[k] = tex     
        #         dr.enable_grad(params[k])
        #         params.update()
        #         perturbed_tex = mi.Bitmap(params[k])
        
        #         mi.util.write_bitmap(os.path.join(tmp_perturbation_path,f"{k}_{0}.png"), data=perturbed_tex, write_async=False)
        # final_asr = total_successful_views / total_cameras
        # print(f"\n=== Attack Summary ===")
        # print(f"Total cameras processed: {total_cameras}")
        # print(f"Successful detections: {total_successful_views}")
        # print(f"Final ASR: {final_asr:.3f}")
        # print(f"Total texture updates: {update_count}")    
        # return scene
    
    samples_per_pixel = spp
    epsilon = eps
    alpha = eps_step #(epsilon / (iters/50))
    label = target_class
    unlabel = untargeted_class

    # --- 新增：一次性初始化 ---9.22
    logger.info("执行一次性参数和优化器初始化...")
    params = mi.traverse(scene)
    # print("Available scene parameters:")    
    # print(params)
    # exit()
    
    #9.24
    orig_texs = []
    for key in param_keys:
        # #9.24在访问特定键之前，打印所有可用的参数
        # print("Available scene parameters:")
        # print(params)
        # if 'mat-adversarial' in key:
        #     print(key, type(params[key]))#9.24

        if isinstance(params[key], dr.cuda.ad.TensorXf):
            orig_tex = dr.cuda.ad.TensorXf(params[key])
        elif isinstance(params[key], dr.cuda.ad.Float):
            orig_tex = dr.cuda.ad.Float(params[key])
        else:
            raise Exception("Unrecognized Differentiable Parameter Data Type.")
        # orig_tex.set_label_(f"{key}_orig_tex")
        dr.set_label(orig_tex, f"{key}_orig_tex")
        orig_texs.append(orig_tex)
        dr.enable_grad(orig_texs[-1])

    diff_params = mi.traverse(scene)
    diff_params.keep([key for key in param_keys])

    # 新增 non_diff_params 的初始化
    non_diff_params = mi.traverse(scene)
    non_diff_params.keep([k1, k2]) # k1 和 k2 在 attack_dt2 中已经定义

    opt = mi.ad.Adam(lr=0.1, params=diff_params)
    for key in param_keys:
        dr.enable_grad(opt[key])
        # opt[key].set_label_(f"{key}_bitmap")
        
        dr.set_label(opt[key], f"{key}_bitmap")
    logger.info("初始化完成。")
    # --- 初始化结束 ---
    # # --- 新增：一次性参数和优化器初始化 ---9.24 修正
    # logger.info("执行一次性参数和优化器初始化...")
    # params = mi.traverse(scene)

    # # 调试：打印与 patch 相关的键
    # patch_related = [k for k in params.keys() if 'elm__2' in k or 'mat-adversarial' in k]
    # print("\n[调试] 与补丁相关的可用参数键：")
    # for kk in patch_related:
    #     print("  ", kk)
    # if not patch_related:
    #     print("  (没有找到包含 elm__2 或 mat-adversarial 的键，请确认场景或 variant)")

    # # 安全过滤掉不存在的键，避免 KeyError
    # filtered_param_keys = []
    # for key in param_keys:
    #     if key not in params:
    #         logger.warning(f"目标参数键不存在，已跳过: {key}")
    #     else:
    #         filtered_param_keys.append(key)

    # if not filtered_param_keys:
    #     raise RuntimeError("param_keys 中没有任何合法参数，请修正配置 scene.target_param_keys")

    # orig_texs = []
    # for key in filtered_param_keys:
    #     # 这里用 key（不是 k）
    #     if 'mat-adversarial' in key:
    #         print(f"[调试] 材质相关键: {key} -> {type(params[key])}")

    #     # 仅在这里才真正访问 params[key]
    #     v = params[key]
    #     if isinstance(v, dr.cuda.ad.TensorXf):
    #         orig_tex = dr.cuda.ad.TensorXf(v)
    #     elif isinstance(v, dr.cuda.ad.Float):
    #         orig_tex = dr.cuda.ad.Float(v)
    #     else:
    #         raise Exception(f"不支持的可微类型: {key} -> {type(v)}")
    #     orig_tex.set_label_(f"{key}_orig_tex")
    #     orig_texs.append(orig_tex)
    #     dr.enable_grad(orig_texs[-1])

    # # 用过滤后的键
    # diff_params = mi.traverse(scene)
    # diff_params.keep(filtered_param_keys)

    # non_diff_params = mi.traverse(scene)
    # non_diff_params.keep([k1, k2])

    # opt = mi.ad.Adam(lr=0.1, params=diff_params)
    # for key in filtered_param_keys:
    #     dr.enable_grad(opt[key])
    #     opt[key].set_label_(f"{key}_bitmap")

    # # 后续循环也改用 filtered_param_keys
    # param_keys = filtered_param_keys
    # logger.info("初始化完成。")
    # # --- 初始化结束 ---
    # 定义总的执行轮数9.22
    total_epochs = 1
    for epoch in range(total_epochs):
        logger.info(f"\n{'='*30} 开始第 {epoch + 1}/{total_epochs} 轮完整优化 {'='*30}\n")
        
        # 调用优化函数，并将返回的更新后场景作为下一次循环的输入
        scene = optim_batch(scene=scene,
                            params=params,
                            opt=opt,
                            orig_texs=orig_texs,
                            non_diff_params=non_diff_params, # 新增传入 non_diff_params
                            batch_size=batch_size,
                            camera_positions=moves_matrices,
                            spp=samples_per_pixel,
                            k=k, label=label,
                            unlabel=unlabel,
                            iters=iters,
                            alpha=alpha,
                            epsilon=epsilon,
                            targeted=targeted)
        
        logger.info(f"\n{'='*30} 第 {epoch + 1}/{total_epochs} 轮优化完成 {'='*30}\n")
    # 循环结束后，删除最终的场景和模型
    del scene,model,moves_matrices,checkpointer, params, opt, orig_texs, non_diff_params
    gc.collect()
    ch.cuda.empty_cache()