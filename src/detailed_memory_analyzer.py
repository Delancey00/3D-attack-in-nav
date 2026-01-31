import torch
import gc
import drjit as dr
import mitsuba as mi
import numpy as np
from collections import defaultdict
import sys
import traceback

class DetailedMemoryAnalyzer:
    """详细的参数级显存分析器"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.variable_registry = {}
        self.memory_snapshots = []
        self.peak_memory_usage = {}
        
    def register_variable(self, var, name, category="general"):
        """注册需要监控的变量"""
        try:
            memory_info = self._analyze_variable_memory(var, name)
            self.variable_registry[name] = {
                'variable': var,
                'category': category,
                'memory_info': memory_info,
                'registered_at': len(self.memory_snapshots)
            }
            return memory_info
        except Exception as e:
            print(f"注册变量 {name} 失败: {str(e)}")
            return None
    
    def _analyze_variable_memory(self, var, name):
        """分析单个变量的详细显存信息"""
        info = {
            'name': name,
            'type': type(var).__name__,
            'size_mb': 0,
            'size_gb': 0,
            'shape': 'N/A',
            'dtype': 'N/A',
            'location': 'N/A',
            'element_count': 0
        }
        
        try:
            # PyTorch Tensors
            if isinstance(var, torch.Tensor):
                info['size_mb'] = var.element_size() * var.nelement() / 1024**2
                info['size_gb'] = info['size_mb'] / 1024
                info['shape'] = str(var.shape)
                info['dtype'] = str(var.dtype)
                info['location'] = str(var.device)
                info['element_count'] = var.nelement()
                
            # DrJit tensors/arrays
            elif hasattr(var, 'array') and hasattr(var, 'shape'):
                if hasattr(var, 'dtype'):
                    dtype_size = 4 if 'float32' in str(var.dtype) else 8  # 估算
                else:
                    dtype_size = 4  # 默认float32
                    
                if hasattr(var.shape, '__len__'):
                    total_elements = np.prod(var.shape)
                else:
                    total_elements = var.shape if isinstance(var.shape, int) else 1
                    
                info['size_mb'] = total_elements * dtype_size / 1024**2
                info['size_gb'] = info['size_mb'] / 1024
                info['shape'] = str(var.shape)
                info['dtype'] = str(getattr(var, 'dtype', 'unknown'))
                info['location'] = 'GPU' if 'cuda' in str(type(var)) else 'CPU'
                info['element_count'] = total_elements
                
            # NumPy arrays
            elif hasattr(var, 'nbytes'):
                info['size_mb'] = var.nbytes / 1024**2
                info['size_gb'] = info['size_mb'] / 1024
                info['shape'] = str(var.shape)
                info['dtype'] = str(var.dtype)
                info['location'] = 'CPU'
                info['element_count'] = var.size
                
            # Lists/tuples of tensors
            elif isinstance(var, (list, tuple)):
                total_size_mb = 0
                element_count = 0
                shapes = []
                dtypes = []
                
                for i, item in enumerate(var):
                    item_info = self._analyze_variable_memory(item, f"{name}[{i}]")
                    total_size_mb += item_info['size_mb']
                    element_count += item_info['element_count']
                    shapes.append(item_info['shape'])
                    dtypes.append(item_info['dtype'])
                
                info['size_mb'] = total_size_mb
                info['size_gb'] = total_size_mb / 1024
                info['shape'] = f"List[{len(var)}]: {shapes[:3]}..." if len(shapes) > 3 else f"List: {shapes}"
                info['dtype'] = f"Mixed: {list(set(dtypes))}" if len(set(dtypes)) > 1 else dtypes[0] if dtypes else 'N/A'
                info['element_count'] = element_count
                info['location'] = 'Mixed'
                
            # Dictionary of parameters
            elif hasattr(var, 'items'):  # dict-like
                total_size_mb = 0
                param_count = 0
                for key, value in var.items():
                    item_info = self._analyze_variable_memory(value, f"{name}.{key}")
                    total_size_mb += item_info['size_mb']
                    param_count += 1
                
                info['size_mb'] = total_size_mb
                info['size_gb'] = total_size_mb / 1024
                info['shape'] = f"Dict[{param_count} params]"
                info['dtype'] = 'Mixed'
                info['element_count'] = param_count
                
        except Exception as e:
            info['error'] = str(e)
            print(f"分析变量 {name} 时出错: {str(e)}")
        
        return info
    
    def analyze_model_parameters(self, model, model_name="model"):
        """详细分析模型各部分的显存占用"""
        print(f"\n{'='*80}")
        print(f"模型 '{model_name}' 详细显存分析")
        print(f"{'='*80}")
        
        model_analysis = {}
        total_params = 0
        total_size_mb = 0
        
        # 分析各个模块
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                module_params = sum(p.numel() for p in module.parameters())
                module_size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**2
                
                if module_params > 0:
                    model_analysis[name] = {
                        'params': module_params,
                        'size_mb': module_size_mb,
                        'type': type(module).__name__
                    }
                    total_params += module_params
                    total_size_mb += module_size_mb
        
        # 按显存占用排序
        sorted_modules = sorted(model_analysis.items(), key=lambda x: x[1]['size_mb'], reverse=True)
        
        print(f"模型总参数: {total_params:,} ({total_size_mb:.1f} MB)")
        print(f"\n显存占用 TOP 10 模块:")
        print(f"{'模块名':<50} {'参数数量':<15} {'显存占用':<12} {'模块类型':<20}")
        print("-" * 100)
        
        for i, (module_name, info) in enumerate(sorted_modules[:10]):
            print(f"{module_name:<50} {info['params']:>14,} {info['size_mb']:>8.1f} MB   {info['type']:<20}")
        
        # 按模块类型汇总
        type_summary = defaultdict(lambda: {'params': 0, 'size_mb': 0, 'count': 0})
        for module_name, info in model_analysis.items():
            module_type = info['type']
            type_summary[module_type]['params'] += info['params']
            type_summary[module_type]['size_mb'] += info['size_mb']
            type_summary[module_type]['count'] += 1
        
        print(f"\n按模块类型汇总:")
        print(f"{'模块类型':<25} {'数量':<8} {'总参数':<15} {'总显存':<12}")
        print("-" * 65)
        for module_type, summary in sorted(type_summary.items(), key=lambda x: x[1]['size_mb'], reverse=True):
            print(f"{module_type:<25} {summary['count']:>6} {summary['params']:>14,} {summary['size_mb']:>8.1f} MB")
        
        return model_analysis
    
    def analyze_rendering_parameters(self, scene_params, param_keys):
        """分析渲染相关参数的显存占用"""
        print(f"\n{'='*80}")
        print(f"渲染参数显存分析")
        print(f"{'='*80}")
        
        rendering_analysis = {}
        total_size_mb = 0
        
        for key in param_keys:
            if key in scene_params:
                param = scene_params[key]
                info = self._analyze_variable_memory(param, key)
                rendering_analysis[key] = info
                total_size_mb += info['size_mb']
                
                print(f"参数: {key}")
                print(f"  类型: {info['type']}")
                print(f"  形状: {info['shape']}")
                print(f"  数据类型: {info['dtype']}")
                print(f"  显存占用: {info['size_mb']:.2f} MB ({info['size_gb']:.3f} GB)")
                print(f"  元素数量: {info['element_count']:,}")
                print()
        
        print(f"渲染参数总显存占用: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        return rendering_analysis
    
    def analyze_gradient_accumulation(self, gradient_list, list_name="gradients"):
        """分析梯度累积的显存占用"""
        print(f"\n{'='*80}")
        print(f"梯度累积 '{list_name}' 显存分析")
        print(f"{'='*80}")
        
        if not gradient_list:
            print("梯度列表为空")
            return {}
        
        total_size_mb = 0
        gradient_analysis = {}
        
        for i, grad in enumerate(gradient_list):
            grad_info = self._analyze_variable_memory(grad, f"{list_name}[{i}]")
            gradient_analysis[i] = grad_info
            total_size_mb += grad_info['size_mb']
        
        avg_size_mb = total_size_mb / len(gradient_list) if gradient_list else 0
        
        print(f"梯度数量: {len(gradient_list)}")
        print(f"单个梯度平均大小: {avg_size_mb:.2f} MB")
        print(f"总显存占用: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        
        if gradient_list:
            sample_grad = gradient_list[0]
            sample_info = self._analyze_variable_memory(sample_grad, "sample_gradient")
            print(f"梯度形状: {sample_info['shape']}")
            print(f"梯度数据类型: {sample_info['dtype']}")
        
        return gradient_analysis
    
    def analyze_camera_data(self, camera_positions, camera_name="camera_positions"):
        """分析相机相关数据的显存占用"""
        print(f"\n{'='*80}")
        print(f"相机数据 '{camera_name}' 显存分析")
        print(f"{'='*80}")
        
        camera_info = self._analyze_variable_memory(camera_positions, camera_name)
        
        print(f"相机数据类型: {camera_info['type']}")
        print(f"数据形状: {camera_info['shape']}")
        print(f"显存占用: {camera_info['size_mb']:.2f} MB ({camera_info['size_gb']:.3f} GB)")
        print(f"元素数量: {camera_info['element_count']:,}")
        
        # 如果是相机位置矩阵，计算相机数量
        if hasattr(camera_positions, 'shape') and len(camera_positions.shape) >= 2:
            if camera_positions.shape[-1] == 4 and camera_positions.shape[-2] == 4:
                num_cameras = camera_positions.shape[0] if len(camera_positions.shape) == 3 else 1
                print(f"相机数量: {num_cameras}")
                print(f"每个相机矩阵大小: {camera_info['size_mb']/num_cameras:.3f} MB")
        
        return camera_info
    
    def create_memory_snapshot(self, tag="snapshot"):
        """创建当前显存状态快照"""
        gpu_info = self._get_gpu_memory_info()
        
        snapshot = {
            'tag': tag,
            'snapshot_id': len(self.memory_snapshots),
            'gpu_memory': gpu_info,
            'variables': {}
        }
        
        # 分析所有注册的变量
        for var_name, var_data in self.variable_registry.items():
            try:
                current_info = self._analyze_variable_memory(var_data['variable'], var_name)
                snapshot['variables'][var_name] = current_info
            except Exception as e:
                snapshot['variables'][var_name] = {'error': str(e)}
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def _get_gpu_memory_info(self):
        """获取GPU显存信息"""
        if not torch.cuda.is_available():
            return None
            
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'total': total_memory,
            'free': total_memory - reserved,
            'usage_percent': (reserved / total_memory) * 100
        }
    
    def compare_snapshots(self, snapshot1_id=None, snapshot2_id=None):
        """比较两个显存快照，找出差异"""
        if len(self.memory_snapshots) < 2:
            print("需要至少2个快照才能进行比较")
            return
        
        snap1 = self.memory_snapshots[snapshot1_id or -2]
        snap2 = self.memory_snapshots[snapshot2_id or -1]
        
        print(f"\n{'='*80}")
        print(f"显存快照比较: '{snap1['tag']}' vs '{snap2['tag']}'")
        print(f"{'='*80}")
        
        # GPU显存变化
        if snap1['gpu_memory'] and snap2['gpu_memory']:
            gpu1, gpu2 = snap1['gpu_memory'], snap2['gpu_memory']
            allocated_diff = gpu2['allocated'] - gpu1['allocated']
            reserved_diff = gpu2['reserved'] - gpu1['reserved']
            
            print(f"GPU显存变化:")
            print(f"  已分配: {gpu1['allocated']:.2f}GB -> {gpu2['allocated']:.2f}GB (变化: {allocated_diff:+.2f}GB)")
            print(f"  已预留: {gpu1['reserved']:.2f}GB -> {gpu2['reserved']:.2f}GB (变化: {reserved_diff:+.2f}GB)")
        
        # 变量变化
        print(f"\n变量显存变化:")
        print(f"{'变量名':<30} {'原大小(MB)':<12} {'新大小(MB)':<12} {'变化(MB)':<12}")
        print("-" * 70)
        
        all_vars = set(snap1['variables'].keys()) | set(snap2['variables'].keys())
        significant_changes = []
        
        for var_name in all_vars:
            size1 = snap1['variables'].get(var_name, {}).get('size_mb', 0)
            size2 = snap2['variables'].get(var_name, {}).get('size_mb', 0)
            diff = size2 - size1
            
            if abs(diff) > 1:  # 只显示变化超过1MB的
                significant_changes.append((var_name, size1, size2, diff))
        
        # 按变化大小排序
        significant_changes.sort(key=lambda x: abs(x[3]), reverse=True)
        
        for var_name, size1, size2, diff in significant_changes[:10]:
            print(f"{var_name:<30} {size1:>10.1f} {size2:>10.1f} {diff:>+10.1f}")
        
        return snap1, snap2
    
    def find_memory_hogs(self, threshold_mb=100):
        """找出显存占用超过阈值的变量"""
        print(f"\n{'='*80}")
        print(f"显存占用大户分析 (阈值: {threshold_mb}MB)")
        print(f"{'='*80}")
        
        memory_hogs = []
        
        for var_name, var_data in self.variable_registry.items():
            memory_info = var_data['memory_info']
            if memory_info['size_mb'] > threshold_mb:
                memory_hogs.append((var_name, memory_info, var_data['category']))
        
        # 按大小排序
        memory_hogs.sort(key=lambda x: x[1]['size_mb'], reverse=True)
        
        if not memory_hogs:
            print(f"没有发现显存占用超过 {threshold_mb}MB 的变量")
            return []
        
        print(f"{'变量名':<35} {'类别':<15} {'大小(MB)':<12} {'形状':<20} {'类型':<15}")
        print("-" * 100)
        
        for var_name, info, category in memory_hogs:
            print(f"{var_name:<35} {category:<15} {info['size_mb']:>8.1f} {str(info['shape']):<20} {info['type']:<15}")
        
        return memory_hogs
    
    def generate_memory_report(self, save_to_file=True):
        """生成完整的显存分析报告"""
        report = {
            'summary': {
                'total_variables': len(self.variable_registry),
                'total_snapshots': len(self.memory_snapshots),
                'current_gpu_memory': self._get_gpu_memory_info()
            },
            'variables': {},
            'snapshots': self.memory_snapshots
        }
        
        # 变量分析
        total_variable_memory = 0
        categories = defaultdict(lambda: {'count': 0, 'total_mb': 0})
        
        for var_name, var_data in self.variable_registry.items():
            info = var_data['memory_info']
            category = var_data['category']
            
            report['variables'][var_name] = info
            total_variable_memory += info['size_mb']
            
            categories[category]['count'] += 1
            categories[category]['total_mb'] += info['size_mb']
        
        report['summary']['total_variable_memory_mb'] = total_variable_memory
        report['summary']['categories'] = dict(categories)
        
        if save_to_file:
            import json
            with open('detailed_memory_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print("详细显存报告已保存到: detailed_memory_report.json")
        
        return report

# 创建全局分析器实例
memory_analyzer = DetailedMemoryAnalyzer()

# 便捷函数
def analyze_model(model, name="model"):
    """分析模型显存占用"""
    return memory_analyzer.analyze_model_parameters(model, name)

def analyze_variable(var, name, category="general"):
    """分析单个变量显存占用"""
    info = memory_analyzer.register_variable(var, name, category)
    if info:
        print(f"\n变量 '{name}' 显存分析:")
        print(f"  类型: {info['type']}")
        print(f"  形状: {info['shape']}")
        print(f"  大小: {info['size_mb']:.2f} MB ({info['size_gb']:.3f} GB)")
        print(f"  数据类型: {info['dtype']}")
        print(f"  位置: {info['location']}")
    return info

def analyze_gradients(gradient_list, name="gradients"):
    """分析梯度列表显存占用"""
    return memory_analyzer.analyze_gradient_accumulation(gradient_list, name)

def analyze_rendering_params(scene_params, param_keys):
    """分析渲染参数显存占用"""
    return memory_analyzer.analyze_rendering_parameters(scene_params, param_keys)

def analyze_cameras(camera_positions, name="cameras"):
    """分析相机数据显存占用"""
    return memory_analyzer.analyze_camera_data(camera_positions, name)

def take_memory_snapshot(tag):
    """创建显存快照"""
    return memory_analyzer.create_memory_snapshot(tag)

def compare_memory_snapshots(snap1_id=None, snap2_id=None):
    """比较显存快照"""
    return memory_analyzer.compare_snapshots(snap1_id, snap2_id)

def find_memory_hogs(threshold_mb=100):
    """找出显存占用大户"""
    return memory_analyzer.find_memory_hogs(threshold_mb)

def generate_full_report():
    """生成完整报告"""
    return memory_analyzer.generate_memory_report()