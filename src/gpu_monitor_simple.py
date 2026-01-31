import torch
import gc
import sys

def quick_gpu_check(tag=""):
    """快速检查GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🔍 [{tag}] GPU显存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    else:
        print(f"[{tag}] CUDA不可用")

def check_variable(var, name):
    """检查单个变量的显存占用"""
    if isinstance(var, torch.Tensor):
        if var.is_cuda:
            size_gb = var.element_size() * var.numel() / 1024**3
            print(f"📦 {name}: {size_gb:.3f}GB, 形状: {var.shape}, 设备: {var.device}")
        else:
            print(f"📦 {name}: CPU张量, 形状: {var.shape}")
    else:
        print(f"📦 {name}: 不是PyTorch张量")

def check_multiple_variables(**kwargs):
    """一次检查多个变量"""
    print("=" * 50)
    print("📊 变量显存检查:")
    total_gpu_memory = 0
    
    for name, var in kwargs.items():
        if isinstance(var, torch.Tensor) and var.is_cuda:
            size_gb = var.element_size() * var.numel() / 1024**3
            total_gpu_memory += size_gb
            print(f"  {name}: {size_gb:.3f}GB")
        elif isinstance(var, list) and var and isinstance(var[0], torch.Tensor):
            # 检查张量列表
            list_memory = 0
            gpu_count = 0
            for tensor in var:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    list_memory += tensor.element_size() * tensor.numel() / 1024**3
                    gpu_count += 1
            if list_memory > 0:
                total_gpu_memory += list_memory
                print(f"  {name}: {list_memory:.3f}GB (包含{gpu_count}个GPU张量)")
    
    print(f"  总计: {total_gpu_memory:.3f}GB")
    print("=" * 50)

def memory_checkpoint(checkpoint_name, show_details=False):
    """内存检查点，显示当前状态"""
    print(f"\n🚩 检查点: {checkpoint_name}")
    print("-" * 30)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"当前分配: {allocated:.3f}GB")
        print(f"当前保留: {reserved:.3f}GB")
        print(f"峰值分配: {max_allocated:.3f}GB")
        
        # 如果显存使用过高，发出警告
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / 1024**3
        usage_percent = (allocated / total_memory) * 100
        
        if usage_percent > 80:
            print(f"⚠️ 警告: GPU显存使用率 {usage_percent:.1f}% - 建议清理内存!")
        elif usage_percent > 60:
            print(f"ℹ️ 注意: GPU显存使用率 {usage_percent:.1f}%")
        else:
            print(f"✅ 显存使用正常: {usage_percent:.1f}%")
    
    if show_details:
        # 显示当前Python进程的内存使用
        import psutil
        import os
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024**3
        print(f"进程RAM使用: {ram_usage:.2f}GB")
    
    print("-" * 30)

def clean_and_check(tag=""):
    """清理内存并检查效果"""
    print(f"🧹 清理内存 {tag}...")
    before_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    after_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    freed = before_allocated - after_allocated
    
    if freed > 0:
        print(f"✅ 释放了 {freed:.3f}GB 显存")
    else:
        print("ℹ️ 没有额外显存被释放")
    
    quick_gpu_check(f"清理后{tag}")