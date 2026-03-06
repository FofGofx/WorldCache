# WorldScore/worldscore/calculate_time_stats.py

import fire
import os
from pathlib import Path
from omegaconf import OmegaConf

from worldscore.common.utils import print_banner


def parse_time_file(time_file_path: Path) -> dict:
    """
    解析 time.txt 文件
    
    Args:
        time_file_path: time.txt 文件路径
        
    Returns:
        dict: 包含 generate_time 和 denoise_time 的字典，如果解析失败返回 None
    """
    try:
        with open(time_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        result = {}
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'generate_time':
                    result['generate_time'] = float(value)
                elif key == 'denoise_time':
                    result['denoise_time'] = float(value)
        
        # 检查是否成功解析了至少一个时间值
        if 'generate_time' in result or 'denoise_time' in result:
            return result
        else:
            return None
    except Exception as e:
        return None


def calculate_time_stats(model_name: str, acceleration_mode: str = None) -> None:
    """
    统计 voyager 模型生成视频的平均时间
    
    Args:
        model_name: 模型名称（如 'voyager'）
        acceleration_mode: 加速模式。Options: 'original', 'ToCa', 'DuCa', 'Taylor'.
            如果提供，将使用对应加速模式的输出目录（带后缀）。
            如果为 None，使用默认输出目录。
    """
    # 加载配置文件
    base_config = OmegaConf.load(os.path.join('config/base_config.yaml'))
    try:
        config = OmegaConf.load(os.path.join('config/model_configs', f"{model_name}.yaml"))
    except FileNotFoundError:
        print(f"-- Model config file not found for {model_name}")
        return
    
    config = OmegaConf.merge(base_config, config)
    
    # Modify output_dir to include acceleration mode suffix if provided
    if acceleration_mode is not None:
        original_output_dir = config.get('output_dir', 'worldscore_output')
        acc_mode_lower = acceleration_mode.lower()
        config['output_dir'] = f"{original_output_dir}_{acc_mode_lower}"
        print(f"-- Using output directory with acceleration mode suffix: {config['output_dir']}")
    
    # 解析环境变量
    config = OmegaConf.to_container(config, resolve=True)
    
    # 构建 static 目录路径
    root_path = Path(
        f"{config['runs_root']}/{config['output_dir']}/static"
    )
    
    if not root_path.exists():
        print(f"-- Static directory not found: {root_path}")
        return
    
    # 统计变量
    total_generate_time = 0.0
    total_denoise_time = 0.0
    total_video_segments = 0  # 总视频段数
    total_instances = 0  # 总实例数
    three_segment_instances = 0  # 三段视频的实例数
    single_segment_instances = 0  # 单段视频的实例数
    warnings = []  # 警告列表
    
    # 遍历 static 目录结构
    visual_styles = sorted([
        x.name for x in root_path.iterdir() if x.is_dir()
    ])
    
    for visual_style in visual_styles:
        visual_style_dir = root_path / visual_style
        scene_types = sorted([
            x.name for x in visual_style_dir.iterdir() if x.is_dir()
        ])
        
        for scene_type in scene_types:
            scene_type_dir = visual_style_dir / scene_type
            
            category_list = sorted([
                f.name for f in scene_type_dir.iterdir() if f.is_dir()
            ])
            
            for category in category_list:
                category_dir = scene_type_dir / category
                instance_list = sorted([
                    f.name for f in category_dir.iterdir() if f.is_dir()
                ])
                
                for instance in instance_list:
                    instance_dir = category_dir / instance
                    time_file = instance_dir / "time.txt"
                    
                    # 检查 time.txt 是否存在
                    if not time_file.exists():
                        warnings.append(f"Missing time.txt: {instance_dir}")
                        continue
                    
                    # 解析 time.txt
                    time_data = parse_time_file(time_file)
                    if time_data is None:
                        warnings.append(f"Failed to parse time.txt: {instance_dir}")
                        continue
                    
                    # 检查是否存在 input_image_3.png 来判断视频段数（仅用于统计）
                    has_three_segments = (instance_dir / "input_image_3.png").exists()
                    
                    if has_three_segments:
                        three_segment_instances += 1
                        video_segments = 3
                    else:
                        single_segment_instances += 1
                        video_segments = 1
                    
                    # 直接累加 time.txt 中的总时间（不除以任何数）
                    if 'generate_time' in time_data:
                        total_generate_time += time_data['generate_time']
                    if 'denoise_time' in time_data:
                        total_denoise_time += time_data['denoise_time']
                    
                    total_video_segments += video_segments
                    total_instances += 1
    
    # 计算平均值
    if total_video_segments > 0:
        avg_generate_time = total_generate_time / total_video_segments
        avg_denoise_time = total_denoise_time / total_video_segments
    else:
        avg_generate_time = 0.0
        avg_denoise_time = 0.0
    
    # 输出结果
    print_banner("Voyager 生成时间统计")
    print(f"模型: {model_name}")
    print(f"总实例数: {total_instances}")
    print(f"  - 三段视频实例数: {three_segment_instances}")
    print(f"  - 单段视频实例数: {single_segment_instances}")
    print(f"总视频段数: {total_video_segments}")
    print(f"平均 generate_time (每段): {avg_generate_time:.4f} 秒")
    print(f"平均 denoise_time (每段): {avg_denoise_time:.4f} 秒")
    
    # 输出警告（如果有）
    if warnings:
        print(f"\n警告数量: {len(warnings)}")
        if len(warnings) <= 10:
            print("警告详情:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print(f"警告详情（前10条）:")
            for warning in warnings[:10]:
                print(f"  - {warning}")
            print(f"  ... 还有 {len(warnings) - 10} 条警告")


def main(model_name: str, acceleration_mode: str = None) -> None:
    """
    主函数
    
    Args:
        model_name: 模型名称（如 'voyager'）
        acceleration_mode: 加速模式。Options: 'original', 'ToCa', 'DuCa', 'Taylor'.
            如果提供，将使用对应加速模式的输出目录（带后缀）。
            如果为 None，使用默认输出目录。
    """
    calculate_time_stats(model_name, acceleration_mode=acceleration_mode)


if __name__ == "__main__":
    fire.Fire(main)
