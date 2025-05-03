import torch
import argparse
import os
from pprint import pprint
import sys

# --- 动态导入 NamedTransition ---
# 假设 replay.py 位于相对于 inspect_data.py 的 ../src/rl/agents/ 目录下
# 或者根据你的项目结构调整路径
try:
    # 尝试找到 replay.py 并导入 NamedTransition
    # 获取 inspect_data.py 所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (假设 inspect_data.py 在项目根目录下)
    project_root = current_dir # 或者 os.path.dirname(current_dir) 如果脚本在子目录
    # 将 src 目录添加到 Python 路径
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path) # 优先搜索 src 目录
    # 从 agents.replay 导入
    from rl.agents.replay import NamedTransition
    print("成功导入 NamedTransition 类。")
except ImportError as e:
    print(f"错误：无法导入 NamedTransition 类: {e}")
    print("请确保 inspect_data.py 脚本可以找到 src/rl/agents/replay.py 文件。")
    print("你可能需要调整脚本中的 sys.path 设置。")
    NamedTransition = None # 定义为 None 以便后续检查
except Exception as e:
    print(f"导入 NamedTransition 时发生未知错误: {e}")
    NamedTransition = None
# --- 导入结束 ---


def inspect_offline_data(filepath):
    """
    加载并检查离线数据文件的内容。

    Args:
        filepath (str): transitions_all.ckpt 文件的路径。
    """
    print(f"--- 开始检查文件: {filepath} ---")

    # 1. 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"错误：文件不存在于路径: {filepath}")
        return
    print(f"文件存在。")

    # 2. 加载数据文件
    print(f"尝试加载文件 (这对于大文件可能需要一些时间)...")
    try:
        print("注意：正在使用 torch.load(weights_only=False)。确保此文件来源可信。")
        # 如果 NamedTransition 成功导入，允许加载它
        if NamedTransition:
             # weights_only=False 以加载包含 NamedTransition 对象的列表
            loaded_data = torch.load(filepath, map_location='cpu', weights_only=False)
        else:
            # 如果无法导入 NamedTransition，可能无法正确加载，尝试只加载结构
            print("警告：无法导入 NamedTransition，尝试仅加载数据结构（可能不完整）。")
            loaded_data = torch.load(filepath, map_location='cpu', weights_only=True) # Fallback

        print("文件加载成功！")
    except Exception as e:
        print(f"错误：加载文件时出错: {e}")
        print("可能原因：文件损坏、格式不兼容、内存不足或无法反序列化 NamedTransition 对象（如果导入失败）。")
        return

    # 3. 检查加载的数据类型和样本数量
    print(f"\n--- 数据概览 ---")
    data_type = type(loaded_data)
    print(f"加载的数据类型: {data_type}")

    if isinstance(loaded_data, list):
        num_transitions = len(loaded_data)
        print(f"总转换 (样本) 数量: {num_transitions}")

        if num_transitions == 0:
            print("文件中没有样本。")
            print("--- 检查结束 ---")
            return

        # 4. 检查第一个样本的结构
        print(f"\n--- 第一个样本结构 ---")
        first_transition = loaded_data[0]
        first_transition_type = type(first_transition)
        print(f"第一个样本的数据类型: {first_transition_type}")

        # --- 修改：通过类名字符串进行检查 ---
        expected_type_name = "src.rl.agents.replay.NamedTransition"
        # 获取实际类型的完整名称（模块名 + 类名）
        actual_type_name = f"{first_transition.__class__.__module__}.{first_transition.__class__.__name__}"

        print(f"DEBUG: 期望类型名称: {expected_type_name}")
        print(f"DEBUG: 实际类型名称: {actual_type_name}")

        if actual_type_name == expected_type_name:
            print("样本类型名称匹配 'src.rl.agents.replay.NamedTransition'。")
            # 获取对象的所有属性（字段）
            attributes = [attr for attr in dir(first_transition) if not attr.startswith('_') and not callable(getattr(first_transition, attr))]
            print("样本包含的属性 (Fields):")
            pprint(attributes)

            print("\n第一个样本各属性类型和形状/值:")
            for attr in attributes:
                value = getattr(first_transition, attr)
                value_type = type(value)

                if isinstance(value, torch.Tensor):
                    print(f"  - \"{attr}\": 类型={value_type}, 形状={value.shape}, 设备={value.device}, 数据类型={value.dtype}")
                elif isinstance(value, tuple):
                    print(f"  - \"{attr}\": 类型={value_type}, 长度={len(value)}")
                    for i, item in enumerate(value):
                        item_type = type(item)
                        if isinstance(item, torch.Tensor):
                            print(f"    - 元组元素 {i}: 类型={item_type}, 形状={item.shape}, 设备={item.device}, 数据类型={item.dtype}")
                        elif isinstance(item, list):
                             print(f"    - 元组元素 {i}: 类型={item_type}, 长度={len(item)}")
                        else:
                            print(f"    - 元组元素 {i}: 类型={item_type}")
                elif isinstance(value, list):
                     print(f"  - \"{attr}\": 类型={value_type}, 长度={len(value)}")
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    print(f"  - \"{attr}\": 类型={value_type}, 值={value}")
                else:
                    print(f"  - \"{attr}\": 类型={value_type}")

            # 检查关键属性是否存在且类型基本正确
            print("\n关键属性检查:")
            required = ['states', 'action_idx', 'action_space', 'reward', 'next_states', 'next_action_space']
            for key in required:
                has_attr = hasattr(first_transition, key)
                print(f"  - \"{key}\": {'存在' if has_attr else '!! 不存在 !!'}", end="")
                if has_attr:
                    val = getattr(first_transition, key)
                    val_type = type(val)
                    if key in ['states', 'reward']:
                        print(f", 类型={val_type} (应为 Tensor)")
                    elif key == 'action_idx':
                         print(f", 类型={val_type} (应为 int)")
                    elif key in ['action_space', 'next_action_space']:
                         print(f", 类型={val_type} (应为 tuple 或 None)")
                    elif key == 'next_states':
                         print(f", 类型={val_type} (应为 Tensor 或 None)")
                    else:
                         print(f", 类型={val_type}")
                else:
                    print()
        # --- 修改结束 ---

        elif isinstance(first_transition, dict): # 保留对旧格式的处理
             print("加载的数据是一个字典。这可能是一个旧格式或不同的保存结构。")
             print("字典的键:")
             pprint(list(first_transition.keys()))
        else:
            # 如果类型名称不匹配
            print(f"错误：期望类型名称为 '{expected_type_name}'，但实际为 '{actual_type_name}'。")
            print("请检查数据生成脚本或 inspect_data.py 中的导入设置。")

    else:
        print("错误：加载的数据不是预期的列表格式。无法分析样本。")

    print("\n--- 检查结束 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查离线 RL 数据文件 (.ckpt)")
    parser.add_argument("filepath", type=str, help="要检查的 transitions_all.ckpt 文件路径")
    args = parser.parse_args()

    inspect_offline_data(args.filepath)