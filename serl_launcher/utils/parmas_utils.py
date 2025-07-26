import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import jax.numpy as jnp
import jax
import numpy as np
from pi0.src.openpi.models import model, pi0_nn as pi0


def print_nested_structure(params, indent=0):
    """
    递归打印嵌套字典的结构、键和值的形状。

    Args:
        params (dict or jnp.ndarray or other): 要打印的参数 (PyTree)。
        indent (int): 当前缩进级别，用于格式化输出。
    """
    spacing = "  " * indent
    if isinstance(params, dict):
        # 如果是字典，先打印字典的标识和键
        # 为了更清晰，可以显示字典中键的概览
        keys_str = ", ".join([f"'{k}'" for k in params.keys()])
        print(f"{spacing}{{ {keys_str} }}")
        # 然后递归打印每个键的值
        for key, value in params.items():
            print(f"{spacing}'{key}':")
            print_nested_structure(value, indent + 1)
    elif isinstance(params, (jnp.ndarray, jax.Array, np.ndarray)): # 检查是否为数组
        # 如果是数组，打印其形状和数据类型
        print(f"{spacing}Array(shape={params.shape}, dtype={params.dtype})")
    else:
        # 对于其他类型（如 None, int, float 等），打印类型和值
        print(f"{spacing}{type(params).__name__}({params})")


def inspect_params_structure(params_tree, max_depth=3, current_depth=0):
    """检查参数树结构"""
    if current_depth > max_depth:
        return
    
    indent = "  " * current_depth
    if isinstance(params_tree, dict):
        print(f"{indent}Dict with keys: {list(params_tree.keys())}")
        # 只检查前几个键，避免输出过多
        keys_to_check = list(params_tree.keys())[:3]
        for key in keys_to_check:
            print(f"{indent}  '{key}':")
            inspect_params_structure(params_tree[key], max_depth, current_depth + 1)
            if len(params_tree.keys()) > 3:
                print(f"{indent}  ... (and {len(params_tree.keys()) - 3} more keys)")
                break
    else:
        print(f"{indent}{type(params_tree)} with shape: {getattr(params_tree, 'shape', 'N/A')}")


def merge_lora_weights_in_tree(params_tree):
    """
    遍历参数树，找到包含 LoRA 权重的节点并进行合并。
    """
    merged_count = {'attn': 0, 'mlp': 0}
    
    def process_node(node, path=""):
        """递归处理节点"""
        if not isinstance(node, dict):
            return node
        
        # 检查 attn-style LoRA: {'w', 'lora_a', 'lora_b'}
        if 'w' in node and 'lora_a' in node and 'lora_b' in node:
            try:
                w = node['w']
                lora_a = node['lora_a']
                lora_b = node['lora_b']
                
                # 正确的 LoRA 合并: w_new = w + lora_a @ lora_b
                delta_w = jnp.matmul(lora_a, lora_b)
                merged_w = w + delta_w
                del node['lora_a']
                del node['lora_b']
                merged_count['attn'] += 1
                return {'w': merged_w}
            except Exception as e:
                print(f"[ERROR] Failed merging attn LoRA at {path}: {e}")
                return node
        
        # 检查 mlp-style LoRA: {'xxx', 'xxx_lora_a', 'xxx_lora_b'}
        keys_to_process = set()
        for key in node.keys():
            if key.endswith('_lora_a'):
                base_name = key[:-len('_lora_a')]
                if f"{base_name}" in node and f"{base_name}_lora_b" in node:
                    keys_to_process.add(base_name)
        
        if keys_to_process:
            new_node = dict(node)
            processed_count = 0
            for base_name in keys_to_process:
                weight_key = base_name
                lora_a_key = f"{base_name}_lora_a"
                lora_b_key = f"{base_name}_lora_b"
                
                if all(k in new_node for k in [weight_key, lora_a_key, lora_b_key]):
                    try:
                        weight = new_node[weight_key]
                        lora_a = new_node[lora_a_key]
                        lora_b = new_node[lora_b_key]
                        
                        # 正确的 LoRA 合并: weight_new = weight + lora_a @ lora_b
                        delta_weight = jnp.matmul(lora_a, lora_b)
                        merged_weight = weight + delta_weight
                        new_node[weight_key] = merged_weight
                        del new_node[lora_a_key]
                        del new_node[lora_b_key]
                        processed_count += 1
                        merged_count['mlp'] += 1
                    except Exception as e:
                        print(f"[ERROR] Failed merging MLP LoRA '{base_name}' at {path}: {e}")
            
            return new_node
        
        # 递归处理所有子节点
        return {k: process_node(v, f"{path}.{k}" if path else k) for k, v in node.items()}
    
    print("[INFO] Starting LoRA merging...")
    result = process_node(params_tree)
    print(f"[INFO] LoRA merging completed. Merged {merged_count['attn']} attn LoRA and {merged_count['mlp']} MLP LoRA.")
    return result

def create_policy(pretrained_policy_path = None, lora = True):
    if lora:
        policy_config=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    else:
        policy_config=pi0.Pi0Config(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    if pretrained_policy_path is not None:
        pretrained_actor_params = model.restore_params(pretrained_policy_path, dtype=jnp.float32)
        if not lora:
            pretrained_actor_params = merge_lora_weights_in_tree(pretrained_actor_params)
    else:
        raise ValueError("pretrained_policy_path must be provided for post training")
    policy_def = pi0.Pi0(config=policy_config)
    return policy_def, pretrained_actor_params