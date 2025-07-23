from orbax.checkpoint import PyTreeCheckpointer
import pprint
import jax
# 替换为你的路径
ckpt_path = '/home/anker/robotwin/Pi0-RL-RoboTwin/checkpoints/checkpoint_3000'

# 加载
checkpointer = PyTreeCheckpointer()
restored = checkpointer.restore(ckpt_path)

# 只打印顶层 keys（最关键！）
print("Top-level keys:")
for k in restored.keys():
    print(f"  - {k}")

