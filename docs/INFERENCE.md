# Inference Instructions

## Install as Package (for Policy Interaction)

If you want to use Ctrl-World's `WorldModelEnv` in another repository for policy interaction:

```bash
# Install package in editable mode
cd /path/to/Ctrl-World
pip install -e .
```

Then import from anywhere:
```python
from models.wm_env import WorldModelEnv

# Initialize the environment
env = WorldModelEnv(wm_ckpt="path/to/checkpoint.pth", control_mode="joint_velocity")
obs, info = env.reset(idx=0)
```