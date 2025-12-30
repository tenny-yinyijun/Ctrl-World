# Ctrl-World (Adapted)

Original README: [readme.md](readme.md)

1. [Installation]()
2. [Training](docs/TRAIN.md)
3. [Evaluation](docs/EVAL.md)
4. [Inference](docs/INFERENCE.md)

## Installation

First create the environment with either uv or conda:
```bash
## Option 1 (recommended): uv

# Install 
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

### Option 2: conda
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt
```

Next, install the required dependencies:

```bash
cd Ctrl-World
bash bash_scripts/setup/download_models.sh
```