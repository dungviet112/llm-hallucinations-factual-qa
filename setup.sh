pip install uv
uv venv --python 3.10 
source hallucination/bin/activate
uv pip install numpy scipy ipykernel pandas scikit-learn
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install git+https://github.com/huggingface/transformers.git
uv pip install matplotlib seaborn sentencepiece evaluate einops rouge-score gputil captum dotenv
uv pip install accelerate==0.30.0 pyvene==0.1.2

uv pip install selfcheckgpt spacy
uv run python -m spacy download en_core_web_sm

# export LD_LIBRARY_PATH=/home/ec2-user/anaconda3/envs/hallucination/lib:$LD_LIBRARY_PATH
