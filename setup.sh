conda create -y -n hallucination python=3.10 numpy scipy ipykernel pandas scikit-learn
source activate hallucination
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/transformers.git
pip install matplotlib seaborn sentencepiece evaluate einops rouge-score gputil captum dotenv
pip install accelerate==0.30.0 pyvene==0.1.2

pip install selfcheckgpt spacy
python -m spacy download en_core_web_sm

# export LD_LIBRARY_PATH=/home/ec2-user/anaconda3/envs/hallucination/lib:$LD_LIBRARY_PATH
