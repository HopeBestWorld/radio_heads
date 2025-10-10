source /home/radio_heads/radio_env/bin/activate
pip install torch==2.8.0+cu129 torchvision torchaudio   --index-url https://download.pytorch.org/whl/cu129   --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-build-isolation -r requirements.txt
python -m ipykernel install --user --name=pytorch --display-name "Python (PyTorch)"
source /opt/pytorch/bin/activate
source /opt/radio_env/bin/activate
