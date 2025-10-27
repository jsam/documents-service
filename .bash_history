ls
python
python -c "import flash_attn; print('Flash attention installed:', flash_attn.__version__)" || echo "Flash attention NOT installed"
/code/.venv/bin/python -c "import flash_attn; print('Flash attention installed:', flash_attn.__version__)" || echo "Flash attention NOT installed"
which python
echo $PATH
ls -la /code/.venv/
ls -la /code/.venv/lib/python3.12/site-packages/ | grep flash
find /code/.venv -name "*flash*" -type f
find /code/.venv -name "*flash*" -type d
/code/.venv/bin/python -c "import sys; print(sys.path)"
/code/.venv/bin/python -c "import pkg_resources; print([p.project_name for p in pkg_resources.working_set])"
/code/.venv/bin/pip list | grep flash
/code/.venv/bin/pip show flash-attn
pip show flash-attn
/code/.venv/bin/python -c "import importlib.util; spec = importlib.util.find_spec('flash_attn'); print('Found:', spec)"
/code/.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
code/.venv/bin/python -c "import torch; torch.ops.load_library('/code/.venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so')" 2>&1 || echo "Direct
   load failed"
/code/.venv/bin/python -c "import torch; torch.ops.load_library('/code/.venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so')" 2>&1 || echo "Direct
   load failed"
ldd /code/.venv/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null || echo "No .so file found"
which uv
/code/.venv/bin/uv --version
uv pip list --python /code/.venv/bin/python
cat /code/uv.lock | grep -A5 -B5 flash-attn
find /code/.venv -name "flash_attn*" -type d
find /code/.venv -name "*flash_attn*"
find /root/.cache/uv -name "*flash*" | head -10
cd /code && uv pip install flash-attn --python /code/.venv/bin/python --verbose
cd /code && uv pip install flash-attn --python /code/.venv/bin/python --no-cache --verbose
pkill -f "uv pip install flash-attn"
uv pip install flash-attn==2.6.3 --find-links https://github.com/Dao-AILab/flash-attention/releases --python /code/.venv/bin/python
/code/.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu124torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.6.3 | grep "browser_download_url.*whl" | head -10
grep -r "flash_attn" /code/server/
grep flash-attn /code/pyproject.toml
grep -r "flash_attn" /code/server/
/code/.venv/bin/python -c "
  try:
      import torch.nn.functional as F
      print('PyTorch has scaled_dot_product_attention:', hasattr(F, 'scaled_dot_product_attention'))
      if hasattr(F, 'scaled_dot_product_attention'):
          print('Available backends:', torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
  except Exception as e:
      print('Error:', e)
  "
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest | grep "tag_name"
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest | grep "browser_download_url.*whl" | grep cu12 | grep cp312 | head -5
/code/.venv/bin/python -c "
  import torch.nn.functional as F
  print('PyTorch has scaled_dot_product_attention:', hasattr(F, 'scaled_dot_product_attention'))
  import torch
  print('Flash SDP enabled:', torch.backends.cuda.flash_sdp_enabled())
  print('Math SDP enabled:', torch.backends.cuda.math_sdp_enabled())
  "
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install ./flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --python /code/.venv/bin/python
/code/.venv/bin/python -c "import flash_attn; print('Flash attention installed:', flash_attn.__version__)"
uv pip uninstall flash-attn --python /code/.venv/bin/python
/code/.venv/bin/python -c "
  try:
      import flash_attn
      print('Flash attention available')
  except ImportError:
      print('Flash attention not available - will use eager attention')
      print('This is fine since the code has fallback logic')
  "
/code/.venv/bin/python -c "
  try:
      import flash_attn
      print('Flash attention available')
  except ImportError:
      print('Flash attention not available - will use eager attention')
      print('This is fine since the code has fallback logic')
"
/code/.venv/bin/python
rm flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
/code/.venv/bin/python
exit
