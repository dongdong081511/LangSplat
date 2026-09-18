#!/usr/bin/env python
"""Wrapper to run train.py with CUDA pre-initialized (sandbox workaround).

Usage: python run_train_mm.py [same args as train.py]
"""
import os
import sys
import torch

# Pre-initialize CUDA by moving a tensor to GPU
# This must happen BEFORE importing any CUDA extensions
try:
    _ = torch.tensor([1.0]).cuda()
    print(f"CUDA pre-initialized: {torch.cuda.get_device_name(0)}")
except RuntimeError as e:
    print(f"WARNING: CUDA pre-init failed: {e}")
    print("Continuing anyway...")

# Now exec train.py with the same arguments
train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
with open(train_path) as f:
    code = f.read()

# Set __name__ to '__main__' so train.py's main block executes
exec(compile(code, train_path, 'exec'), {'__name__': '__main__', '__file__': train_path})
