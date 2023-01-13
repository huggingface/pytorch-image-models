""" JIT scripting/tracing utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import os

import torch


def set_jit_legacy():
    """ Set JIT executor to legacy w/ support for op fusion
    This is hopefully a temporary need in 1.5/1.5.1/1.6 to restore performance due to changes
    in the JIT exectutor. These API are not supported so could change.
    """
    #
    assert hasattr(torch._C, '_jit_set_profiling_executor'), "Old JIT behavior doesn't exist!"
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(True)
    #torch._C._jit_set_texpr_fuser_enabled(True)


def set_jit_fuser(fuser):
    if fuser == "te":
        # default fuser should be == 'te'
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        try:
            torch._C._jit_set_nvfuser_enabled(False)
        except Exception:
            pass
    elif fuser == "old" or fuser == "legacy":
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        try:
            torch._C._jit_set_nvfuser_enabled(False)
        except Exception:
            pass
    elif fuser == "nvfuser" or fuser == "nvf":
        os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
        #os.environ['PYTORCH_NVFUSER_DISABLE_FMA'] = '1'
        #os.environ['PYTORCH_NVFUSER_JIT_OPT_LEVEL'] = '0'
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_nvfuser_guard_mode(True)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        assert False, f"Invalid jit fuser ({fuser})"
