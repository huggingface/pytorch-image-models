from .version import __version__
from .layers import is_scriptable, is_exportable, set_scriptable, set_exportable
from .models import create_model, list_models, list_pretrained, is_model, list_modules, model_entrypoint, \
    is_model_pretrained, get_pretrained_cfg, get_pretrained_cfg_value
import os
import json
import importlib
import sys

# import device specific accelerator module
device_extension_info = open("./timm/device_extension.json", 'r')
device_extension_info = json.load(device_extension_info)

os_var = ""
for device_key in device_extension_info.keys():
    os_var_modules = device_extension_info[device_key]
    os_var += device_key + ':'

    for module in os_var_modules:
        os_var += module + ':'
    os_var = os_var[:-1]
    os_var += ','

os.environ["DEVICE_EXT"] = os_var[:-1]

if os.getenv('DEVICE_EXT'):
    this_module = sys.modules[__name__]
    backends = os.getenv('DEVICE_EXT').split(',')
    for backend in backends:
        module_info = backend.split(':')
        module_name = module_info[1].strip()
        module_alias = list()
        if len(module_info) > 2:
            for i in range(2, len(module_info)):
                module_alias.append(module_info[i].strip())
        try:
            extra_module = importlib.import_module(module_name)
            for alia in module_alias:
                setattr(this_module, alia, extra_module)
            print(module_alias)
        except ImportError:
            pass
