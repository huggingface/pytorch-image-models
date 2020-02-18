dependencies = ['torch']

from timm.models import registry

current_module = __import__(__name__)
current_module.__dict__.update(registry._model_entrypoints)
#for fn_name in registry.list_models():
#    fn = registry.model_entrypoint(fn_name)
#    setattr(current_module, fn_name, fn)

