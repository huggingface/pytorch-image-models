dependencies = ['torch']
from timm.models import registry

globals().update(registry._model_entrypoints)
