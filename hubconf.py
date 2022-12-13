dependencies = ['torch']
import timm
globals().update(timm.models._registry._model_entrypoints)
