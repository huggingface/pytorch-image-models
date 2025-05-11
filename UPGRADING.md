# Upgrading from previous versions

I generally try to maintain code interface and especially model weight compability across many `timm` versions. Sometimes there are exceptions.

## Checkpoint remapping

Pretrained weight remapping is handled by `checkpoint_filter_fn` in a model implementation module. This remaps old pretrained checkpoints to new, and also 3rd party (original) checkpoints to `timm` format if the model was modified when brough into `timm`.

The `checkpoint_filter_fn` is automatically called when loading pretrained weights via `pretrained=True`, but they can be called manually if you call the fn directly with the current model instance and old state dict.

## Upgrading from 0.6 and earlier

Many changes were made since the 0.6.x stable releases. They were previewed in 0.8.x dev releases but not everyone transitioned.
* `timm.models.layers` moved to `timm.layers`:
  * `from timm.models.layers import name` will still work via deprecation mapping (but please transition to `timm.layers`).
  * `import timm.models.layers.module` or `from timm.models.layers.module import name` needs to be changed now.
* Builder, helper, non-model modules in `timm.models` have a `_` prefix added, ie `timm.models.helpers` -> `timm.models._helpers`, there are temporary deprecation mapping files but those will be removed.
* All models now support `architecture.pretrained_tag` naming (ex `resnet50.rsb_a1`).
  * The pretrained_tag is the specific weight variant (different head) for the architecture.
  * Using only `architecture` defaults to the first weights in the default_cfgs for that model architecture.
  * In adding pretrained tags, many model names that existed to differentiate were renamed to use the tag  (ex: `vit_base_patch16_224_in21k` -> `vit_base_patch16_224.augreg_in21k`). There are deprecation mappings for these.
* A number of models had their checkpoints remaped to match architecture changes needed to better support `features_only=True`, there are `checkpoint_filter_fn` methods in any model module that was remapped. These can be passed to `timm.models.load_checkpoint(..., filter_fn=timm.models.swin_transformer_v2.checkpoint_filter_fn)` to remap your existing checkpoint.
* The Hugging Face Hub (https://huggingface.co/timm) is now the primary source for `timm` weights. Model cards include link to papers, original source, license. 
* Previous 0.6.x can be cloned from [0.6.x](https://github.com/rwightman/pytorch-image-models/tree/0.6.x) branch or installed via pip with version.
