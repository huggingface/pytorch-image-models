import inspect
import re
from typing import Optional, Tuple, List

import torch


def _torch_version() -> Tuple[int, int]:
    major, minor = re.match(r'(\d+)\.(\d+)', torch.__version__).groups()
    return int(major), int(minor)


def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output


def onnx_export(
        model: torch.nn.Module,
        output_file: str,
        example_input: Optional[torch.Tensor] = None,
        training: bool = False,
        verbose: bool = False,
        check: bool = True,
        check_forward: bool = False,
        batch_size: int = 64,
        input_size: Tuple[int, int, int] = None,
        opset: Optional[int] = None,
        dynamic_size: bool = False,
        aten_fallback: bool = False,
        keep_initializers: Optional[bool] = None,
        use_dynamo: bool = False,
        input_names: List[str] = None,
        output_names: List[str] = None,
        external_data: Optional[bool] = None,
        optimize: bool = True,
):
    """ Export a model to ONNX.

    Args:
        model: Model to export.
        output_file: Output ONNX filename.
        example_input: Example input tensor, created from batch_size and input_size if not provided.
        training: Export in training mode.
        verbose: Verbose export output.
        check: Run the ONNX model checker on the exported model.
        check_forward: Compare ONNX runtime output against PyTorch output (eval mode only).
        batch_size: Batch size of example input.
        input_size: Input size (C, H, W) of example input, uses model pretrained_cfg if not set.
        opset: ONNX opset version, uses exporter default if not set.
        dynamic_size: Export with dynamic height and width.
        aten_fallback: Fallback to ATen ops (TorchScript exporter only).
        keep_initializers: Keep initializers as graph inputs.
        use_dynamo: Use the torch.export (dynamo) based exporter instead of the TorchScript exporter.
        input_names: Names of graph inputs.
        output_names: Names of graph outputs.
        external_data: Store weights in a separate '.data' file (dynamo exporter only). Defaults to
            enabling external data only when the model exceeds the 2GB protobuf limit.
        optimize: Run the ONNX graph optimizer after export (dynamo exporter only, PyTorch >= 2.6).
    """
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, 'default_cfg'), 'Cannot file model default config, input size must be provided'
            input_size = model.default_cfg.get('input_size')
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    with torch.inference_mode():
        original_out = model(example_input)

    input_names = input_names or ["input0"]
    output_names = output_names or ["output0"]

    input_axes = {0: 'batch'}
    if dynamic_size:
        input_axes[2] = 'height'
        input_axes[3] = 'width'

    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    # PyTorch >= 2.5 supports dynamo export via torch.onnx.export(..., dynamo=True, dynamic_shapes=...), and
    # PyTorch >= 2.9 defaults `dynamo` to True, so it must be passed explicitly for the TorchScript exporter.
    export_args = inspect.signature(torch.onnx.export).parameters
    has_dynamo_arg = 'dynamo' in export_args

    if use_dynamo and 'dynamic_shapes' in export_args:
        extra_args = {}
        if 'external_data' in export_args:
            if external_data is None:
                param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                param_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
                external_data = param_bytes >= 2 ** 31 - 2 ** 27  # leave headroom below 2GB protobuf limit
            extra_args['external_data'] = external_data
        if 'optimize' in export_args:
            extra_args['optimize'] = optimize
        if _torch_version() >= (2, 7):
            # string axis names in dynamic_shapes are supported (and named in the graph) from PyTorch 2.7
            extra_args['dynamic_shapes'] = (input_axes,)
        else:
            # PyTorch 2.5 - 2.6, strings not supported and dynamic_axes are converted to strict Dims that
            # fail on common shape constraints and fall back to static shapes, use non-strict Dims instead
            dim = getattr(torch.export.Dim, 'DYNAMIC', torch.export.Dim.AUTO)
            extra_args['dynamic_shapes'] = ({k: dim for k in input_axes},)
        # torch.export specializes size 1 dims (and ops such as reshape may bake the size in with some
        # PyTorch versions), trace with batch size >= 2 to keep batch dynamic.
        export_input = example_input
        if export_input.shape[0] == 1:
            export_input = torch.cat([example_input, example_input])
        torch.onnx.export(
            model,
            (export_input,),
            output_file,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            keep_initializers_as_inputs=bool(keep_initializers),
            dynamo=True,
            **extra_args,
        )
    elif use_dynamo:
        # PyTorch 2.1 - 2.4, torch.onnx.dynamo_export was removed in 2.9
        assert hasattr(torch.onnx, 'dynamo_export'), 'Dynamo ONNX export requires PyTorch >= 2.1'
        export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic_size)
        export_output = torch.onnx.dynamo_export(
            model,
            example_input,
            export_options=export_options,
        )
        export_output.save(output_file)
    else:
        extra_args = {'dynamo': False} if has_dynamo_arg else {}
        torch.onnx.export(
            model,
            example_input,
            output_file,
            training=training_mode,
            export_params=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            keep_initializers_as_inputs=keep_initializers,
            dynamic_axes={input_names[0]: input_axes, output_names[0]: {0: 'batch'}},
            opset_version=opset,
            operator_export_type=export_type,
            **extra_args,
        )

    if check:
        # check by path so models with external data (> 2GB) can be checked
        onnx.checker.check_model(output_file, full_check=True)  # assuming throw on error
        if check_forward and not training:
            import numpy as np
            onnx_out = onnx_forward(output_file, example_input)
            np.testing.assert_almost_equal(original_out.numpy(), onnx_out, decimal=3)

