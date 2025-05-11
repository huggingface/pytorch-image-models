from typing import Optional, Tuple, List

import torch


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
):
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, 'default_cfg')
            input_size = model.default_cfg.get('input_size')
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    with torch.no_grad():
        original_out = model(example_input)

    input_names = input_names or ["input0"]
    output_names = output_names or ["output0"]

    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
    if dynamic_size:
        dynamic_axes['input0'][2] = 'height'
        dynamic_axes['input0'][3] = 'width'

    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    if use_dynamo:
        export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic_size)
        export_output = torch.onnx.dynamo_export(
            model,
            example_input,
            export_options=export_options,
        )
        export_output.save(output_file)
        torch_out = None
    else:
        torch_out = torch.onnx._export(
            model,
            example_input,
            output_file,
            training=training_mode,
            export_params=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            keep_initializers_as_inputs=keep_initializers,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            operator_export_type=export_type
        )

    if check:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        if check_forward and not training:
            import numpy as np
            onnx_out = onnx_forward(output_file, example_input)
            if torch_out is not None:
                np.testing.assert_almost_equal(torch_out.numpy(), onnx_out, decimal=3)
                np.testing.assert_almost_equal(original_out.numpy(), torch_out.numpy(), decimal=5)
            else:
                np.testing.assert_almost_equal(original_out.numpy(), onnx_out, decimal=3)

