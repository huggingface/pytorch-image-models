import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.models import create_model

_logger = logging.getLogger(__name__)

class build_kd_model(nn.Module):
    def __init__(self, args):
        super(build_kd_model, self).__init__()

        _logger.info(f"Creating KD model: from '{args.kd_model_name}'")
        in_chans = 3
        if args.in_chans is not None:
            in_chans = args.in_chans
        model_kd = create_model(
            model_name=args.kd_model_name,
            num_classes=args.num_classes,
            pretrained=True,
            in_chans=in_chans)

        # compile model
        model_kd.cpu().eval()
        try:
            model_kd = torch.compile(model_kd)
            _logger.info(f"torch.compile applied successfully to KD model")
        except Exception as e:
            _logger.warning(f"torch.compile failed with error {e}, continuing KD model without torch compilation")

        self.model = model_kd.cuda()
        self.mean_model_kd = model_kd.default_cfg['mean']
        self.std_model_kd = model_kd.default_cfg['std']

    # handling different normalization of teacher and student
    def normalize_input(self, input, student_model):
        if hasattr(student_model, 'module'):
            model_s = student_model.module
        else:
            model_s = student_model

        mean_student = model_s.default_cfg['mean']
        std_student = model_s.default_cfg['std']

        input_kd = input
        if mean_student != self.mean_model_kd or std_student != self.std_model_kd:
            std = (self.std_model_kd[0] / std_student[0], self.std_model_kd[1] / std_student[1],
                   self.std_model_kd[2] / std_student[2])
            transform_std = T.Normalize(mean=(0, 0, 0), std=std)

            mean = (self.mean_model_kd[0] - mean_student[0], self.mean_model_kd[1] - mean_student[1],
                    self.mean_model_kd[2] - mean_student[2])
            transform_mean = T.Normalize(mean=mean, std=(1, 1, 1))

            input_kd = transform_mean(transform_std(input))

        return input_kd


def add_kd_loss(_loss, output, input, model, model_kd, args):
    # student probability calculation
    prob_s = torch.nn.functional.log_softmax(output, dim=-1)

    # teacher probability calculation
    with torch.no_grad():
        input_kd = model_kd.normalize_input(input, model)
        out_t = model_kd.model(input_kd.detach())
        prob_t = torch.nn.functional.softmax(out_t, dim=-1)

    # adding KL loss
    if not args.use_kd_only_loss:
        _loss += args.alpha_kd * torch.nn.functional.kl_div(prob_s, prob_t, reduction='batchmean')
    else:  # only kd
        _loss = args.alpha_kd * torch.nn.functional.kl_div(prob_s, prob_t, reduction='batchmean')

    return _loss

