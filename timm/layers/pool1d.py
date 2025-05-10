import torch


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x