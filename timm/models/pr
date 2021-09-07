def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

from einops import rearrange

def interpolate(raw_param: torch.tensor, pretrained_weight: torch.tensor, token_num: int = 1):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', pretrained_weight.shape, raw_param.shape)
    
    tokens, positional_embedding = pretrained_weight[:, :token_num], pretrained_weight[:, token_num:]
    raw_dim, pretrained_dim = get_dim(raw_param), get_dim(positional_embedding)
    _logger.info('Position embedding grid-size from %s to %s', [pretrained_dim, pretrained_dim], raw_dim)
    
    grid_positional_embedding = rearrange(positional_embedding, '1 (h w) d -> 1 d h w', h=pretrained_dim)
    resized_grid_positional_embedding = F.interpolate(grid_positional_embedding, size=[raw_dim, raw_dim], mode='bicubic', align_corners=False)
    positional_embedding = rearrange(resized_grid_positional_embedding, '1 d h w -> 1 (h w) d')
    
    pretrained_weight = torch.cat([tokens, positional_embedding], dim=1)
    
    return pretrained_weight


def get_dim(weight):
    return int(math.sqrt(float(weight.size(1))))
