import numpy as np
import pandas as pd


results = {
    'results-imagenet.csv': [
        'results-imagenet-real.csv',
        'results-imagenetv2-matched-frequency.csv',
        'results-sketch.csv'
    ],
    'results-imagenet-a-clean.csv': [
        'results-imagenet-a.csv',
    ],
    'results-imagenet-r-clean.csv': [
        'results-imagenet-r.csv',
    ],
}


def diff(base_df, test_csv):
    base_models = base_df['model'].values
    test_df = pd.read_csv(test_csv)
    test_models  = test_df['model'].values

    rank_diff = np.zeros_like(test_models, dtype='object')
    top1_diff = np.zeros_like(test_models, dtype='object')
    top5_diff = np.zeros_like(test_models, dtype='object')
    
    for rank, model in enumerate(test_models):
        if model in base_models:            
            base_rank = int(np.where(base_models == model)[0])
            top1_d = test_df['top1'][rank] - base_df['top1'][base_rank]
            top5_d = test_df['top5'][rank] - base_df['top5'][base_rank]
            
            # rank_diff
            if rank == base_rank:
                rank_diff[rank] = f'0'
            elif rank > base_rank:
                rank_diff[rank] = f'-{rank - base_rank}'
            else:
                rank_diff[rank] = f'+{base_rank - rank}'
                
            # top1_diff
            if top1_d >= .0:
                top1_diff[rank] = f'+{top1_d:.3f}'
            else:
                top1_diff[rank] = f'-{abs(top1_d):.3f}'
            
            # top5_diff
            if top5_d >= .0:
                top5_diff[rank] = f'+{top5_d:.3f}'
            else:
                top5_diff[rank] = f'-{abs(top5_d):.3f}'
                
        else: 
            rank_diff[rank] = ''
            top1_diff[rank] = ''
            top5_diff[rank] = ''

    test_df['top1_diff'] = top1_diff
    test_df['top5_diff'] = top5_diff
    test_df['rank_diff'] = rank_diff

    test_df['param_count'] = test_df['param_count'].map('{:,.2f}'.format)
    test_df.sort_values(['top1', 'top5', 'model'], ascending=[False, False, True], inplace=True)
    test_df.to_csv(test_csv, index=False, float_format='%.3f')


for base_results, test_results in results.items():
    base_df = pd.read_csv(base_results)
    base_df.sort_values(['top1', 'top5', 'model'], ascending=[False, False, True], inplace=True)
    for test_csv in test_results:
        diff(base_df, test_csv)
    base_df['param_count'] = base_df['param_count'].map('{:,.2f}'.format)
    base_df.to_csv(base_results, index=False, float_format='%.3f')
