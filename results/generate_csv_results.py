import numpy as np
import pandas as pd

results = {
    'results-imagenet.csv': pd.read_csv('results-imagenet.csv'),
    'results-imagenetv2-matched-frequency.csv': pd.read_csv('results-imagenetv2-matched-frequency.csv'),
    'results-sketch.csv': pd.read_csv('results-sketch.csv'),
    'results-imagenet-a.csv': pd.read_csv('results-imagenet-a.csv'),
    'results-imagenet-r.csv': pd.read_csv('results-imagenet-r.csv'),
    'results-imagenet-real.csv': pd.read_csv('results-imagenet-real.csv'),
}


def diff(csv_file):    
    base_models = results['results-imagenet.csv']['model'].values
    csv_models  = results[csv_file]['model'].values

    rank_diff = np.zeros_like(csv_models, dtype='object')
    top1_diff = np.zeros_like(csv_models, dtype='object')
    top5_diff = np.zeros_like(csv_models, dtype='object')
    
    for rank, model in enumerate(csv_models):
        if model in base_models:            
            base_rank = int(np.where(base_models==model)[0])
            top1_d = results[csv_file]['top1'][rank]-results['results-imagenet.csv']['top1'][base_rank]
            top5_d = results[csv_file]['top5'][rank]-results['results-imagenet.csv']['top5'][base_rank]
            
            # rank_diff
            if   rank == base_rank: rank_diff[rank] = f'='
            elif rank >  base_rank: rank_diff[rank] = f'-{rank-base_rank}'
            else:                   rank_diff[rank] = f'+{base_rank-rank}'
                
            # top1_diff
            if top1_d >= .0: top1_diff[rank] = f'+{top1_d:.3f}'
            else           : top1_diff[rank] = f'-{abs(top1_d):.3f}'
            
            # top5_diff
            if top5_d >= .0: top5_diff[rank] = f'+{top5_d:.3f}'
            else           : top5_diff[rank] = f'-{abs(top5_d):.3f}'
                
        else: 
            rank_diff[rank] = 'X'
            top1_diff[rank] = 'X'
            top5_diff[rank] = 'X'
    
    results[csv_file]['rank_diff'] = rank_diff
    results[csv_file]['top1_diff'] = top1_diff
    results[csv_file]['top5_diff'] = top5_diff
    
    results[csv_file]['param_count'] = results[csv_file]['param_count'].map('{:,.2f}'.format)

    results[csv_file].to_csv(csv_file, index=False, float_format='%.3f')


for csv_file in results:
    if csv_file != 'results-imagenet.csv':
        diff(csv_file)
