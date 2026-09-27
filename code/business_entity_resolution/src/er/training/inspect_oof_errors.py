"""Inspect Out-Of-Fold Errors on Fold 0 with UTF-8 encoding and exact feature names."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from er.features import FEATURES_V3, rows_to_matrix

def main():
    folds_cfg = json.loads(Path('splits/f05-v1/parallel-v1/inner_train_folds.json').read_text(encoding='utf-8'))
    val_qids = {q for q, f in folds_cfg['fold_by_query'].items() if f == 0}

    l04_dir = Path('runs/local-v2/L04_richer_features')
    bst_arm1 = lgb.Booster(model_file=str(l04_dir / 'l04_rich_matcher.txt'))

    df_s1 = pd.read_csv('student_resource/student_resource/dataset/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False).set_index('entity_id')

    for country in ('India', 'US'):
        pool_dict, _ = joblib.load(f'cache/retrieval/pool_dict_{country}.joblib')
        res = joblib.load(l04_dir / f'l04_features_{country}.joblib')['results']
        val_rows, val_pairs, val_lbls = [], [], []
        for item in res:
            qid, q_rows, q_lbls, _, q_pairs = item
            if qid in val_qids:
                val_rows.extend(q_rows)
                val_pairs.extend(q_pairs)
                val_lbls.extend(q_lbls)
                
        X_val = rows_to_matrix(val_rows)
        probs = bst_arm1.predict(X_val)
        lbls = np.array(val_lbls)
        fn_indices = np.flatnonzero((probs < 0.655) & (lbls == 1))
        fp_indices = np.flatnonzero((probs >= 0.655) & (lbls == 0))
        
        print(f'\n=================== {country} Sample False Negatives (Total={len(fn_indices)}) ===================')
        for idx in fn_indices[:5]:
            qid, tid = val_pairs[idx]
            p = probs[idx]
            q_rec = df_s1.loc[qid]
            t_rec = pool_dict[tid]
            row = val_rows[idx]
            feat_dict = dict(zip(FEATURES_V3, row))
            print(f'Query ({qid}):  Name: "{q_rec["business_name"]}" | Addr: "{q_rec["business_address"]}"')
            print(f'Target ({tid}): Name: "{t_rec["business_name"]}" | Addr: "{t_rec["business_address"]}"')
            print(f'  Model Prob: {p:.4f} (Threshold 0.655)')
            print(f'  Key Feats: name_wratio={feat_dict.get("name_wratio", 0):.3f}, name_sort={feat_dict.get("name_sort", 0):.3f}, addr_sort={feat_dict.get("addr_sort", 0):.3f}, addr_word_jac={feat_dict.get("addr_word_jac", 0):.3f}, pin_equal={feat_dict.get("pin_equal", 0)}, same_addr_diff_name={feat_dict.get("same_addr_diff_name", 0)}\n')

        print(f'\n=================== {country} Sample False Positives (Total={len(fp_indices)}) ===================')
        for idx in fp_indices[:5]:
            qid, tid = val_pairs[idx]
            p = probs[idx]
            q_rec = df_s1.loc[qid]
            t_rec = pool_dict[tid]
            row = val_rows[idx]
            feat_dict = dict(zip(FEATURES_V3, row))
            print(f'Query ({qid}):  Name: "{q_rec["business_name"]}" | Addr: "{q_rec["business_address"]}"')
            print(f'Target ({tid}): Name: "{t_rec["business_name"]}" | Addr: "{t_rec["business_address"]}"')
            print(f'  Model Prob: {p:.4f} (Threshold 0.655)')
            print(f'  Key Feats: name_wratio={feat_dict.get("name_wratio", 0):.3f}, name_sort={feat_dict.get("name_sort", 0):.3f}, addr_sort={feat_dict.get("addr_sort", 0):.3f}, addr_word_jac={feat_dict.get("addr_word_jac", 0):.3f}, pin_equal={feat_dict.get("pin_equal", 0)}, same_name_diff_addr={feat_dict.get("same_name_diff_addr", 0)}\n')

if __name__ == '__main__':
    main()
