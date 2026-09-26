# L04 Rich Matcher Runner
from __future__ import annotations
import argparse, json, time, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib, lightgbm as lgb, numpy as np, pandas as pd
from scipy.sparse import load_npz
from er.analyze_b0_decisions import apply_policy, counts, exact_scores, prepare, select_policies
from er.candidate_generation import generate_natural_candidates
from er.features import FEATURES_V3, pair_feature_row_v3, rows_to_matrix
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12

def bootstrap_delta(delta: np.ndarray, countries: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    unique_c = sorted(set(countries))
    strata = [np.flatnonzero(countries == c) for c in unique_c]
    boot_means = np.array([
        np.concatenate([delta[rng.choice(s, len(s), replace=True)] for s in strata]).mean()
        for _ in range(n_boot)
    ])
    ci = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    return float(delta.mean()), ci

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', default='student_resource/student_resource/dataset/train')
    ap.add_argument('--gt', default='student_resource/student_resource/dataset/train/train_ground_truth.tsv')
    ap.add_argument('--manifest-dir', default='splits/f05-v1/parallel-v1')
    ap.add_argument('--cache-dir', default='cache/retrieval')
    ap.add_argument('--out-dir', default='runs/local-v2/L04_richer_features')
    ap.add_argument('--reports-dir', default='reports/dev_probe')
    ap.add_argument('--n-cores', type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(' L04: Richer Structured Features (38 Features)')
    print(f' CPU Cores: {args.n_cores} | Output Dir: {out_dir}')
    print('=' * 70)

    gt_map = load_gt_map(Path(args.gt))
    train_info = json.loads((manifest_dir / 'train_12k.json').read_text(encoding='utf-8'))
    calib_info = json.loads((manifest_dir / 'calibration_5k.json').read_text(encoding='utf-8'))
    screen_info = json.loads((manifest_dir / 'screen_2k.json').read_text(encoding='utf-8'))
    inner_folds_info = json.loads((manifest_dir / 'inner_train_folds.json').read_text(encoding='utf-8'))

    train_qids = train_info['query_ids']
    calib_qids = calib_info['query_ids']
    screen_qids = screen_info['query_ids']
    inner_folds_map = inner_folds_info['fold_by_query']

    train_set = set(train_qids)
    calib_set = set(calib_qids)
    screen_set = set(screen_qids)
    all_qids = train_qids + calib_qids + screen_qids

    s1 = pd.read_csv(Path(args.train_dir) / 'train_source1.tsv', sep='\t', dtype=str, keep_default_na=False).set_index('entity_id')

    query_records = {
        qid: {
            'entity_id': qid,
            'business_name': s1.loc[qid, 'business_name'],
            'business_address': s1.loc[qid, 'business_address'],
            'country': s1.loc[qid, 'country'],
        }
        for qid in all_qids
    }

    train_rows, train_labels, train_groups, train_folds = [], [], [], []
    calib_rows, calib_labels, calib_groups, calib_pairs = [], [], [], []
    screen_rows, screen_labels, screen_groups, screen_pairs = [], [], [], []

    for country in ('India', 'US'):
        t_c_start = time.time()
        print(f'Processing {country}...')
        cache_country_file = out_dir / f'l04_features_{country}.joblib'
        if cache_country_file.exists():
            print(f'Loading cached features from {cache_country_file}...')
            c_data = joblib.load(cache_country_file)
            c_results = c_data['results']
        else:
            c_qids = [q for q in all_qids if s1.loc[q, 'country'] == country]
            pool_dict, pool_ids = joblib.load(cache_dir / f'pool_dict_{country}.joblib')
            dupe_map = joblib.load(cache_dir / f'dupe_map_{country}.joblib')
            struct_idx = joblib.load(cache_dir / f'structured_index_{country}.joblib')
            lex_artifacts = {}
            for mode in ('joint', 'name_only', 'address_only'):
                vec = joblib.load(cache_dir / f'vec_{country}_{mode}.joblib')
                p_mat = load_npz(cache_dir / f'mat_{country}_{mode}.npz')
                lex_artifacts[mode] = (vec, p_mat)
            q_names = [normalize_name(s1.loc[q, 'business_name'], country) for q in c_qids]
            q_addrs = [normalize_address(s1.loc[q, 'business_address'], country) for q in c_qids]
            cands_by_q, ch_lookups = generate_natural_candidates(
                query_ids=c_qids, query_names=q_names, query_addrs=q_addrs, country=country,
                pool_ids=pool_ids, lexical_artifacts=lex_artifacts, structured_index=struct_idx,
                dupe_map=dupe_map, k_per_channel={'joint': 100, 'name_only': 100, 'address_only': 150, 'structured': 100},
                top_k_final=100, n_threads=args.n_cores
            )
            needed_pids = {pid for qid in c_qids for pid, _ in cands_by_q.get(qid, [])}
            adapter = NormalizedRecordAdapter()
            q_meta_cache = {qid: adapter.normalize(qid, s1.loc[qid, 'business_name'], s1.loc[qid, 'business_address'], country) for qid in c_qids}
            def _norm_pid(pid: str):
                p_rec = pool_dict[pid]
                return pid, adapter.normalize(pid, p_rec['business_name'], p_rec['business_address'], country)
            with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
                pool_meta_cache = dict(pool.map(_norm_pid, [p for p in needed_pids if p in pool_dict]))

            def _extract_query(qid: str):
                q_meta = q_meta_cache[qid]
                true_tgts = set(gt_map.get(qid, []))
                cands = cands_by_q.get(qid, [])
                top1_rrf = cands[0][1] if len(cands) > 0 else 0.0
                top2_rrf = cands[1][1] if len(cands) > 1 else 0.0
                l_rows, l_lbls, l_grps, l_pairs = [], [], [], []
                for rank, (pid, rrf_score) in enumerate(cands, start=1):
                    if pid not in pool_meta_cache: continue
                    p_meta = pool_meta_cache[pid]
                    chmap = {ch: d[pid] for ch, qdict in ch_lookups.items() if (d := qdict.get(qid)) and pid in d}
                    feat = pair_feature_row_v3(q=q_meta, t=p_meta, chmap=chmap, rrf=rrf_score, rank=rank, top1_rrf=top1_rrf, top2_rrf=top2_rrf, src_is_s2=pid.startswith('S2-'))
                    l_rows.append(feat)
                    l_lbls.append(1 if pid in true_tgts else 0)
                    l_grps.append(qid)
                    l_pairs.append((qid, pid))
                return qid, l_rows, l_lbls, l_grps, l_pairs

            with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
                c_results = list(pool.map(_extract_query, c_qids))
            joblib.dump({'results': c_results}, cache_country_file, compress=3)
            del pool_dict, pool_ids, dupe_map, struct_idx, lex_artifacts, pool_meta_cache, q_meta_cache
            gc.collect()

        for qid, q_r, q_l, q_g, q_p in c_results:
            if qid in train_set:
                train_rows.extend(q_r)
                train_labels.extend(q_l)
                train_groups.extend(q_g)
                fold = inner_folds_map.get(qid, 0)
                train_folds.extend([fold] * len(q_r))
            elif qid in calib_set:
                calib_rows.extend(q_r)
                calib_labels.extend(q_l)
                calib_groups.extend(q_g)
                calib_pairs.extend(q_p)
            elif qid in screen_set:
                screen_rows.extend(q_r)
                screen_labels.extend(q_l)
                screen_groups.extend(q_g)
                screen_pairs.extend(q_p)
        print(f'{country} processed in {time.time() - t_c_start:.2f}s')

    X_train = rows_to_matrix(train_rows)
    y_train = np.array(train_labels, dtype=np.int32)
    folds_train = np.array(train_folds, dtype=np.int32)
    X_calib = rows_to_matrix(calib_rows)
    X_screen = rows_to_matrix(screen_rows)

    params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'num_leaves': 127, 'max_depth': 9, 'min_child_samples': 100, 'learning_rate': 0.05, 'n_jobs': args.n_cores, 'random_state': 42, 'verbose': -1}
    best_iters = []
    for fold in range(3):
        val_mask = (folds_train == fold)
        dtrain = lgb.Dataset(X_train[~val_mask], label=y_train[~val_mask], feature_name=FEATURES_V3)
        dval = lgb.Dataset(X_train[val_mask], label=y_train[val_mask], feature_name=FEATURES_V3, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters.append(bst.best_iteration)
    mean_trees = int(np.round(np.mean(best_iters)))
    print(f'Mean optimal trees: {mean_trees}')

    dfull = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES_V3)
    final_model = lgb.train(params, dfull, num_boost_round=mean_trees)
    final_model.save_model(str(out_dir / 'l04_rich_matcher.txt'))

    cal_df = pd.DataFrame({'query_id': [p[0] for p in calib_pairs], 'target_id': [p[1] for p in calib_pairs], 'is_match': calib_labels, 'probability': final_model.predict(X_calib)})
    scr_df = pd.DataFrame({'query_id': [p[0] for p in screen_pairs], 'target_id': [p[1] for p in screen_pairs], 'is_match': screen_labels, 'probability': final_model.predict(X_screen)})
    cal_df.to_parquet(out_dir / 'calibration_predictions.parquet', index=False)
    scr_df.to_parquet(out_dir / 'screen_predictions.parquet', index=False)

    cal_prep = prepare(cal_df, {q: gt_map.get(q, []) for q in calib_qids}, query_records)
    scr_prep = prepare(scr_df, {q: gt_map.get(q, []) for q in screen_qids}, query_records)

    policies = select_policies(cal_prep)
    (out_dir / 'calibrated_policies.json').write_text(json.dumps(policies, indent=2))
    print('L04 completed successfully in', time.time() - t_start, 's')

if __name__ == '__main__':
    main()