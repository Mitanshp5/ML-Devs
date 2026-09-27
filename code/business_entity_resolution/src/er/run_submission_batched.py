"""Bounded feature blocks, once-per-country sparse conversion, verified resume."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time
import unicodedata

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, load_npz

from er.candidate_generation import generate_natural_candidates
from er.features import FEATURES_V3, FEATURES_V4, pair_feature_row_v3, pair_feature_row_v4, rows_to_matrix
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
from er.submission_artifacts import (atomic_json, country_lock, json_hash, sha256_file,
                                    shard_paths, text_hash, validate_rows, verify_checkpoint)


def load_bundle(bundle):
    config = json.loads((bundle / 'decision_policy.json').read_text(encoding='utf-8'))
    schema = json.loads((bundle / 'feature_schema.json').read_text(encoding='utf-8'))
    builders = {tuple(FEATURES_V3): pair_feature_row_v3, tuple(FEATURES_V4): pair_feature_row_v4}
    builder = builders[tuple(schema['features'])]
    weights = config['ensemble_weights']
    if any(w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError('Model weights must be nonnegative and sum to one')
    models = {name: lgb.Booster(model_file=str(bundle / name)) for name in weights}
    for name, model in models.items():
        if model.feature_name() != schema['features']:
            raise ValueError(f'{name}: model feature names/order differ from the declared schema')
    identity = dict(policy=config, schema=schema,
                    models={name: text_hash(bundle / name) for name in models})
    return models, weights, config[config['selected_policy']], builder, identity


def make_contract(identity, input_sha, cache, country, limit):
    manifest = json.loads((cache / f'manifest_{country}.json').read_text(encoding='utf-8'))
    source = Path(__file__).parent
    semantic_files = ['features.py', 'normalization.py', 'normalized_adapter.py',
                      'candidate_generation.py', 'retrieval/lexical.py', 'retrieval/structured.py']
    return dict(version=2, country=country, source1_sha256=input_sha, limit=limit,
                unicode_version=unicodedata.unidata_version,
                bundle=identity, cache_artifacts=manifest['artifacts_sha256'],
                feature_code={p: text_hash(source / p) for p in semantic_files},
                packages={p: version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'lightgbm',
                                                  'sparse-dot-topn', 'rapidfuzz', 'joblib']},
                retrieval=dict(joint=100, name_only=100, address_only=150, structured=100, final=100))


def prepare_matrix(path, storage, expected_sha):
    """Keep the query-transpose CSR view ready; optionally back arrays with disk."""
    if storage == 'ram':
        original = load_npz(path)
        return original.tocsc()
    folder = path.parent / 'prepared_csc' / (path.stem + '_' + expected_sha[:16])
    folder.mkdir(parents=True, exist_ok=True)
    with country_lock(folder, 'prepare'):
        ready = folder / 'ready.json'
        if not ready.exists():
            original = load_npz(path)
            matrix = original.tocsc()
            del original
            arrays = {}
            for name in ('data', 'indices', 'indptr'):
                dest = folder / f'{name}.npy'
                temp = dest.with_suffix('.partial')
                with temp.open('wb') as stream:
                    np.save(stream, getattr(matrix, name), allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, dest)
                arrays[name] = sha256_file(dest)
            atomic_json(ready, dict(source_sha256=expected_sha, shape=list(matrix.shape), arrays=arrays))
            del matrix
            gc.collect()
        meta = json.loads(ready.read_text(encoding='utf-8'))
        if meta['source_sha256'] != expected_sha:
            raise ValueError('Prepared matrix source hash mismatch')
        for name, expected in meta['arrays'].items():
            if sha256_file(folder / f'{name}.npy') != expected:
                raise ValueError(f'Prepared matrix checksum mismatch: {folder}/{name}.npy')
        arrays = [np.load(folder / f'{n}.npy', mmap_mode='r', allow_pickle=False)
                  for n in ('data', 'indices', 'indptr')]
        matrix = csc_matrix(tuple(arrays), shape=tuple(meta['shape']), copy=False)
        if any(not np.shares_memory(getattr(matrix, name), arr)
               for name, arr in zip(('data', 'indices', 'indptr'), arrays)):
            raise RuntimeError('SciPy copied a mapped sparse array; use --matrix-storage ram')
        return matrix


def score_rows(qids, names, addrs, country, cands, lookups, pool_dict,
               models, weights, policy, builder, n_cores, feature_batch_size):
    """Yield final TSV rows; only one feature block's normalized records survive."""
    params = policy.get(country, policy['global'])
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        for start in range(0, len(qids), feature_batch_size):
            adapter = NormalizedRecordAdapter()

            def extract(item):
                qid, name, addr = item
                qm = adapter.normalize(qid, name, addr, country)
                candidates = cands.get(qid, [])
                first = candidates[0][1] if candidates else 0.0
                second = candidates[1][1] if len(candidates) > 1 else 0.0
                rows = []
                for rank, (pid, rrf) in enumerate(candidates, 1):
                    rec = pool_dict[pid]
                    tm = adapter.normalize(pid, rec['business_name'], rec['business_address'], country)
                    channels = {ch: qmap[qid][pid] for ch, qmap in lookups.items()
                                if qid in qmap and pid in qmap[qid]}
                    rows.append(builder(qm, tm, channels, rrf, rank, first, second, pid.startswith('S2-')))
                return rows

            stop = start + feature_batch_size
            block_ids = qids[start:stop]
            blocks = list(executor.map(extract, zip(block_ids, names[start:stop], addrs[start:stop])))
            matrix = rows_to_matrix([row for block in blocks for row in block])
            adapter.clear()
            del blocks
            prob = np.zeros(len(matrix), dtype=np.float64)
            if len(matrix):
                for name, weight in weights.items():
                    prob += weight * models[name].predict(matrix, num_threads=n_cores)
                if not np.isfinite(prob).all():
                    raise ValueError('Non-finite prediction')
            offset = 0
            for qid in block_ids:
                ids = [pid for pid, _ in cands.get(qid, [])]
                scores = prob[offset:offset + len(ids)]
                offset += len(ids)
                matched = [pid for pid, p in zip(ids, scores) if p >= params['tm']] if len(scores) and scores.max() >= params['ts'] else []
                yield f"{qid}\t{','.join(matched)}\n", f"{qid}\t{','.join(ids)}\n"
            del matrix, prob, adapter


def run_country(args, country, frame, contract, models, weights, policy, builder):
    output, cache = Path(args.output_dir), Path(args.cache_dir)
    shards = output / 'shards'
    identity = json_hash(contract)
    all_ids = frame.entity_id.tolist()
    for path in shards.glob(f'complete_{country}_*.json'):
        meta = json.loads(path.read_text(encoding='utf-8'))
        start = meta['start']
        if start < 0 or start >= len(frame) or start % args.batch_size or meta['rows'] != min(args.batch_size, len(frame) - start):
            raise ValueError(f'{path.name}: existing shard layout differs; keep this country\'s original batch size or use a new output directory')
    pending = []
    for start in range(0, len(frame), args.batch_size):
        ids = all_ids[start:start + args.batch_size]
        if args.resume and verify_checkpoint(shards, country, start, ids, identity):
            print(f'Verified resume: {country}_{start:09d}', flush=True)
        else:
            pending.append(start)
    if not pending:
        print(f'{country}: all {len(frame):,} queries already verified', flush=True)
        return
    atomic_json(output / f'contract_{country}.json', contract)
    print(f'{country}: verifying cache hashes before loading', flush=True)
    for filename, expected in contract['cache_artifacts'].items():
        if sha256_file(cache / filename) != expected:
            raise ValueError(f'Cache checksum mismatch: {filename}')
    print(f'Loading {country}: {len(frame):,} queries, {len(pending)} pending batches', flush=True)
    # Convert one channel at a time BEFORE loading the large record dictionaries.
    # CSC pool matrices transpose to CSR views, avoiding an O(index-size) copy per batch.
    lex = {}
    for mode in ('joint', 'name_only', 'address_only'):
        filename = f'mat_{country}_{mode}.npz'
        prepared = prepare_matrix(cache / filename, args.matrix_storage, contract['cache_artifacts'][filename])
        lex[mode] = (joblib.load(cache / f'vec_{country}_{mode}.joblib'), prepared)
        del prepared
        gc.collect()
        print(f'Prepared {country}/{mode} once', flush=True)
    pool_dict, pool_ids = joblib.load(cache / f'pool_dict_{country}.joblib')
    dupe_map = joblib.load(cache / f'dupe_map_{country}.joblib')
    struct_idx = joblib.load(cache / f'structured_index_{country}.joblib')
    for _, mat in lex.values():
        if mat.shape[0] != len(pool_ids):
            raise ValueError('Retrieval matrix and pool IDs differ in length')
    started = time.perf_counter()
    timings = []
    for index, start in enumerate(pending):
        t0 = time.perf_counter()
        part = frame.iloc[start:start + args.batch_size]
        qids, names, addrs = (part[col].tolist() for col in ['entity_id', 'business_name', 'business_address'])
        cands, lookups = generate_natural_candidates(
            query_ids=qids, query_names=[normalize_name(n, country) for n in names],
            query_addrs=[normalize_address(a, country) for a in addrs], country=country,
            pool_ids=pool_ids, lexical_artifacts=lex, structured_index=struct_idx,
            dupe_map=dupe_map, k_per_channel={'joint': 100, 'name_only': 100, 'address_only': 150, 'structured': 100},
            top_k_final=100, n_threads=args.n_cores)
        retrieval_s = time.perf_counter() - t0
        mp, cp, done = shard_paths(shards, country, start)
        # Preserve unverifiable older work, including an interrupted checkpoint.
        existing = [p for p in (mp, cp, done) if p.exists()]
        if existing:
            archive = shards / 'legacy' / str(time.time_ns())
            archive.mkdir(parents=True)
            for p in existing:
                shutil.copy2(p, archive / p.name)
            done.unlink(missing_ok=True)
        mt, ct = mp.with_suffix('.tsv.partial'), cp.with_suffix('.tsv.partial')
        with mt.open('w', encoding='utf-8', newline='\n') as mf, ct.open('w', encoding='utf-8', newline='\n') as cf:
            for ml, cl in score_rows(qids, names, addrs, country, cands, lookups, pool_dict,
                                    models, weights, policy, builder, args.n_cores, args.feature_batch_size):
                mf.write(ml)
                cf.write(cl)
            for stream in (mf, cf):
                stream.flush()
                os.fsync(stream.fileno())
        stats = validate_rows(mt, ct, qids)
        os.replace(mt, mp)
        os.replace(ct, cp)
        elapsed = time.perf_counter() - t0
        atomic_json(done, dict(version=2, contract_id=identity, country=country, start=start,
                               query_ids_sha256=json_hash(qids), matching_sha256=sha256_file(mp),
                               candidates_sha256=sha256_file(cp), retrieval_s=retrieval_s,
                               batch_seconds=elapsed, **stats))
        timings.append(elapsed)
        remaining_h = (len(pending) - index - 1) * float(np.mean(timings[-10:])) / 3600
        print(f'Completed {country}_{start:09d}: {len(qids):,} queries, {stats["pairs"]:,} pairs; '
              f'{elapsed:.1f}s (retrieval {retrieval_s:.1f}s); provisional ETA {remaining_h:.2f}h', flush=True)
        del cands, lookups
        gc.collect()
    atomic_json(output / f'inference_manifest_{country}.json', dict(country=country, queries=len(frame),
                contract_id=identity, batch_size=args.batch_size, batches_this_run=len(pending),
                batch_seconds=timings, seconds=time.perf_counter() - started))
    print(f'{country} complete', flush=True)


def merge_shards(output, frame, contracts):
    shards = output / 'shards'
    ordered = []
    with ExitStack() as stack:
        for country, part in frame.groupby('country', sort=True):
            stack.enter_context(country_lock(output, country))
            qids = part.entity_id.tolist()
            metadata = sorted((json.loads(p.read_text(encoding='utf-8')) for p in
                               shards.glob(f'complete_{country}_*.json')), key=lambda m: m['start'])
            offset = 0
            for meta in metadata:
                count = meta['rows']
                if meta['start'] != offset or count <= 0 or offset + count > len(qids):
                    raise ValueError(f'{country}: missing, overlapping or out-of-range shard at {offset}')
                if not verify_checkpoint(shards, country, offset, qids[offset:offset + count],
                                         json_hash(contracts[country])):
                    raise ValueError(f'{country}/{offset}: missing or legacy checkpoint; rerun its country launcher')
                ordered.append(shard_paths(shards, country, offset))
                offset += count
            if offset != len(qids):
                raise ValueError(f'{country}: only {offset:,}/{len(qids):,} rows complete')
        # Preflight completes before touching existing merged outputs.
        mt, ct = output / 'matching_results.tsv.partial', output / 'candidate_pairs.tsv.partial'
        with mt.open('wb') as mf, ct.open('wb') as cf:
            mf.write(b'source1_entity_id\tmatched_entity_ids\n')
            cf.write(b'source1_entity_id\tcandidate_entity_ids\n')
            for mp, cp, _ in ordered:
                with mp.open('rb') as src:
                    shutil.copyfileobj(src, mf, 1024 * 1024)
                with cp.open('rb') as src:
                    shutil.copyfileobj(src, cf, 1024 * 1024)
            for stream in (mf, cf):
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(mt, output / 'matching_results.tsv')
        os.replace(ct, output / 'candidate_pairs.tsv')
        atomic_json(output / 'merge_manifest.json', dict(queries=len(frame), shards=len(ordered),
                    contracts={c: json_hash(v) for c, v in contracts.items()},
                    matching_sha256=sha256_file(output / 'matching_results.tsv'),
                    candidates_sha256=sha256_file(output / 'candidate_pairs.tsv')))
    print(f'Merged {len(frame):,} verified queries', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle-dir', default='production_bundle_final')
    ap.add_argument('--cache-dir', default='cache/retrieval_test')
    ap.add_argument('--dataset-dir', default='student_resource/student_resource/dataset')
    ap.add_argument('--output-dir', default='output/final')
    ap.add_argument('--batch-size', type=int, default=2000)
    ap.add_argument('--feature-batch-size', type=int, default=64)
    ap.add_argument('--matrix-storage', choices=['ram', 'mmap'], default='ram',
                    help='mmap adds local prepared arrays on disk so the OS can reclaim matrix pages')
    ap.add_argument('--n-cores', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--country')
    ap.add_argument('--merge-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if min(args.batch_size, args.feature_batch_size, args.n_cores) < 1 or args.n_cores > 12:
        ap.error('Positive batch sizes and between 1 and 12 CPU threads are required')
    if args.limit < 0:
        ap.error('--limit must be nonnegative')
    output = Path(args.output_dir)
    (output / 'shards').mkdir(parents=True, exist_ok=True)
    models, weights, policy, builder, identity = load_bundle(Path(args.bundle_dir))
    source = Path(args.dataset_dir) / 'test/test_source1.tsv'
    frame = pd.read_csv(source, sep='\t', dtype=str, keep_default_na=False)
    if not frame.entity_id.is_unique:
        raise ValueError('Duplicate query IDs in source1')
    if args.limit:
        per = max(1, args.limit // frame.country.nunique())
        frame = frame.groupby('country', sort=True, group_keys=False).head(per).head(args.limit)
    if args.country:
        frame = frame[frame.country == args.country]
    if frame.empty:
        raise ValueError('No queries selected')
    input_sha = sha256_file(source)
    contracts = {c: make_contract(identity, input_sha, Path(args.cache_dir), c, args.limit)
                 for c in sorted(frame.country.unique())}
    if args.merge_only:
        merge_shards(output, frame, contracts)
        return
    for country, part in frame.groupby('country', sort=True):
        with country_lock(output, country):
            run_country(args, country, part, contracts[country], models, weights, policy, builder)
        gc.collect()
    if not args.country:
        merge_shards(output, frame, contracts)


if __name__ == '__main__':
    main()
