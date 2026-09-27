"""Small parity fixtures and failure cases; no full dataset or inference job."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from er.candidate_generation import generate_natural_candidates
from er.features import pair_feature_row_v4, rows_to_matrix
from er.normalized_adapter import NormalizedRecordAdapter
from er.normalization import normalize_name, normalize_address
from er.retrieval.lexical import build_vectorizer, channel_texts
from er.retrieval.structured import build_index
from er.run_submission_batched import load_bundle, merge_shards, prepare_matrix, run_country, score_rows
from er.submission_artifacts import (atomic_json, country_lock, json_hash, sha256_file,
                                    shard_paths, validate_rows, verify_checkpoint)
from er.validate_submission_streaming import validate_files


ROOT = Path(__file__).resolve().parents[3]


def test_prepared_retrieval_and_block_scoring_preserve_predictions():
    country = 'India'
    ids = ['S2-1', 'S2-2', 'S3-3', 'S2-4', 'S3-5']
    names = ['Alpha pvt ltd', 'Alpha pvt ltd', 'अल्फा', 'Beta', '']
    addrs = ['12 Main Street 400001', '13 Main Street 400001', '12 Main Street 400001', '', '']
    pool = {pid: dict(business_name=n, business_address=a) for pid, n, a in zip(ids, names, addrs)}
    pn = [normalize_name(n, country) for n in names]
    pa = [normalize_address(a, country) for a in addrs]
    lex = {}
    for mode in ['joint', 'name_only', 'address_only']:
        vec = build_vectorizer(min_df=1, max_df=1.0)
        lex[mode] = vec, vec.fit_transform(channel_texts(pn, pa, mode))
    qids = ['S1-a', 'S1-b', 'S1-c']
    qn = ['Alpha pvt ltd', 'Beta', '']
    qa = ['12 Main Street 400001', '', '']
    struct, _ = build_index(pn, pa, stop=set())
    params = dict(query_ids=qids, query_names=[normalize_name(n, country) for n in qn],
                  query_addrs=[normalize_address(a, country) for a in qa], country=country,
                  pool_ids=ids, structured_index=struct, dupe_map={}, n_threads=1)
    reference = generate_natural_candidates(**params, lexical_artifacts=lex)
    prepared = {k: (vec, mat.tocsc()) for k, (vec, mat) in lex.items()}
    actual = generate_natural_candidates(**params, lexical_artifacts=prepared)
    assert actual == reference  # includes every score, tie rank, candidate ID and channel
    cands, lookups = actual
    models, weights, policy, builder, _ = load_bundle(ROOT / 'production_bundle_final')
    # Frozen model is unchanged: its recorded 394-tree claim was metadata only.
    assert next(iter(models.values())).num_trees() == 400
    # Independent original all-rows-at-once scoring reference.
    adapter = NormalizedRecordAdapter()
    rows, pairs = [], []
    for qid, name, addr in zip(qids, qn, qa):
        qm = adapter.normalize(qid, name, addr, country)
        candidates = cands[qid]
        first = candidates[0][1] if candidates else 0
        second = candidates[1][1] if len(candidates) > 1 else 0
        for rank, (pid, rrf) in enumerate(candidates, 1):
            rec = pool[pid]
            tm = adapter.normalize(pid, rec['business_name'], rec['business_address'], country)
            channels = {ch: qmap[qid][pid] for ch, qmap in lookups.items() if pid in qmap.get(qid, {})}
            rows.append(pair_feature_row_v4(qm, tm, channels, rrf, rank, first, second, pid.startswith('S2-')))
            pairs.append((qid, pid))
    matrix = rows_to_matrix(rows)
    probabilities = sum(weights[n] * m.predict(matrix, num_threads=1) for n, m in models.items())
    expected = []
    for qid in qids:
        scored = [(pid, p) for (q, pid), p in zip(pairs, probabilities) if q == qid]
        thresholds = policy.get(country, policy['global'])
        accepted = [pid for pid, p in scored if p >= thresholds['tm']] if max((p for _, p in scored), default=0) >= thresholds['ts'] else []
        expected.append((f"{qid}\t{','.join(accepted)}\n", f"{qid}\t{','.join(pid for pid, _ in scored)}\n"))
    for block_size in (1, 2, 64):
        assert list(score_rows(qids, qn, qa, country, cands, lookups, pool, models, weights,
                               policy, builder, 2, block_size)) == expected


def create_checkpoint(shards, country, start, ids, contract):
    mp, cp, done = shard_paths(shards, country, start)
    mp.write_text(''.join(f'{q}\tS2-1\n' for q in ids), encoding='utf-8')
    cp.write_text(''.join(f'{q}\tS2-1,S3-2\n' for q in ids), encoding='utf-8')
    atomic_json(done, dict(version=2, contract_id=json_hash(contract), country=country, start=start,
                query_ids_sha256=json_hash(ids), matching_sha256=sha256_file(mp),
                candidates_sha256=sha256_file(cp), **validate_rows(mp, cp, ids)))
    return mp, cp, done


def test_resume_rejects_corruption_and_changed_model(tmp_path):
    ids, contract = ['S1-1', 'S1-2'], {'model': 'frozen'}
    mp, cp, done = create_checkpoint(tmp_path, 'India', 0, ids, contract)
    assert verify_checkpoint(tmp_path, 'India', 0, ids, json_hash(contract))
    with pytest.raises(ValueError, match='contract_id'):
        verify_checkpoint(tmp_path, 'India', 0, ids, json_hash({'model': 'other'}))
    cp.write_text('S1-1\tS2-1\n')
    with pytest.raises(ValueError, match='checksum'):
        verify_checkpoint(tmp_path, 'India', 0, ids, json_hash(contract))
    done.write_text('{"country":"India","start":0,"rows":2}')
    assert not verify_checkpoint(tmp_path, 'India', 0, ids, json_hash(contract))


def test_merge_variable_country_batches_and_missing_shard(tmp_path):
    shards = tmp_path / 'shards'
    shards.mkdir()
    contracts = {'India': {'country': 'India'}, 'France': {'country': 'France'}}
    frame = pd.DataFrame({'entity_id': ['S1-f', 'S1-i1', 'S1-i2', 'S1-i3'],
                          'country': ['France', 'India', 'India', 'India']})
    create_checkpoint(shards, 'France', 0, ['S1-f'], contracts['France'])
    create_checkpoint(shards, 'India', 0, ['S1-i1', 'S1-i2'], contracts['India'])
    dest = tmp_path / 'matching_results.tsv'
    dest.write_text('preserve-existing')
    with pytest.raises(ValueError, match='only 2/3'):
        merge_shards(tmp_path, frame, contracts)
    assert dest.read_text() == 'preserve-existing'
    create_checkpoint(shards, 'India', 2, ['S1-i3'], contracts['India'])
    merge_shards(tmp_path, frame, contracts)
    assert len(dest.read_text().splitlines()) == 5
    # Overlap must fail before clobbering the complete result.
    create_checkpoint(shards, 'India', 1, ['S1-i2'], contracts['India'])
    with pytest.raises(ValueError, match='overlapping'):
        merge_shards(tmp_path, frame, contracts)


def test_country_lock_excludes_second_writer(tmp_path):
    with country_lock(tmp_path, 'India'):
        with pytest.raises(RuntimeError, match='already running'):
            with country_lock(tmp_path, 'India'):
                pass
    with country_lock(tmp_path, 'India'):
        pass  # lock is released, not permanently held by an old file


def test_mapped_matrix_equal_to_ram_and_rejects_corruption(tmp_path):
    from scipy.sparse import csr_matrix, save_npz
    source = tmp_path / 'fixture.npz'
    original = csr_matrix(np.array([[0, 2, 1], [1, 0, 0]], dtype=np.float32))
    save_npz(source, original)
    digest = sha256_file(source)
    mapped = prepare_matrix(source, 'mmap', digest)
    np.testing.assert_array_equal(mapped.toarray(), original.toarray())
    assert mapped.T.format == 'csr'
    np.testing.assert_array_equal(prepare_matrix(source, 'mmap', digest).toarray(), mapped.toarray())
    del mapped
    data = next((tmp_path / 'prepared_csc').glob('*/data.npy'))
    # Release memory mappings before modifying their backing file (Windows).
    import gc
    gc.collect()
    data.write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        prepare_matrix(source, 'mmap', digest)


def test_streaming_validation_rejects_wrong_ids_and_missing_matches(tmp_path):
    for src, ids in [('1', ['S1-a', 'S1-b']), ('2', ['S2-1']), ('3', ['S3-2'])]:
        (tmp_path / f'test_source{src}.tsv').write_text('entity_id\n' + '\n'.join(ids) + '\n')
    mp, cp = tmp_path / 'matches.tsv', tmp_path / 'candidates.tsv'
    mp.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-1\nS1-b\t\n')
    cp.write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-1,S3-2\nS1-b\t\n')
    assert validate_files(mp, cp, tmp_path)['candidate_pairs'] == 2
    cp.write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS3-2\nS1-b\t\n')
    with pytest.raises(ValueError, match='missing from candidates'):
        validate_files(mp, cp, tmp_path)
    cp.write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-1,S3-missing\nS1-b\t\n')
    with pytest.raises(ValueError, match='nonexistent'):
        validate_files(mp, cp, tmp_path)


def test_tiny_country_archives_legacy_and_resumes_without_loading_indexes(tmp_path, monkeypatch):
    import argparse
    import joblib
    from scipy.sparse import save_npz
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    cache.mkdir()
    (output / 'shards').mkdir(parents=True)
    names, addrs = ['Alpha', 'Beta'], ['1 Main St 400001', '2 Main St 400001']
    ids = ['S2-1', 'S3-2']
    pool = {pid: dict(business_name=n, business_address=a) for pid, n, a in zip(ids, names, addrs)}
    joblib.dump((pool, ids), cache / 'pool_dict_India.joblib')
    joblib.dump({}, cache / 'dupe_map_India.joblib')
    pn = [normalize_name(n, 'India') for n in names]
    pa = [normalize_address(a, 'India') for a in addrs]
    struct, _ = build_index(pn, pa, stop=set())
    joblib.dump(struct, cache / 'structured_index_India.joblib')
    for mode in ('joint', 'name_only', 'address_only'):
        vec = build_vectorizer(min_df=1, max_df=1.0)
        mat = vec.fit_transform(channel_texts(pn, pa, mode))
        joblib.dump(vec, cache / f'vec_India_{mode}.joblib')
        save_npz(cache / f'mat_India_{mode}.npz', mat)
    contract = {'cache_artifacts': {p.name: sha256_file(p) for p in cache.iterdir()}, 'model': 'fixture'}
    frame = pd.DataFrame(dict(entity_id=['S1-a', 'S1-b'], business_name=names, business_address=addrs))
    models, weights, policy, builder, _ = load_bundle(ROOT / 'production_bundle_final')
    mp, cp, done = shard_paths(output / 'shards', 'India', 0)
    mp.write_text('old matching')
    cp.write_text('old candidates')
    done.write_text(json.dumps(dict(country='India', start=0, rows=2)))
    args = argparse.Namespace(output_dir=output, cache_dir=cache, batch_size=2, resume=True,
                              matrix_storage='ram', n_cores=1, feature_batch_size=1)
    run_country(args, 'India', frame, contract, models, weights, policy, builder)
    assert verify_checkpoint(output / 'shards', 'India', 0, ['S1-a', 'S1-b'], json_hash(contract))
    archived = list((output / 'shards' / 'legacy').glob('*/matching_*.tsv'))
    assert len(archived) == 1 and archived[0].read_text() == 'old matching'
    def unexpected_load(*args, **kwargs):
        raise AssertionError('Completed-country resume must not load indexes')
    monkeypatch.setattr(joblib, 'load', unexpected_load)
    run_country(args, 'India', frame, contract, models, weights, policy, builder)
