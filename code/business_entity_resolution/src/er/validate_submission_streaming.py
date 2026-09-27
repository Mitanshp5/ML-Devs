"""Validate final TSVs row by row; never retain the full candidate mapping."""
from __future__ import annotations

import argparse
import csv
from itertools import zip_longest
from pathlib import Path

from er.submission_artifacts import atomic_json, parse_result_row, sha256_file


def read_source_ids(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if 'entity_id' not in (reader.fieldnames or []):
            raise ValueError(f'{path}: missing entity_id column')
        return {row['entity_id'] for row in reader}


def validate_files(matching, candidate, test_dir, check_ids=True):
    test_dir = Path(test_dir)
    required = read_source_ids(test_dir / 'test_source1.tsv')
    valid = None
    if check_ids:
        # About 10M unique IDs, rather than 173M per-query candidate entries.
        # Missing target files fail explicitly; no silent skipped ID checks.
        valid = read_source_ids(test_dir / 'test_source2.tsv')
        valid.update(read_source_ids(test_dir / 'test_source3.tsv'))
    seen = set()
    pairs = matches = 0
    with Path(matching).open(encoding='utf-8') as mf, Path(candidate).open(encoding='utf-8') as cf:
        if mf.readline().rstrip('\r\n') != 'source1_entity_id\tmatched_entity_ids':
            raise ValueError('Invalid matching header')
        if cf.readline().rstrip('\r\n') != 'source1_entity_id\tcandidate_entity_ids':
            raise ValueError('Invalid candidate header')
        for number, (ml, cl) in enumerate(zip_longest(mf, cf), 2):
            if ml is None or cl is None:
                raise ValueError('Matching and candidate row counts differ')
            mq, mids = parse_result_row(ml)
            cq, cids = parse_result_row(cl)
            if mq != cq:
                raise ValueError(f'Row {number}: matching/candidate query IDs differ')
            if mq not in required or mq in seen:
                raise ValueError(f'Row {number}: unknown or duplicated query {mq}')
            if not mids <= cids:
                raise ValueError(f'Row {number}: accepted match missing from candidates')
            if valid is not None and not cids <= valid:
                raise ValueError(f'Row {number}: nonexistent target IDs')
            seen.add(mq)
            matches += len(mids)
            pairs += len(cids)
    if seen != required:
        raise ValueError(f'Missing {len(required - seen):,} source1 queries')
    return dict(status='passed', queries=len(seen), candidate_pairs=pairs,
                matches=matches, target_ids_checked=check_ids,
                matching_sha256=sha256_file(matching), candidates_sha256=sha256_file(candidate))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matching', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--report', default='output/final/validation_report.json')
    args = ap.parse_args()
    print('Validating all rows and IDs; retaining only unique source IDs and one candidate row', flush=True)
    report = validate_files(args.matching, args.candidate, args.test_dir)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(f"PASSED: {report['queries']:,} queries, {report['candidate_pairs']:,} candidates, {report['matches']:,} matches", flush=True)


if __name__ == '__main__':
    main()
