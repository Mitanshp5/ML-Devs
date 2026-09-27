"""Atomic, content-verified checkpoints shared by inference and merging."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def text_hash(path):
    # Git may check text out with CRLF on Windows and LF on macOS.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def country_lock(output, country):
    """An OS lock is released even after a crash; an old lock file is harmless."""
    path = Path(output) / f'{country}.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'{country} is already running or being merged in {output}') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def parse_result_row(line):
    fields = line.rstrip('\r\n').split('\t')
    if len(fields) != 2 or not fields[0]:
        raise ValueError('Malformed submission row')
    qid, text = fields
    ids = text.split(',') if text else []
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate target for {qid}')
    if any(not v.startswith(('S2-', 'S3-')) for v in ids):
        raise ValueError(f'Invalid target prefix for {qid}')
    return qid, set(ids)


def validate_rows(matching, candidates, query_ids):
    """Check a pair of headerless shards without retaining candidate sets."""
    pairs = matches = rows = 0
    with Path(matching).open(encoding='utf-8') as mf, Path(candidates).open(encoding='utf-8') as cf:
        for expected, ml, cl in zip_longest(query_ids, mf, cf):
            if expected is None or ml is None or cl is None:
                raise ValueError('Shard row count does not match its query range')
            mq, mids = parse_result_row(ml)
            cq, cids = parse_result_row(cl)
            if mq != expected or cq != expected:
                raise ValueError(f'Shard query order/identity mismatch: expected {expected}')
            if not mids <= cids:
                raise ValueError(f'Matches outside candidate set for {expected}')
            pairs += len(cids)
            matches += len(mids)
            rows += 1
    return dict(rows=rows, pairs=pairs, matches=matches)


def shard_paths(shards, country, start):
    sid = f'{country}_{start:09d}'
    return tuple(Path(shards) / f'{prefix}_{sid}.{ext}' for prefix, ext in
                 [('matching', 'tsv'), ('candidates', 'tsv'), ('complete', 'json')])


def verify_checkpoint(shards, country, start, query_ids, contract_id):
    mp, cp, done = shard_paths(shards, country, start)
    if not (mp.exists() and cp.exists() and done.exists()):
        return False
    meta = json.loads(done.read_text(encoding='utf-8'))
    if meta.get('version') != 2:
        # Old checkpoints have no model or input provenance. Recompute them;
        # inference archives their files before replacement.
        return False
    expected = dict(contract_id=contract_id, country=country, start=start,
                    rows=len(query_ids), query_ids_sha256=json_hash(list(query_ids)))
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'{done.name}: incompatible {key}; use a new output directory')
    for key, path in [('matching_sha256', mp), ('candidates_sha256', cp)]:
        if meta.get(key) != sha256_file(path):
            raise ValueError(f'{path.name}: checksum mismatch; restore or explicitly recompute this shard')
    stats = validate_rows(mp, cp, query_ids)
    if any(meta.get(k) != v for k, v in stats.items()):
        raise ValueError(f'{done.name}: incorrect row/pair counts')
    return True
