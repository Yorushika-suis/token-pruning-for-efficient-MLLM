"""Executable constants and legal work keys for approved v4-A-r4."""
import hashlib
import struct

PLAN_SHA256 = '3e5d06dbcac18cc8e5575eacf5b58aea144c9e915fdfffe4cd2cfc07c7c95a73'
VERSION = 'v4-A-r4-implementation-1'
FRAMES, SIDE, ACTIVE, GRID, NV = 64, 384, 364, 13, 169
PROTOCOLS = ('controlled-gray', 'unblanked-native')
ARMS = ('identity', 'hflip_active364', 'vflip_active364', 'shift_active364_wrap_r3c5')
BUDGETS = {'R010': 16, 'R015': 23, 'R020': 31, 'R025': 38}
INTERNAL = ('adts_v2', 'adts_v1', 'attn', 'FlashVID-div-ablation@P1U169')
CONTROLS = ('random_uniform_without_replacement', 'grid_uniform',
            'spatial_stratified_legacy_rule', 'translated_fixed_grid_torus')
METHODS = INTERNAL + CONTROLS
RANDOM = (CONTROLS[0], CONTROLS[2], CONTROLS[3])
SEEDS = tuple(range(5))
PURPOSE_ARMS = (('protocol_pair', 'identity'),) + tuple(('geometry_arm', a) for a in ARMS)
DOMAIN = 'flashvid-adts/v4-stream-1'
METHOD_VERSIONS = {m: m + '/v4-1' for m in METHODS}
HOTSPOTS = (131, 11, 157, 26, 156, 1, 144, 0, 130, 14)


def seed_set(method):
    if method not in METHODS:
        raise ValueError('unregistered method')
    return SEEDS if method in RANDOM else ('deterministic',)


def material(seed, video_id, algorithm_version, purpose, budget, arm):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('run_seed must be an integer in 0..4')
    if algorithm_version not in METHOD_VERSIONS.values() or budget not in BUDGETS or (purpose, arm) not in PURPOSE_ARMS:
        raise ValueError('unregistered stream field')
    if not isinstance(video_id, str) or not video_id or '/' in video_id or '\\' in video_id or video_id in ('.', '..'):
        raise ValueError('unsafe video ID')
    fields = [DOMAIN.encode(), struct.pack('>Q', seed), video_id.encode('utf-8')]
    fields += [s.encode('ascii') for s in (algorithm_version, purpose, budget, arm)]
    return b''.join(struct.pack('>I', len(s)) + s for s in fields)


def stream(seed, video_id, method, purpose, budget, arm):
    b = material(seed, video_id, METHOD_VERSIONS[method], purpose, budget, arm)
    value = int.from_bytes(hashlib.blake2b(b, digest_size=16).digest(), 'big')
    return {'material_hex': b.hex(), 'seed128': value, 'algorithm_version': METHOD_VERSIONS[method], 'domain': DOMAIN}


def feature_keys(ids):
    return [[v, p, a] for v in ids for p in PROTOCOLS for a in ARMS]


def mask_keys(ids, methods=METHODS):
    return [[v, p, a, m, b, s, purpose] for v in ids for p in PROTOCOLS
            for purpose, a in PURPOSE_ARMS for m in methods for b in BUDGETS for s in seed_set(m)]


def check_streams(ids):
    """The same key shared by protocols is intentional; distinct keys cannot collide."""
    seen = {}
    for v in ids:
        for m in RANDOM:
            for b in BUDGETS:
                for s in SEEDS:
                    for purpose, arm in PURPOSE_ARMS:
                        rec = stream(s, v, m, purpose, b, arm)
                        old = seen.setdefault(rec['seed128'], rec['material_hex'])
                        if old != rec['material_hex']:
                            raise ValueError('distinct stream keys collide; no ad hoc salting allowed')
    return {'distinct_keys': len(seen), 'seed128_unique': len(seen)}
