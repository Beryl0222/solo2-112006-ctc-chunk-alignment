"""随机差分压力测试：算子 vs 穷举参考（不作为 unittest 的一部分，直接运行）。"""

import math
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from ctc_operator import NEG, StreamCtc
from test_ctc_operator import brute_preview, brute_train


def brute_preview_constrained(width, labels, frames, confirmations):
    """带确认约束的穷举预览：实例区间须与承诺一致。"""
    total = len(frames)
    from test_ctc_operator import collapse, instances_of, log_softmax
    lsm = [log_softmax(row) for row in frames]
    best = None
    from itertools import product
    for path in product(range(width), repeat=total):
        col = collapse(path)
        if col != list(labels[: len(col)]):
            continue
        inst = instances_of(path)
        if any(i >= len(inst) or (inst[i][1], inst[i][2]) != confirmations[i]
               for i in range(len(confirmations))):
            continue
        score = 0.0
        for t in range(total):
            v = lsm[t][path[t]]
            if v == NEG:
                score = NEG
                break
            score += v
        if score == NEG:
            continue
        key = (len(col), score, tuple(-c for c in path))
        if best is None or key > best[0]:
            best = (key, path, score)
    if best is None:
        return None
    return best[1], len(collapse(best[1])), best[2]


def random_case(rng):
    width = rng.randint(2, 4)
    L = rng.randint(1, 4)
    labels = [rng.randint(1, width - 1) for _ in range(L)]
    total = rng.randint(L, 9)
    frames = []
    for _ in range(total):
        row = []
        finite = 0
        for _ in range(width):
            if rng.random() < 0.25:
                row.append(None)
            else:
                row.append(round(rng.uniform(-4, 4), 2))
                finite += 1
        if finite == 0:
            row[rng.randrange(width)] = round(rng.uniform(-4, 4), 2)
        frames.append(row)
    return width, labels, total, frames


def run_case(rng, case_id):
    width, labels, total, frames = random_case(rng)
    op = StreamCtc([f"s{i}" for i in range(width)], labels, workspace_floats=500)
    # 随机分块投递
    start = 0
    cid = 0
    while start < total:
        n = rng.randint(1, total - start)
        op.chunk(f"c{cid}", 0, start, frames[start:start + n])
        start += n
        cid += 1
    full = [[NEG if x is None else float(x) for x in row] for row in frames]

    # 预览对拍（可能多轮：确认后再预览）
    confirmations = []
    for round_ in range(len(labels) + 1):
        pv = op.preview()
        want = brute_preview_constrained(width, labels, full[: pv.available], confirmations)
        if want is None:
            assert pv.covered < pv.available, (case_id, "expected dead free region")
        else:
            want_path, want_emitted, want_score = want
            assert pv.path == want_path, (case_id, "preview path", pv.path, want_path)
            assert pv.emitted == want_emitted
            assert math.isclose(pv.score, want_score, rel_tol=1e-9, abs_tol=1e-9), (
                case_id, "preview score", pv.score, want_score)
        for i, (b, e) in enumerate(confirmations):
            inst = pv.instances[i]
            assert (inst.begin, inst.end) == (b, e), (case_id, "confirmed moved", i)
        if round_ < len(labels) and len(pv.instances) > len(confirmations):
            inst = pv.instances[len(confirmations)]
            op.confirm(inst.index, inst.begin, inst.end)
            confirmations.append((inst.begin, inst.end))
        else:
            break

    op.finish(total)
    train = op.train("t")
    want = brute_train(width, labels, full, confirmations)
    if want is None:
        assert not train.ok, (case_id, "should be untrainable")
        assert train.reason == "no-legal-path"
        return
    assert train.ok, (case_id, "should be trainable", confirmations)
    want_nll, want_grads = want
    assert math.isclose(train.nll, want_nll, rel_tol=1e-9, abs_tol=1e-9), (
        case_id, "nll", train.nll, want_nll)
    back = op.backward("t")
    flat = [row for g in back.gradients for row in g.rows]
    assert len(flat) == total
    for t, (got, want_row) in enumerate(zip(flat, want_grads)):
        for c, (g, w) in enumerate(zip(got, want_row)):
            assert math.isclose(g, w, rel_tol=1e-7, abs_tol=1e-7), (
                case_id, "grad", t, c, g, w)


def main():
    rng = random.Random(20261120)
    n = 300
    for case_id in range(n):
        run_case(rng, case_id)
    print(f"{n} 个随机用例全部通过")


if __name__ == "__main__":
    main()
