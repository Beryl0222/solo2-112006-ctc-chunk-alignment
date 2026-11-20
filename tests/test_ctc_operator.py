"""流式 CTC 算子的测试：以穷举全部路径的参考实现对照协议数值口径。"""

from itertools import product
from pathlib import Path
import json
import math
import unittest

from ctc_operator import (
    NEG,
    ProtocolError,
    StreamCtc,
    WorkspaceError,
)
from replay_capture import replay

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


def log_softmax(row):
    m = max(row)
    z = m + math.log(sum(math.exp(x - m) for x in row if x != NEG))
    return [x - z if x != NEG else NEG for x in row]


def collapse(path):
    merged = []
    for c in path:
        if not merged or merged[-1] != c:
            merged.append(c)
    return [c for c in merged if c != 0]


def instances_of(path):
    out = []
    i = 0
    while i < len(path):
        c = path[i]
        if c == 0:
            i += 1
            continue
        j = i + 1
        while j < len(path) and path[j] == c:
            j += 1
        out.append((c, i, j))
        i = j
    return out


def satisfies(path, labels, confirmations):
    """路径折叠等于目标，且前 k 个实例区间与确认完全一致。"""
    if collapse(path) != list(labels):
        return False
    inst = instances_of(path)
    for i, (begin, end) in enumerate(confirmations):
        if i >= len(inst) or (inst[i][1], inst[i][2]) != (begin, end):
            return False
    return True


def brute_train(width, labels, frames, confirmations):
    """穷举约束 CTC：返回 (nll, 每帧对 logits 的梯度)；无合法路径返回 None。"""
    total = len(frames)
    lsm = [log_softmax(row) for row in frames]
    z = 0.0
    post = [[0.0] * width for _ in range(total)]
    for path in product(range(width), repeat=total):
        if not satisfies(path, labels, confirmations):
            continue
        w = 1.0
        for t in range(total):
            v = lsm[t][path[t]]
            if v == NEG:
                w = 0.0
                break
            w *= math.exp(v)
        if w == 0.0:
            continue
        z += w
        for t in range(total):
            post[t][path[t]] += w
    if z == 0.0:
        return None
    grads = []
    for t in range(total):
        sm = [math.exp(v) if v != NEG else 0.0 for v in lsm[t]]
        grads.append([sm[c] - post[t][c] / z for c in range(width)])
    return -math.log(z), grads


def brute_preview(width, labels, frames):
    """穷举预览：目标个数优先，其次路径分数，再其次逐帧词号字典序。"""
    total = len(frames)
    lsm = [log_softmax(row) for row in frames]
    best = None
    for path in product(range(width), repeat=total):
        col = collapse(path)
        if col != list(labels[: len(col)]):
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
        return (), 0, 0.0
    return best[1], len(collapse(best[1])), best[2]


def frames_of(capture, revision_by_chunk):
    """按给定修订展开采集中的帧（测试辅助）。"""
    chunks = {}
    for event in capture["events"]:
        if event[0] == "chunk":
            _, cid, rev, start, frames = event
            chunks.setdefault(cid, {})[rev] = (start, frames)
        elif event[0] == "revisions":
            for cid, rev, start, frames in event[1]:
                chunks.setdefault(cid, {})[rev] = (start, frames)
    total = max(start + len(frames) for revs in chunks.values() for start, frames in revs.values())
    out = [None] * total
    for cid, rev in revision_by_chunk.items():
        start, frames = chunks[cid][rev]
        for i, row in enumerate(frames):
            out[start + i] = [NEG if x is None else float(x) for x in row]
    assert all(r is not None for r in out)
    return out


def load_capture():
    with CAPTURE.open(encoding="utf-8") as stream:
        return json.load(stream)


class ReplayTests(unittest.TestCase):
    """回放采集：数值与穷举参考一致，版本与交接可追溯。"""

    def setUp(self):
        self.capture = load_capture()
        self.op, self.outcomes = replay(self.capture)
        self.by_kind = {}
        for event, result in self.outcomes:
            self.by_kind.setdefault(event[0], []).append((event, result))

    def test_preview_matches_brute_force(self):
        event, pv = self.by_kind["preview"][0]
        frames = frames_of(self.capture, {"c0": 0, "c1": 0, "c2": 0})
        path, emitted, score = brute_preview(3, self.capture["labels"], frames)
        self.assertEqual(pv.available, 7)
        self.assertEqual(pv.path, path)
        self.assertEqual(pv.emitted, emitted)
        self.assertAlmostEqual(pv.score, score, places=12)
        self.assertEqual([(i.index, i.begin, i.end) for i in pv.instances],
                         [(0, 0, 2), (1, 3, 4), (2, 5, 6)])

    def test_confirm_and_freeze_filtering(self):
        deliveries = [d for _, r in self.by_kind["revisions"] for d in r]
        status = {(d.chunk, d.revision): d.status for d in deliveries}
        self.assertEqual(status[("c0", 1)], "dropped-frozen")   # 覆盖冻结范围
        self.assertEqual(status[("c2", 1)], "applied")          # 同批合法修订不被吞掉

    def test_train_versions_and_nll(self):
        trains = {event[1]: r for event, r in self.by_kind["train"]}
        self.assertTrue(trains["old"].ok)
        self.assertFalse(trains["blocked"].ok)
        self.assertEqual(trains["blocked"].reason, "no-legal-path")
        self.assertIsNone(trains["blocked"].nll)
        self.assertTrue(trains["new"].ok)
        # 被丢弃的修订不参与版本
        self.assertEqual(dict(trains["blocked"].version.chunks),
                         {"c0": 0, "c1": 0, "c2": 1})
        self.assertEqual(dict(trains["new"].version.chunks),
                         {"c0": 0, "c1": 0, "c2": 2})
        labels = self.capture["labels"]
        frames_old = frames_of(self.capture, {"c0": 0, "c1": 0, "c2": 0})
        frames_new = frames_of(self.capture, {"c0": 0, "c1": 0, "c2": 2})
        nll_old, _ = brute_train(3, labels, frames_old, [(0, 2)])
        nll_new, _ = brute_train(3, labels, frames_new, [(0, 2)])
        self.assertAlmostEqual(trains["old"].nll, nll_old, delta=1e-6)
        self.assertAlmostEqual(trains["new"].nll, nll_new, delta=1e-6)
        self.assertIsNone(brute_train(3, labels,
                                      frames_of(self.capture, {"c0": 0, "c1": 0, "c2": 1}),
                                      [(0, 2)]))

    def test_backward_gradients_match_brute_force(self):
        backs = {event[1]: r for event, r in self.by_kind["backward"]}
        labels = self.capture["labels"]
        for tag, revs in (("old", {"c0": 0, "c1": 0, "c2": 0}),
                          ("new", {"c0": 0, "c1": 0, "c2": 2})):
            result = backs[tag]
            self.assertTrue(result.ok)
            frames = frames_of(self.capture, revs)
            nll, grads = brute_train(3, labels, frames, [(0, 2)])
            self.assertAlmostEqual(result.nll, nll, delta=1e-6)
            # 梯度按原块身份和修订交接
            self.assertEqual([(g.chunk, g.revision) for g in result.gradients],
                             [("c0", revs["c0"]), ("c1", revs["c1"]), ("c2", revs["c2"])])
            flat = [row for g in result.gradients for row in g.rows]
            self.assertEqual(len(flat), len(frames))
            for got, want in zip(flat, grads):
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, delta=1e-6)

    def test_old_credential_still_available_after_new_version(self):
        # 先前可训练版本仍可按原凭据取得梯度，且数值不随后续修订改变
        first = self.op.backward("old")
        self.op.train("later")
        second = self.op.backward("old")
        self.assertEqual(first, second)

    def test_workspace_peaks_within_declared_budget(self):
        budget = self.capture["workspace_floats"]
        for event, result in self.outcomes:
            peak = getattr(result, "peak_floats", 0)
            self.assertLessEqual(peak, budget, f"{event[0]} 峰值 {peak} 超预算")


class OperatorTests(unittest.TestCase):
    def make(self, labels=(1, 1, 2), budget=200):
        return StreamCtc(["空白", "风", "停"], labels, workspace_floats=budget)

    def test_tie_break_prefers_lexicographically_smaller_path(self):
        op = StreamCtc(["空白", "风"], [1], workspace_floats=50)
        op.chunk("c", 0, 0, [[0, 0], [0, 0]])
        pv = op.preview()
        # [0,1]、[1,0]、[1,1] 分数完全相同，取逐帧词号字典序最小者
        self.assertEqual(pv.path, (0, 1))
        self.assertEqual([(i.begin, i.end) for i in pv.instances], [(1, 2)])
        self.assertAlmostEqual(pv.score, -2 * math.log(2), places=12)

    def test_preview_reads_only_continuous_prefix(self):
        op = self.make()
        op.chunk("late", 0, 2, [[0, 5, None], [0, 5, None]])
        self.assertEqual(op.preview().available, 0)  # 第 0 帧未到，预览为空
        op.chunk("early", 0, 0, [[0, 5, None], [0, 5, None]])
        self.assertEqual(op.preview().available, 4)

    def test_confirm_commits_and_open_region_may_shift(self):
        op = self.make()
        op.chunk("c0", 0, 0, [[0, 5, None], [0, 5, None]])
        op.chunk("c1", 0, 2, [[5, None, None], [0, 5, None]])
        op.chunk("c2", 0, 4, [[3, 0, 0], [0, None, 7], [5, None, 0]])
        pv = op.preview()
        self.assertEqual([(i.begin, i.end) for i in pv.instances], [(0, 2), (3, 4), (5, 6)])
        op.confirm(0, 0, 2)
        # 开放区改发：已确认界线不动，未确认部分随之改变
        op.chunk("c2", 1, 4, [[5, 0, 1], [1, None, 6], [6, None, 0]])
        pv2 = op.preview()
        self.assertEqual((pv2.instances[0].begin, pv2.instances[0].end), (0, 2))
        self.assertEqual(pv2.version.confirmations, ((0, 2),))

    def test_confirm_requires_order_and_exact_match(self):
        op = self.make()
        op.chunk("c0", 0, 0, [[0, 5, None], [0, 5, None]])
        op.chunk("c1", 0, 2, [[5, None, None], [0, 5, None]])
        op.chunk("c2", 0, 4, [[3, 0, 0], [0, None, 7], [5, None, 0]])
        with self.assertRaises(ProtocolError):
            op.confirm(1, 3, 4)          # 必须按实例先后
        with self.assertRaises(ProtocolError):
            op.confirm(0, 0, 3)          # 区间只允许精确匹配
        op.confirm(0, 0, 2)
        with self.assertRaises(ProtocolError):
            op.confirm(0, 0, 2)          # 下一实例是 1

    def test_repeated_symbols_are_distinct_instances(self):
        op = StreamCtc(["空白", "风"], [1, 1], workspace_floats=50)
        op.chunk("c", 0, 0, [[0, 5], [5, 0], [0, 5]])
        pv = op.preview()
        self.assertEqual([(i.index, i.begin, i.end) for i in pv.instances],
                         [(0, 0, 1), (1, 2, 3)])
        op.confirm(0, 0, 1)
        op.confirm(1, 2, 3)
        self.assertEqual(op.preview().version.confirmations, ((0, 1), (2, 3)))

    def test_insufficient_repeat_transition_is_not_trainable(self):
        op = StreamCtc(["空白", "风"], [1, 1], workspace_floats=50)
        op.chunk("c", 0, 0, [[None, 5], [None, 5]])  # 没有空白可作重复字过渡
        op.finish(2)
        result = op.train("t")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "no-legal-path")

    def test_missing_frames_are_not_filled_with_blanks(self):
        op = self.make()
        op.chunk("c0", 0, 0, [[0, 5, None], [0, 5, None]])
        op.finish(7)
        result = op.train("t")  # 未送齐：显式不可训练，不以虚填空白代替观测
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "frames-missing")
        op2 = self.make()
        op2.chunk("c0", 0, 0, [[0, 5, None]])
        self.assertEqual(op2.train("t").reason, "not-finished")

    def test_explicit_minus_infinity_candidates(self):
        op = StreamCtc(["空白", "风"], [1], workspace_floats=50)
        op.chunk("c", 0, 0, [[0, None], [0, None]])  # 目标符号处处 -inf
        op.finish(2)
        self.assertEqual(op.train("t").reason, "no-legal-path")
        self.assertEqual(op.preview().emitted, 0)

    def test_dead_free_region_only_hands_off_confirmed_prefix(self):
        op = StreamCtc(["空白", "风"], [1, 1], workspace_floats=50)
        op.chunk("c0", 0, 0, [[None, 5]])
        op.chunk("c1", 0, 1, [[5, None], [0, 5]])
        pv = op.preview()
        self.assertEqual([(i.begin, i.end) for i in pv.instances], [(0, 1), (2, 3)])
        op.confirm(0, 0, 1)
        # 开放区改发：第 1 帧变成只能出风。自由区首帧须为空白（重复字过渡），
        # 但空白观测为 -inf，余下帧再无合法路径；已确认界线不受影响。
        op.chunk("c1", 1, 1, [[None, 5], [0, 5]])
        pv2 = op.preview()
        self.assertEqual(pv2.covered, 1)
        self.assertLess(pv2.covered, pv2.available)
        self.assertEqual([(i.begin, i.end) for i in pv2.instances], [(0, 1)])
        op.finish(3)
        self.assertEqual(op.train("t").reason, "no-legal-path")

    def test_finish_is_idempotent_and_immutable(self):
        op = self.make()
        op.finish(7)
        op.finish(7)
        with self.assertRaises(ProtocolError):
            op.finish(8)

    def test_empty_labels_and_zero_length(self):
        op = StreamCtc(["空白", "风"], [], workspace_floats=50)
        op.chunk("c", 0, 0, [[1, 0], [2, 0]])
        pv = op.preview()
        self.assertEqual(pv.emitted, 0)
        self.assertEqual(pv.path, (0, 0))
        op.finish(2)
        train = op.train("t")
        self.assertTrue(train.ok)
        lsm = [log_softmax(row) for row in [[1.0, 0.0], [2.0, 0.0]]]
        self.assertAlmostEqual(train.nll, -(lsm[0][0] + lsm[1][0]), places=12)
        back = op.backward("t")
        self.assertEqual(len(back.gradients[0].rows), 2)
        op0 = StreamCtc(["空白", "风"], [], workspace_floats=50)
        op0.finish(0)
        self.assertEqual(op0.train("t").nll, 0.0)

    def test_chunking_does_not_change_values(self):
        labels = (1, 1, 2)
        frames = [[0, 5, None], [0, 5, None], [5, None, None], [0, 5, None],
                  [3, 0, 0], [0, None, 7], [5, None, 0]]
        results = []
        for splits in ((7,), (2, 2, 3), (1, 1, 5), (4, 3)):
            op = self.make(labels)
            start = 0
            for i, n in enumerate(splits):
                op.chunk(f"c{i}", 0, start, frames[start:start + n])
                start += n
            op.confirm(0, 0, 2)
            op.finish(7)
            train = op.train("t")
            back = op.backward("t")
            flat = tuple(tuple(r) for g in back.gradients for r in g.rows)
            results.append((train.nll, flat, op.preview().path))
        first = results[0]
        for nll, grads, path in results[1:]:
            self.assertEqual(nll, first[0])
            self.assertEqual(grads, first[1])
            self.assertEqual(path, first[2])

    def test_workspace_budget_enforced(self):
        op = self.make(budget=20)
        op.chunk("c0", 0, 0, [[0, 5, None], [0, 5, None]])
        op.chunk("c1", 0, 2, [[5, None, None], [0, 5, None]])
        op.chunk("c2", 0, 4, [[3, 0, 0], [0, None, 7], [5, None, 0]])
        op.confirm(0, 0, 2)
        op.finish(7)
        self.assertTrue(op.train("t").ok)       # 前向只需一行状态
        with self.assertRaises(WorkspaceError):
            op.backward("t")                    # 后向检查点 + 块内重算超预算

    def test_geometry_and_revision_rules(self):
        op = self.make()
        op.chunk("c", 0, 0, [[0, 5, None]])
        with self.assertRaises(ProtocolError):
            op.chunk("c", 1, 1, [[0, 5, None]])   # 同 id 必须同起点及帧数
        with self.assertRaises(ProtocolError):
            op.chunk("c", 0, 0, [[0, 4, None]])   # 同修订号数据冲突
        self.assertEqual(op.chunk("c", 0, 0, [[0, 5, None]]).status, "duplicate")
        with self.assertRaises(ProtocolError):
            op.chunk("d", 0, 0, [[0, 5, None]])   # 区间重叠
        op.finish(3)
        with self.assertRaises(ProtocolError):
            op.chunk("e", 0, 2, [[0, 5, None], [0, 5, None]])  # 超出全长

    def test_credential_rules(self):
        op = self.make()
        op.chunk("c", 0, 0, [[0, 5, None]])
        op.finish(1)
        op.train("x")
        with self.assertRaises(ProtocolError):
            op.train("x")
        with self.assertRaises(ProtocolError):
            op.backward("unknown")
        # 不可训练版本的凭据同样显式给出原因
        op2 = StreamCtc(["空白", "风"], [1], workspace_floats=50)
        op2.chunk("c", 0, 0, [[0, None]])
        op2.finish(1)
        self.assertFalse(op2.train("bad").ok)
        back = op2.backward("bad")
        self.assertFalse(back.ok)
        self.assertEqual(back.reason, "no-legal-path")

    def test_version_traceability(self):
        op = self.make()
        op.chunk("c0", 0, 0, [[0, 5, None], [0, 5, None]])
        op.chunk("c1", 0, 2, [[5, None, None], [0, 5, None]])
        pv = op.preview()
        op.confirm(0, 0, 2)
        op.finish(4)
        train = op.train("t")
        # 编辑看到的预览与训练看到的损失是同一个版本的事实
        self.assertNotEqual(pv.version, train.version)  # 确认改变了边界承诺
        self.assertEqual(train.version.confirmations, ((0, 2),))
        self.assertEqual(dict(train.version.chunks), {"c0": 0, "c1": 0})
        back = op.backward("t")
        self.assertEqual(back.version, train.version)


if __name__ == "__main__":
    unittest.main()
