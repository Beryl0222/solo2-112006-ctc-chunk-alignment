"""流式 CTC 算子：承接帧块回传、校对确认与训练交接。

配合《交接协议.md》实现：

- ``alphabet[0]`` 为 CTC 空白，``labels`` 为固定目标符号串；帧数据是 logits，
  ``None`` 表示负无穷，每帧至少一列有限；路径分值为逐帧全词表
  log-softmax 之和，折叠为标准 CTC 规则（先并连续相同，再去空白）。
- 预览只读取从第 0 帧开始连续可用的前缀，允许折叠为目标任意前缀；
  排序口径：已输出目标个数（多者优先）→ 路径分数（高者优先）→
  逐帧词号序列字典序（小者优先）。相邻相同符号仍需空白分隔，
  块边界不重置这条规则。
- ``confirm(instance, begin, end)`` 按实例先后、与当前预览对应实例的
  区间精确匹配；确认后 ``[0, 最后确认实例右界)`` 冻结，覆盖冻结范围的
  修订不参与当前版本，同批其余合法修订照常应用。
- ``finish`` 声明全长（不代表送齐）。完整且可训练的版本给出负对数似然；
  梯度按 ``(块 id, 修订)`` 交接。确认或修订可能使路径域为空，此时
  显式返回不可训练及原因；先前可训练版本仍按原凭据取梯度。
- ``workspace_floats`` 约束前后向与回溯的浮点临时元素峰值：帧块输入、
  交接输出与外部顺序读写不计入。前向后向采用块检查点重算，
  不在内存保存全部帧状态表；浮点峰值仅依赖内部块划分的
  ``(块数 + 块长) * 状态数`` 量级。数值结果为规范顺序的 IEEE 双精度，
  与帧块如何切分无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NEG = float("-inf")

# 每次动态规划为标量临时量（归一常数、累加器等）预留的工作区槽位。
_SCALAR_SLOTS = 8

# 帧视图顺序读取的分片粒度（帧块输入不计入工作区）。
_READ_GRAIN = 1024


class ProtocolError(Exception):
    """违反交接协议的操作（几何冲突、确认不匹配、凭据未知等）。"""


class WorkspaceError(Exception):
    """声明的 workspace_floats 不足以容纳前后向/回溯的浮点临时元素。"""


class _Workspace:
    """浮点临时元素记账：跟踪峰值并在超预算时失败。

    只计浮点临时元素（动态规划行、检查点、块内重算表）。
    帧块输入、交接输出与外部顺序读写不计入。
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.current = 0
        self.peak = 0

    def alloc(self, n: int) -> None:
        if n <= 0:
            return
        self.current += n
        if self.current > self.peak:
            self.peak = self.current
        if self.peak > self.budget:
            raise WorkspaceError(
                f"浮点临时元素峰值 {self.peak} 超出 workspace_floats={self.budget}"
            )

    def release(self, n: int) -> None:
        if n <= 0:
            return
        self.current -= n


def _logaddexp(a: float, b: float) -> float:
    """规范顺序的标量 logaddexp（IEEE 双精度，确定可复现）。"""
    if a == NEG:
        return b
    if b == NEG:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _row_norm(row) -> float:
    """逐帧全词表 log-softmax 的归一常数 logsumexp（按下标顺序求和）。"""
    m = max(row)
    total = 0.0
    for x in row:
        if x != NEG:
            total += math.exp(x - m)
    return m + math.log(total)


class _Ext:
    """目标（后缀）串的扩展序列与 CTC 转移规则。

    状态 ``2i`` 为第 i 个间隔空白，状态 ``2i+1`` 为第 i 个目标符号；
    相邻相同符号之间的空白不可跳过（空白分隔规则），与块边界无关。
    """

    __slots__ = ("labels", "size", "symbol", "pred", "succ")

    def __init__(self, labels):
        self.labels = tuple(labels)
        size = 2 * len(self.labels) + 1
        symbol = [0] * size
        for i, c in enumerate(self.labels):
            symbol[2 * i + 1] = c
        pred = []
        for s in range(size):
            ps = [s]
            if s >= 1:
                ps.append(s - 1)
            if s >= 2 and symbol[s] != symbol[s - 2]:
                ps.append(s - 2)
            pred.append(tuple(ps))
        succ = []
        for s in range(size):
            ns = [s]
            if s + 1 < size:
                ns.append(s + 1)
            if s + 2 < size and symbol[s + 2] != symbol[s]:
                ns.append(s + 2)
            succ.append(tuple(ns))
        self.size = size
        self.symbol = symbol
        self.pred = pred
        self.succ = succ


def _lse_step(alpha, row, norm, ext):
    """前向一步（log 域，原地降序更新，保持规范求值顺序）。"""
    symbol = ext.symbol
    for s in range(ext.size - 1, -1, -1):
        acc = NEG
        for p in ext.pred[s]:
            acc = _logaddexp(acc, alpha[p])
        alpha[s] = acc + row[symbol[s]] - norm


def _lse_back_step(beta, row, norm, ext):
    """后向一步（log 域，原地升序更新）。"""
    symbol = ext.symbol
    for s in range(ext.size):
        acc = NEG
        for q in ext.succ[s]:
            acc = _logaddexp(acc, beta[q] + row[symbol[q]] - norm)
        beta[s] = acc


def _forward(view, a, b, ext, init_label, ws, block=0):
    """自由区 ``[a, b)`` 的前向扫描。

    返回 ``(logZ, checkpoints)``；``block > 0`` 时在每个内部块边界保存
    检查点（供后向重算），否则只顺序读取帧块、不留存帧状态。
    """
    size = ext.size
    symbol = ext.symbol
    n_frames = b - a
    n_blocks = -(-n_frames // block) if block else 0
    ws.alloc(size + _SCALAR_SLOTS)
    if n_blocks:
        ws.alloc(max(0, n_blocks - 1) * size)
    try:
        alpha = [NEG] * size
        checkpoints = []
        t = a
        first = True
        while t < b:
            if block:
                e = min(t + block, b)
            else:
                e = min(t + _READ_GRAIN, b)  # 按需分片顺序读取，不假定全量常驻
            for row in view.read(t, e):  # 帧块输入：不计入工作区
                norm = _row_norm(row)
                if first:
                    alpha[0] = row[0] - norm
                    if size > 1:
                        alpha[1] = (row[symbol[1]] - norm) if init_label else NEG
                    first = False
                else:
                    _lse_step(alpha, row, norm, ext)
                t += 1
            if block and e < b:
                checkpoints.append(tuple(alpha))
    finally:
        ws.release(size + _SCALAR_SLOTS + (max(0, n_blocks - 1) * size if n_blocks else 0))
    if size == 1:
        return alpha[0], checkpoints
    return _logaddexp(alpha[size - 1], alpha[size - 2]), checkpoints


def _backward(view, a, b, ext, init_label, log_z, checkpoints, block, ws, emit_grad):
    """自由区 ``[a, b)`` 的后向扫描，逐帧产出对原 logits 的梯度。

    仅在块边界保存前向检查点，块内前向表按需重算；浮点临时元素峰值
    为 ``(块数 + 块长) * 状态数`` 量级，不保存全部帧状态。
    """
    size = ext.size
    symbol = ext.symbol
    n_frames = b - a
    n_blocks = -(-n_frames // block)
    ws.alloc(size + _SCALAR_SLOTS)          # beta 行与标量
    ws.alloc((n_blocks - 1) * size)         # 前向检查点（前向扫描时已计入，此处按峰值复用记账）
    ws.alloc(block * size)                  # 块内重算的前向表
    try:
        beta = [NEG] * size
        beta[size - 1] = 0.0
        if size >= 2:
            beta[size - 2] = 0.0
        for j in range(n_blocks - 1, -1, -1):
            s_j = a + j * block
            e_j = min(s_j + block, b)
            rows = view.read(s_j, e_j)      # 帧块输入：不计入工作区
            table = []
            prev = None
            for i, row in enumerate(rows):
                norm = _row_norm(row)
                t = s_j + i
                if t == a:
                    cur = [NEG] * size
                    cur[0] = row[0] - norm
                    if size > 1:
                        cur[1] = (row[symbol[1]] - norm) if init_label else NEG
                elif i == 0:
                    cur = list(checkpoints[j - 1])
                    _lse_step(cur, row, norm, ext)
                else:
                    cur = list(prev)
                    _lse_step(cur, row, norm, ext)
                table.append(cur)
                prev = cur
            for i in range(len(rows) - 1, -1, -1):
                t = s_j + i
                row = rows[i]
                norm = _row_norm(row)
                grad = [math.exp(x - norm) if x != NEG else 0.0 for x in row]
                alpha_t = table[i]
                for s in range(size):
                    av = alpha_t[s]
                    bv = beta[s]
                    if av == NEG or bv == NEG:
                        continue
                    grad[symbol[s]] -= math.exp(av + bv - log_z)
                emit_grad(t, grad)          # 交接输出：不计入工作区
                if t > a:
                    _lse_back_step(beta, row, norm, ext)
    finally:
        ws.release(size + _SCALAR_SLOTS + (n_blocks - 1) * size + block * size)


def _viterbi(view, a, b, ext, init_label, ws):
    """自由区 ``[a, b)`` 的 Viterbi 对齐（预览回溯）。

    分数相同（完全相同的对数分数）时保留逐帧词号序列字典序较小的
    候选路径；最终状态按（已输出目标个数, 路径分数, 字典序）选取。
    返回 ``(score, path)``；自由区无任何合法路径时返回 ``(NEG, None)``。
    """
    size = ext.size
    symbol = ext.symbol
    ws.alloc(size + _SCALAR_SLOTS)
    try:
        score = [NEG] * size
        paths = [None] * size
        t = a
        first = True
        while t < b:
            e = min(t + _READ_GRAIN, b)  # 按需分片顺序读取可用前缀
            for row in view.read(t, e):
                norm = _row_norm(row)
                if first:
                    score[0] = row[0] - norm
                    paths[0] = (0,)
                    if size > 1 and init_label:
                        score[1] = row[symbol[1]] - norm
                        paths[1] = (symbol[1],)
                    first = False
                else:
                    for s in range(size - 1, -1, -1):
                        best_score = NEG
                        best_path = None
                        for p in ext.pred[s]:
                            sp = score[p]
                            if sp == NEG:
                                continue
                            if sp > best_score or (
                                sp == best_score and best_path is not None and paths[p] < best_path
                            ):
                                best_score = sp
                                best_path = paths[p]
                        if best_path is None:
                            score[s] = NEG
                            paths[s] = None
                        else:
                            score[s] = best_score + row[symbol[s]] - norm
                            paths[s] = best_path + (symbol[s],)
                t += 1
        chosen = None
        for s in range(size):
            if score[s] == NEG:
                continue
            count = (s + 1) // 2
            if chosen is None:
                chosen = (count, score[s], paths[s])
                continue
            c0, s0, p0 = chosen
            if count > c0 or (count == c0 and (score[s] > s0 or (score[s] == s0 and paths[s] < p0))):
                chosen = (count, score[s], paths[s])
    finally:
        ws.release(size + _SCALAR_SLOTS)
    if chosen is None:
        return NEG, None
    return chosen[1], chosen[2]


def _backward_block_size(n_frames, size, budget):
    """为后向扫描选取内部块长：最小化 ``(块数 + 块长) * 状态数`` 峰值。"""
    best = None
    for block in range(1, n_frames + 1):
        n_blocks = -(-n_frames // block)
        peak = (n_blocks + block) * size + _SCALAR_SLOTS
        if best is None or peak < best[0] or (peak == best[0] and block > best[1]):
            best = (peak, block)
    if best is None or best[0] > budget:
        raise WorkspaceError(
            f"后向扫描至少需要 {best[0] if best else '—'} 个浮点临时元素，"
            f"超出 workspace_floats={budget}"
        )
    return best[1]


def _segments(confirmations, labels):
    """已确认区间化为帧段序列：间隙为空白，实例区间为对应目标符号。"""
    prev = 0
    for i, (begin, end) in enumerate(confirmations):
        if begin > prev:
            yield prev, begin, 0
        yield begin, end, labels[i]
        prev = end


def _forced_symbols(confirmations, labels):
    out = []
    for begin, end, c in _segments(confirmations, labels):
        out.extend([c] * (end - begin))
    return tuple(out)


def _instances(path):
    """从逐帧词号序列解析词实例及其半开绝对帧区间（重复字按位置区分）。"""
    out = []
    i = 0
    n = len(path)
    while i < n:
        c = path[i]
        if c == 0:
            i += 1
            continue
        j = i + 1
        while j < n and path[j] == c:
            j += 1
        out.append(Instance(index=len(out), symbol=c, begin=i, end=j))
        i = j
    return tuple(out)


@dataclass(frozen=True)
class Instance:
    """一个词实例及其占据的半开绝对帧区间 [begin, end)。"""

    index: int
    symbol: int
    begin: int
    end: int


@dataclass(frozen=True)
class Version:
    """版本事实：参与版本的 (块 id, 修订)、边界承诺与声明全长。"""

    chunks: tuple
    confirmations: tuple
    total: object

    def digest(self) -> str:
        import hashlib

        return hashlib.sha1(repr(self).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Delivery:
    chunk: str
    revision: int
    status: str  # "applied" / "dropped-frozen" / "duplicate"


@dataclass(frozen=True)
class PreviewResult:
    available: int          # 从第 0 帧连续可用的帧数
    covered: int            # 返回路径实际覆盖的帧数（自由区无解时小于 available）
    path: tuple             # 逐帧词号序列
    emitted: int            # 已输出目标符号个数
    score: float            # 路径对数分数
    instances: tuple        # 词实例及其绝对帧区间
    version: Version
    peak_floats: int


@dataclass(frozen=True)
class TrainResult:
    ok: bool
    reason: object          # None / "not-finished" / "frames-missing" / "no-legal-path"
    nll: object
    version: Version
    credential: str
    peak_floats: int


@dataclass(frozen=True)
class ChunkGradient:
    chunk: str
    revision: int
    start: int
    rows: tuple             # 该块各帧对原 logits 的梯度


@dataclass(frozen=True)
class BackwardResult:
    ok: bool
    reason: object
    nll: object
    gradients: object       # tuple[ChunkGradient, ...]，按块起点排序
    version: Version
    credential: str
    peak_floats: int


class _Chunk:
    __slots__ = ("start", "length", "current", "data")

    def __init__(self, start, length, revision, rows):
        self.start = start
        self.length = length
        self.current = revision
        self.data = {revision: rows}


class _FrameView:
    """按块拼接的帧视图：前向后向只通过 ``read(a, b)`` 顺序读取。"""

    def __init__(self, parts):
        # parts: 按起点排序的 (start, rows) ，行内 None 已替换为 -inf
        self._parts = tuple(parts)

    def read(self, a, b):
        out = []
        for start, rows in self._parts:
            end = start + len(rows)
            if end <= a:
                continue
            if start >= b:
                break
            lo = max(a, start) - start
            hi = min(b, end) - start
            out.extend(rows[lo:hi])
        if len(out) != b - a:
            raise ProtocolError(f"帧区间 [{a}, {b}) 未被完全覆盖")
        return out


class _Snapshot:
    __slots__ = ("credential", "version", "layout", "confirmations", "total",
                 "ok", "reason", "nll", "gradients", "peak")

    def __init__(self, credential, version, layout, confirmations, total,
                 ok, reason, nll, peak):
        self.credential = credential
        self.version = version
        self.layout = layout
        self.confirmations = confirmations
        self.total = total
        self.ok = ok
        self.reason = reason
        self.nll = nll
        self.gradients = None
        self.peak = peak


class StreamCtc:
    """承接语音标注台流程的流式 CTC 算子。

    输入为固定目标符号串、可替换的帧块与校对边界；编辑（预览/确认）
    与训练（损失/梯度）看到同一个版本的事实。
    """

    def __init__(self, alphabet, labels, workspace_floats=1 << 20):
        if not alphabet:
            raise ProtocolError("alphabet 不能为空")
        self.alphabet = tuple(alphabet)
        self.width = len(self.alphabet)
        self.labels = tuple(int(c) for c in labels)
        for c in self.labels:
            if c <= 0 or c >= self.width:
                raise ProtocolError("目标符号必须是 alphabet 中的非空白索引")
        if workspace_floats < 1:
            raise ProtocolError("workspace_floats 必须为正")
        self._budget = int(workspace_floats)
        self._chunks = {}
        self._intervals = []          # 按起点排序的 (start, end, chunk_id)
        self._confirmations = []
        self._total = None
        self._snapshots = {}
        self.deliveries = []          # 交接日志：每次块投递的结果（含被冻结丢弃的修订）

    # ------------------------------------------------------------------
    # 帧块投递

    def chunk(self, chunk_id, revision, start, frames):
        """投递单个帧块（新块或修订）。"""
        delivery = self._deliver(chunk_id, revision, start, frames)
        self.deliveries.append(delivery)
        return delivery

    def revisions(self, entries):
        """批量投递修订：逐条校验，覆盖冻结范围的丢弃，不吞掉同批合法修订。"""
        out = []
        for entry in entries:
            chunk_id, revision, start, frames = entry
            delivery = self._deliver(chunk_id, revision, start, frames)
            self.deliveries.append(delivery)
            out.append(delivery)
        return tuple(out)

    def _deliver(self, chunk_id, revision, start, frames):
        if not isinstance(revision, int) or revision < 0:
            raise ProtocolError("修订号必须是非负整数")
        if not isinstance(start, int) or start < 0:
            raise ProtocolError("帧块起点必须是非负整数")
        rows = self._validate_frames(frames)
        length = len(rows)
        if length == 0:
            raise ProtocolError("帧块不能为空")
        if self._total is not None and start + length > self._total:
            raise ProtocolError("帧块超出 finish 声明的全长")
        rec = self._chunks.get(chunk_id)
        if rec is None:
            self._check_overlap(start, start + length)
            if start < self._frozen():
                return Delivery(chunk_id, revision, "dropped-frozen")
            self._chunks[chunk_id] = _Chunk(start, length, revision, rows)
            self._insert_interval(start, start + length, chunk_id)
            return Delivery(chunk_id, revision, "applied")
        if (start, length) != (rec.start, rec.length):
            raise ProtocolError("同 id 帧块必须保持相同起点及帧数")
        if revision == rec.current:
            if rows == rec.data[revision]:
                return Delivery(chunk_id, revision, "duplicate")
            raise ProtocolError("同一 (块 id, 修订) 的数据冲突")
        if revision < rec.current:
            raise ProtocolError("修订号必须递增")
        if start < self._frozen():
            return Delivery(chunk_id, revision, "dropped-frozen")
        rec.data[revision] = rows
        rec.current = revision
        return Delivery(chunk_id, revision, "applied")

    def _validate_frames(self, frames):
        rows = []
        for row in frames:
            if len(row) != self.width:
                raise ProtocolError("帧 logits 列数必须与 alphabet 一致")
            vals = []
            finite = 0
            for x in row:
                if x is None:
                    vals.append(NEG)
                    continue
                v = float(x)
                if math.isnan(v) or v == float("inf"):
                    raise ProtocolError("logits 只能取有限值或负无穷")
                vals.append(v)
                finite += 1
            if finite == 0:
                raise ProtocolError("每帧至少一列有限")
            rows.append(tuple(vals))
        if not rows:
            raise ProtocolError("帧块不能为空")
        return tuple(rows)

    def _check_overlap(self, start, end):
        for s, e, cid in self._intervals:
            if s < end and start < e:
                raise ProtocolError(f"帧块区间 [{start}, {end}) 与已有块 {cid} 重叠")

    def _insert_interval(self, start, end, chunk_id):
        i = 0
        while i < len(self._intervals) and self._intervals[i][0] < start:
            i += 1
        self._intervals.insert(i, (start, end, chunk_id))

    # ------------------------------------------------------------------
    # 校对

    def preview(self):
        """在可用前缀上给出当前版本的对齐预览。"""
        ws = _Workspace(self._budget)
        available = self._available_prefix()
        frozen = self._frozen()
        k = len(self._confirmations)
        view = self._view(self._layout())
        forced = _forced_symbols(self._confirmations, self.labels)
        forced_score, legal = self._forced_score(view, forced)
        ext = _Ext(self.labels[k:])
        init_label = self._init_label_allowed(k)
        covered = available
        if available > frozen:
            score, path = _viterbi(view, frozen, available, ext, init_label, ws)
            if path is None:
                # 自由区已无任何合法路径：只能交接已确认的强制前缀
                score, path = 0.0, ()
                covered = frozen
        else:
            score, path = 0.0, ()
        if not legal:
            forced_score = NEG
        full = forced + path
        return PreviewResult(
            available=available,
            covered=covered,
            path=full,
            emitted=len(_instances(full)),
            score=forced_score + score,
            instances=_instances(full),
            version=self._version(),
            peak_floats=ws.peak,
        )

    def confirm(self, instance, begin, end):
        """承诺词实例 ``instance`` 占据绝对帧区间 [begin, end)。

        必须按实例先后，且与当前预览对应实例的范围精确匹配。
        """
        if instance != len(self._confirmations):
            raise ProtocolError("确认必须按照实例先后")
        pv = self.preview()
        if instance >= len(pv.instances):
            raise ProtocolError("当前预览尚未输出该词实例")
        inst = pv.instances[instance]
        if (inst.begin, inst.end) != (begin, end):
            raise ProtocolError(
                f"确认区间 [{begin}, {end}) 与当前预览实例区间 "
                f"[{inst.begin}, {inst.end}) 不一致，区间只允许精确匹配"
            )
        self._confirmations.append((begin, end))
        return inst

    def finish(self, total):
        """声明全长；长度不代表实际送齐。"""
        total = int(total)
        if total < 0:
            raise ProtocolError("全长不能为负")
        if self._total is not None:
            if self._total != total:
                raise ProtocolError("全长已声明，不能更改")
            return
        for s, e, cid in self._intervals:
            if e > total:
                raise ProtocolError(f"已有帧块 {cid} 超出声明全长")
        self._total = total

    # ------------------------------------------------------------------
    # 训练

    def train(self, credential):
        """查询当前版本是否可训练；可训练则登记凭据并返回负对数似然。"""
        if credential in self._snapshots:
            raise ProtocolError(f"凭据 {credential!r} 已登记")
        ws = _Workspace(self._budget)
        version = self._version()
        layout = self._layout()
        ok, reason, nll = False, None, None
        if self._total is None:
            reason = "not-finished"
        elif self._available_prefix() < self._total:
            reason = "frames-missing"
        else:
            k = len(self._confirmations)
            frozen = self._frozen()
            view = self._view(layout)
            forced = _forced_symbols(self._confirmations, self.labels)
            forced_score, legal = self._forced_score(view, forced)
            if not legal:
                reason = "no-legal-path"
            elif frozen == self._total:
                if k == len(self.labels):
                    ok, nll = True, -forced_score
                else:
                    reason = "no-legal-path"
            else:
                ext = _Ext(self.labels[k:])
                init_label = self._init_label_allowed(k)
                log_z, _ = _forward(view, frozen, self._total, ext, init_label, ws)
                if log_z == NEG:
                    reason = "no-legal-path"
                else:
                    ok, nll = True, -(forced_score + log_z)
        snapshot = _Snapshot(credential, version, layout, tuple(self._confirmations),
                             self._total, ok, reason, nll, ws.peak)
        self._snapshots[credential] = snapshot
        return TrainResult(ok=ok, reason=reason, nll=nll if ok else None,
                           version=version, credential=credential, peak_floats=ws.peak)

    def backward(self, credential):
        """按原凭据取回该版本对原 logits 的逐帧梯度（按块身份与修订交接）。"""
        snapshot = self._snapshots.get(credential)
        if snapshot is None:
            raise ProtocolError(f"未知凭据 {credential!r}")
        if not snapshot.ok:
            return BackwardResult(ok=False, reason=snapshot.reason, nll=None,
                                  gradients=None, version=snapshot.version,
                                  credential=credential, peak_floats=0)
        if snapshot.gradients is None:
            snapshot.gradients, snapshot.peak = self._compute_gradients(snapshot)
        return BackwardResult(ok=True, reason=None, nll=snapshot.nll,
                              gradients=snapshot.gradients, version=snapshot.version,
                              credential=credential, peak_floats=snapshot.peak)

    def _compute_gradients(self, snapshot):
        ws = _Workspace(self._budget)
        total = snapshot.total
        k = len(snapshot.confirmations)
        frozen = snapshot.confirmations[-1][1] if k else 0
        view = self._view(snapshot.layout)
        grad_rows = [None] * total      # 交接输出：不计入工作区

        def emit(t, grad):
            grad_rows[t] = tuple(grad)

        # 已确认强制前缀：后验为 1，梯度 = softmax - onehot
        for begin, end, c in _segments(snapshot.confirmations, self.labels):
            t = begin
            while t < end:
                e = min(t + _READ_GRAIN, end)
                for row in view.read(t, e):
                    norm = _row_norm(row)
                    grad = [math.exp(x - norm) if x != NEG else 0.0 for x in row]
                    grad[c] -= 1.0
                    emit(t, grad)
                    t += 1
        # 自由区：约束 CTC 前向后向
        if frozen < total:
            ext = _Ext(self.labels[k:])
            init_label = self._init_label_allowed(k)
            block = _backward_block_size(total - frozen, ext.size, self._budget)
            log_z, checkpoints = _forward(view, frozen, total, ext, init_label, ws, block)
            _backward(view, frozen, total, ext, init_label, log_z, checkpoints,
                      block, ws, emit)
        gradients = []
        for chunk_id, revision, start, length in snapshot.layout:
            rows = tuple(grad_rows[start:start + length])
            if any(r is None for r in rows):
                raise ProtocolError("帧区间未被完全覆盖")
            gradients.append(ChunkGradient(chunk=chunk_id, revision=revision,
                                           start=start, rows=rows))
        return tuple(gradients), ws.peak

    # ------------------------------------------------------------------
    # 版本与视图

    def _frozen(self):
        return self._confirmations[-1][1] if self._confirmations else 0

    def _init_label_allowed(self, k):
        """自由区首帧能否直接落在目标符号 k 上（相邻相同符号需空白分隔）。"""
        return k < len(self.labels) and (k == 0 or self.labels[k] != self.labels[k - 1])

    def _available_prefix(self):
        cur = 0
        for s, e, _cid in self._intervals:
            if s > cur:
                break
            if e > cur:
                cur = e
        return cur

    def _version(self):
        return Version(
            chunks=tuple(sorted((cid, rec.current) for cid, rec in self._chunks.items())),
            confirmations=tuple(self._confirmations),
            total=self._total,
        )

    def _layout(self):
        return tuple(
            (cid, rec.current, rec.start, rec.length)
            for cid, rec in sorted(self._chunks.items(), key=lambda kv: kv[1].start)
        )

    def _view(self, layout):
        parts = []
        for chunk_id, revision, start, _length in layout:
            rec = self._chunks[chunk_id]
            rows = rec.data.get(revision)
            if rows is None:
                raise ProtocolError(f"块 {chunk_id} 修订 {revision} 的数据已不可用")
            parts.append((start, rows))
        return _FrameView(parts)

    def _forced_score(self, view, forced):
        """已确认强制前缀的对数分数；强制符号观测为 -inf 时不可训练。"""
        score = 0.0
        legal = True
        t = 0
        n = len(forced)
        while t < n:
            e = min(t + _READ_GRAIN, n)
            for row in view.read(t, e):
                norm = _row_norm(row)
                v = row[forced[t]] - norm
                if v == NEG:
                    legal = False
                else:
                    score += v
                t += 1
        return (score if legal else NEG), legal
