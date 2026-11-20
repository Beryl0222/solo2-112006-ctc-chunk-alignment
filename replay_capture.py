"""把交接采集回放进流式 CTC 算子，并打印每个事件的结果。

用法::

    python -m replay_capture samples/capture.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_operator import StreamCtc
from read_capture import read_capture


def replay(capture):
    """按采集中的事件顺序驱动算子，返回每个事件的结果记录。"""
    op = StreamCtc(
        capture["alphabet"],
        capture["labels"],
        workspace_floats=capture.get("workspace_floats", 1 << 20),
    )
    outcomes = []
    for event in capture["events"]:
        kind = event[0]
        if kind == "chunk":
            _, chunk_id, revision, start, frames = event
            outcomes.append((event, op.chunk(chunk_id, revision, start, frames)))
        elif kind == "revisions":
            outcomes.append((event, op.revisions(event[1])))
        elif kind == "preview":
            outcomes.append((event, op.preview()))
        elif kind == "confirm":
            _, instance, begin, end = event
            outcomes.append((event, op.confirm(instance, begin, end)))
        elif kind == "finish":
            outcomes.append((event, op.finish(event[1])))
        elif kind == "train":
            outcomes.append((event, op.train(event[1])))
        elif kind == "backward":
            outcomes.append((event, op.backward(event[1])))
        else:
            raise ValueError(f"未知事件类型: {kind!r}")
    return op, outcomes


def _fmt_float(x):
    return "—" if x is None else f"{x:.6f}"


def main():
    parser = argparse.ArgumentParser(description="回放交接采集并打印算子结果")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    with Path(args.capture).open(encoding="utf-8") as stream:
        capture = json.load(stream)
    _, outcomes = replay(capture)
    for index, (event, result) in enumerate(outcomes, start=1):
        kind = event[0]
        if kind == "chunk":
            print(f"{index:02} chunk {result.chunk} r{result.revision} -> {result.status}")
        elif kind == "revisions":
            joined = "; ".join(f"{d.chunk} r{d.revision} {d.status}" for d in result)
            print(f"{index:02} revisions -> {joined}")
        elif kind == "preview":
            inst = "; ".join(
                f"#{i.index} [{i.begin},{i.end})" for i in result.instances
            )
            print(
                f"{index:02} preview -> available={result.available} "
                f"emitted={result.emitted} score={result.score:.6f} "
                f"path={list(result.path)} instances=[{inst}] v={result.version.digest()}"
            )
        elif kind == "confirm":
            print(f"{index:02} confirm #{result.index} [{result.begin},{result.end}) -> ok")
        elif kind == "finish":
            print(f"{index:02} finish total={event[1]} -> ok")
        elif kind == "train":
            if result.ok:
                print(
                    f"{index:02} train {result.credential} -> ok "
                    f"nll={result.nll:.6f} v={result.version.digest()}"
                )
            else:
                print(
                    f"{index:02} train {result.credential} -> not-trainable "
                    f"({result.reason}) v={result.version.digest()}"
                )
        elif kind == "backward":
            if result.ok:
                chunks = "; ".join(
                    f"{g.chunk} r{g.revision} [{g.start},{g.start + len(g.rows)})"
                    for g in result.gradients
                )
                print(
                    f"{index:02} backward {result.credential} -> ok "
                    f"nll={result.nll:.6f} grads: {chunks} v={result.version.digest()}"
                )
            else:
                print(
                    f"{index:02} backward {result.credential} -> not-trainable "
                    f"({result.reason}) v={result.version.digest()}"
                )


if __name__ == "__main__":
    main()
