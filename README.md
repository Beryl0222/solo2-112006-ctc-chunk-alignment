# 流式 CTC 对齐、封段与训练梯度

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。

- `ctc_operator.py`：流式 CTC 算子 `StreamCtc`。以固定目标符号串、可替换的
  帧块（`chunk` / `revisions`）和校对边界（`preview` / `confirm`）为输入，
  编辑和训练看到同一个 `Version` 的事实。`train` 返回约束 CTC 负对数似然，
  `backward` 按原块身份和修订交接各 logits 梯度；版本不可训练时显式给出
  原因（`not-finished` / `frames-missing` / `no-legal-path`），先前可训练
  版本仍按原凭据取梯度。前向后向按块检查点重算，浮点临时元素峰值受
  `workspace_floats` 约束，不保存全部帧状态；数值为规范顺序的 IEEE 双精度，
  与帧块切分无关。
- `replay_capture.py`：把采集回放进算子并打印每个事件的结果。
- `read_capture.py`：读取采集文件并列出操作顺序。
- `stress_diff.py`：随机差分压力测试，与穷举全部路径的参考实现对拍。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m replay_capture samples/capture.json
python -m unittest discover -s tests -v
python stress_diff.py
```
