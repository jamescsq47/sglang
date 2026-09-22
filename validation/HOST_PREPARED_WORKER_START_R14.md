# R14：Slow PREPARE 后后台启动（CPU 验证范围）

## 原因与范围

R13 已去掉复制完成后的额外 COMMIT 轮次，但复制开始前仍需要：
rank0 native PREPARE → Host worker claim → native workset plan 分配完整空间
→ attach/prepare ACK → scheduler 观察 status=1 → 下一次 native START。
R14 仅去掉最后的重复 START 等待，不改变资源分配和恢复路线。

只在 TCP event control、TP>1 且 TP async prepare 开启时生效。
TP1、文件控制 legacy 和未开启 TP async prepare 的路径保持原样。

## 授权链

1. rank0 的 PREPARE 已选择同一 snapshot；scheduler 注册 attempt-scoped
   progress context，仍通过原生 TP workset plan 分配完整 parent+suffix。
2. 各 rank 的原有后台准备完成后，ledger 记录已 attach 的 lease 和 prepare ACK。
3. Host worker 在本地 pushed mirror 中验证：完整 prepared rank 集合、完整
   claims、相同 owner/claim/read epoch，且本地真实 broker lease 对象、ID、owner、
   io_reserved 状态和 io_attempt 均与冻结 context 相符。
4. 当前 context 未取消/替换/retire 才打开既有 `start_allowed`，直接进入原有
   `mark_io_inflight → ledger phase CAS → RemoteHost read`。不新增状态机、线程、
   collective、文件访问或 Forward 内的同步 RPC。

`H2D_LOADING` 本身不是全组准备完成证据：第一个 recovery claim 就可能设置
这个状态。因此明确检查 `h2d_prepared_ranks`，不能只看 state。

## 失败与生命周期

- 未准备好、容量不足：保持 Host pin 和原有有限 lane/workset，等待原生分配；
  不启动任何未获得完整空间的 copy。
- 旧 context、旧 epoch、镜像滞后、未知 ACK：不能授权；控制连接失效沿原逻辑
  fail-closed，不能推断已完成或回收资源。
- 取消与打开 start_allowed 竞争：bool 不是物理 fence。后续 broker mark I/O
  仍拒绝 retiring lease；ledger CAS 和远端 read receipt 继续保护 source Host。
- 某 rank 已 post 后发生 peer failure：等待实际 Future/transport fence，随后走
  原有 all-rank retry/drain 与 native retire；不提前复用目标页。
- 全组复制、native BIND、绑定后 handed ACK、rank0 native ADMIT 完全保留。
  收到 prepare/start 条件不等于完成 KV，也不允许提前进入模型 Forward。
- 退出/断线不强行释放未 fence 的 source/target；原有 shutdown fail-closed 不变。

## 验证与观测

新增测试位于 `test_agentic_tp_host_bind_handoff.py`，已由现有
`validation/check_multinode_cpu.sh` 的 `test_agentic_tp*.py` 纳入门禁。
使用真实 CPU workset broker、Host ledger；物理 DMA 用可控制的 Future 模拟，
不等于真实 GPU/RDMA 或吞吐验收。

覆盖 TP2/8 缺最后 prepare ACK 不启动、无需 scheduler START 的后台启动、
peer 提前进入 inflight、滞后 mirror、旧 context/epoch/lease、控制断线、
授权后取消、post 前/后 peer 失败、真实 fence 后全组 retire、重试清空旧 prepared
记录，以及 TP1/legacy/async-disabled 隔离。

每次成功的既有 `host_completion_timing` 增加三个单次 monotonic 指标：
`prepared_to_start_ms`、`start_to_submit_ms`、`submit_to_copy_ms`。
缺失时间戳输出 NaN，不伪造零耗时。第一项仍含必要的末尾 rank 准备等待；
第二项包含原有 I/O claim CAS，第三项含 I/O worker 排队与远端复制。

八项验收映射：唯一所有者与 TP 原子性由原始 claim/lease/真实 fence/BIND/ADMIT
保持；P→D Direct、P→D Host durable、D→P Host durable 三个释放点不变；
进度解耦只去除重复 native START；父 KV 与显式重算语义不变；必须通过 CPU
生命周期测试和独立审核后才允许 GPU 验证。此文件不宣称 live 性能提升。
