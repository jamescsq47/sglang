# Slow Host-ready 流水线解耦：待实施设计

状态：**只读代码分析与设计，尚未实现。R15 只修 Direct 根因；必须先看 R15
实测，再决定是否实施本方案。** 本文不更改默认容量、deadline、实验参数或 TP1。
设计真源仍为 Slime 的 `AGENTIC_PD_DESIGN_INVARIANTS.md`。

## 1. 要解决的不是复制带宽

R14 的共同日志窗口 09:17:24–09:19:24：220 个工具已在 Host durable 前返回的
snapshot，durable→selected 平均 44.29 秒，p90 47 秒；durable 218、selected
222、释放 218，约 1.82 snapshot/s。它证明存在长期排队，但这两分钟内并非持续
发散。不能把全部 44 秒归因于某一个控制步骤。

相邻窗口的单 rank timing 显示 prepared→start 约 120ms、submit→copy 290ms、
loaded ACK→bound queue 560ms。后者包含等 native BIND 及本地绑定的时间，日志
没有把两者严格拆开，不能宣称 560ms 全是空转。R13 已合并 BIND 和本地 handoff，
R14 已去掉额外 START 依赖，不应再以删除这些已删除的阶段作为新改进。

剩余结构问题：选择、准备意图、逻辑补槽、BIND、ADMIT/CLEAR 多次依赖同一
`recv_requests()` 节拍；Direct 和 Slow 都消费这个 native 控制边界。物理传输
解耦不等于控制侧补充新工作的进度已经解耦。

### R14 补槽证据与测量边界

只读复查窗口 09:16:10–09:21:10（停止实验前 300 秒）：

| 观测 | 结果 | 能说明什么 |
|---|---|---|
| `tp_host_selected` | 524 次，1.747/s | 实际进入恢复管线的速率 |
| 选择时记录的 depth | 5/8：79，6/8：120，7/8：152，8/8：173 | 补槽时始终还有至少4个旧逻辑上下文；不是物理 DMA 同时数量 |
| 出现选择的秒桶 | 184 个，其中79个秒桶包含4次选择 | 补位明显成批；秒级时间戳不能证明同一 native iteration |
| 相邻非空选择秒桶间隔 | p50 1s、p90 3s、最大4s | 存在补位间隙；不是每个 request 的独立等待时长 |
| 保守重建的全空补位间隙 | 至少12段完整1秒，共12/300秒 | 有待恢复工作时，管线并非始终有已选择的未完成 copy |

最后一项的算法：每个 snapshot 从 selected 日志秒开始算占用，直至最后一个
rank 的 `h2d_complete` 日志秒**再加1秒**才算离开；没有完整8个回执的项继续算
占用至窗口末。如此故意扩大占用区间后仍全空的区间，才计入下界。例如
09:19:33选择4项，所有 rank 在09:19:34已报告复制完成，下一次选择为09:19:36，
所以09:19:35–36至少有1秒完整补位空隙。
与D日志交叉后，这12个区间各有至少16个（最多88个）工具已返回、Host已durable、
后来确实被选择的 snapshot 仍未选择；这不是没有后续数据的空管线。该数仅是
可确认的候选下界，不冒充当时 Router/P HTTP waiter 的权威队列长度。

这不是物理 lane 利用率的精确测量：selected 可能尚未取得 lease，copy 日志又
晚于真实 DMA fence。**更不能用这12秒解释全部44秒排队**；44秒包含稳定队列
在有限服务速率下的等待，12秒只是可确认的补位空隙下界。

当前日志没有逐 snapshot 的 native ADMIT/CLEAR 时间戳，因而无法给出
all-copy→ADMIT、ADMIT→CLEAR 的精确耗时或各自损失比例。`host_completion_timing`
证明 handed ACK 完成，不证明全组已经执行 ADMIT；`p_to_d_release` 更不能替代
ADMIT，因为中间还包含 Prefill 与 P→D 交付。depth 分布也是选择时采样，不是
“resident=8 占总时间多少”的统计。

代码能确认三个不同的补槽约束：物理 lane 已由 worker 的真实 fence 立即归还；
逻辑 copying 名额取 scheduler 缓存的全组 `status<2`，只在下一次 native recv
刷新；resident 的本地 credit 在 ADMIT 归还，但 rank0 `active_host` 容量计数还
包含等 CLEAR 的控制记录，而且本轮先补槽、后移除 CLEAR。因此一条已全组
admitted 的记录可能多影响一次补槽判断。这个顺序问题真实存在，但现有日志
不能量化它占全部等待的比例。

## 2. 三种生命周期必须分开

| 对象 | 取得条件 | 归还条件 | 不是它负责的事 |
|---|---|---|---|
| transport lane | rank0 选择同一 snapshot，各 rank 保留有界 I/O 资源 | 本 rank 真正 copy fence 完成、无引用 I/O buffer；组级补槽须有全 rank 对应回执 | 不能代表整个 KV lease 已释放，也不应等待 Prefill |
| 完整 workset 物理所有权 | native TP plan 实际分配 parent+suffix，Host claim/lease 身份一致 | 正常经 Req→P-ready→D/Host durable 交付，或原有 fenced cancel/TP retire | 复制完成不能释放 HBM；最终 ADMIT 也只是换所有者，不释放 HBM |
| native runnable/admit 资格 | 全 rank 已完成 Radix bind、Req 接管及既有成功回执，由 rank0 发统一 ADMIT | 同一 native 边界把 Req 放入原有可运行队列；之后按原 Prefill 调度 | 不是 I/O lane；控制记录 CLEAR 不应额外占用传输名额 |

当前已做到：`_release_quiesced_h2d_lane()` 在物理 copy 完成时归还 lane；
`_finish_tp_host_handoff()` 在 ADMIT 时归还恢复 resident credit，但 workset
仍由 Req 持有。现在的 4 条 lane 和至多 8 个恢复 resident 均保留，**不加 cap**。
已经 ADMIT 的 Req 不再是恢复 resident；不能为了补槽在 ADMIT 前虚报释放。

## 3. 最小完整目标：后台生产命令，native 只执行物理提交

复用现有 `AgenticPHostStagingManager._control_worker()`、Host completion queue、
`SocketTPGroupMailbox`/持久 TP 消息和 Host ledger，迁移现有命令的生产位置，
不增加第二套业务状态机，也不增加后台 allocator/Radix 线程。

1. **只读 waiter 描述入队。** scheduler 接收真实 Req 后，发布不可变描述
   （rid、parent generation、完整 prompt 长度/摘要、原始入队顺序及取消身份）。
   worker 不持有可写 Req。后台使用现有 Host-ready 增量事件与这些描述配对；
   不扫描 NFS，不重复复制整段 prompt，不轮询所有历史记录。
2. **rank0 唯一选择并发 PREPARE。** rank0 的现有 Host worker 在容量回执到达时
   从原队列选择、补槽，复用现有 PREPARE 命令内容与 attempt identity 通过 TP
   消息下发。followers 只安装描述并执行同一 prepare；不自行选择 snapshot。
   组内命令应有界、去重且有执行确认，不能覆盖尚未被全组消费的命令。
3. **claim/准备在后台，物理分配仍 native。** 各 rank 沿用 Host claim→请求
   workset intent；只有全组相同身份的准备描述能进入 rank0 的不可变 workset
   plan。下一次已有 native boundary 对同一 plan 实际分配一次并报告 grant。
   “一次 plan”指本次 grant 不再反复申请/重新分配；活跃 lease 仍必须按原协议
   留在冻结计划里直至 retire，不能为了减少广播直接删掉。
4. **已准备即复制。** 沿用 R14 的 exact claim/epoch/all-prepared 检查，worker
   收到完整 lease 后直接复制；不再插入 native START。真实 copy 完成立即产生
   原 phase2 回执，并归还本地 transport lane。rank0 收到全组回执后可后台
   PREPARE 下一项，不等 BIND/ADMIT/CLEAR 才承认物理 lane 已空闲。
5. **一次必要 native BIND。** 全 rank copy 完成是 BIND 前提；后台只能把此事实
   放入原 completion queue。rank0 在下一已有 native boundary 批量下发 BIND，
   scheduler 本地执行 Radix bind+Req handoff。仍需原异步全组 bound/manifest/
   handed 提交，不能根据一张卡完成就进入 Forward，也不能在 worker 改 Radix。
6. **最终统一 ADMIT。** 全组成功后由 rank0 在已有 native boundary 统一 ADMIT。
   所有 rank 同顺序成为 runnable；恢复 resident credit 此时归还。CLEAR 只做
   原控制记录的幂等清理，不再让已全组 ADMIT 的记录占住恢复 credit 等下一轮。

仍不可消除的等待：真实容量不足、native allocator 安全边界、全 rank copy/
bind 成功、最终统一 ADMIT。目标是去除这些边界之间纯观察/通知的重复等待，
不是承诺完全不受正在执行的 Forward 影响。

## 4. 具体替换点，不保留双写者

| 当前代码位置（以函数名定位） | 拟替换/删除的责任 | 必须保留 |
|---|---|---|
| scheduler `_agentic_tp_prepare_admission_control()` 的 Host selection、`copying < depth`、`resident_count` | 把 Host 选择/补槽唯一写者移到 rank0 后台；不再每轮以旧 `group_status<2` 推断物理 copy 名额 | native workset plan、BIND/ADMIT 的统一顺序与 bounded batch |
| `active_host` 同时作为 protocol registry 和容量计数 | 用既有 exact completion/admitted receipts 区分 transport/resident；已结束的控制记录不再算容量 | 旧 attempt 记录保留至 CLEAR，不能先丢身份/取消证据 |
| `_queue_host_prepare()` 只能由 native PREPARE 后的 gate 进入 | 复用其 worker-private 描述构造与 lane 规则，由收到的 rank0 PREPARE 安装；移除该模式下逐 native 阶段重入 | 去重、Host pin→workset intent 顺序、原始等待顺序 |
| `_agentic_tp_reduce_host_status()` 对后台已发布 phase2/4 再扫描 Req | TCP 模式消费既有 pushed exact 完成事实，scheduler 只补充它独占的 bind/admit/cancel 完成 | cancel 的优先级、失败黏性、旧 epoch 不覆盖新状态 |
| `_drain_agentic_kv_waiting_queue()` 中反复遍历 prepare/start | TCP 新模式只消费 native BIND/ADMIT 与普通计算 admission；不靠遍历推进复制 | 全组同顺序、普通请求与恢复请求无来源优先级 |

这里只迁移责任，不能让 scheduler 和 worker 同时改 `active_host`/attempt serial。
native control 包继续携带统一计划；后台事件是输入事实，不另开一个会与模型
collective 交错的广播。已有 START 可保留兼容解析，但不作为新模式启动门槛。

## 5. 取消、失败与退出：沿用哪些 fence

| 发生位置 | 原有安全路径与保留资源 |
|---|---|
| 未选择/未 claim | 取消不可变 waiter 描述及旧命令；无 HBM，不凭 timeout 删除 Host |
| pinned/leased、尚未 post | exact claim/lease/epoch；取消准备 Future、发布 unstarted remote read receipt，`cancel_io_attempt`，再 native TP retire |
| 某 rank 已 post | 先真实 Future/NIXL/CUDA fence；ABORTING/RETRY_PENDING 全 rank drain，包括未启动的 rank；不得凭命令 ACK 归还 Host/HBM |
| copy 完成、尚未 bind | lane 可以归还；Host claim 与 HBM lease 保留，原 cleanup 确认 quiescence 后退还；旧 completion 不触发新的 BIND |
| 部分 rank BIND/handoff 成功 | 继续 R13 rollback：binding→abort_bind，handed→release_handed 和无 req slot 的 Mamba runtime 清理；共享 Radix 引用保留，suffix 等 TP retire |
| handed 后、ADMIT 前 | 取消压过 late positive；原 exact job 完成/排空后清理，禁止旧 phase4 放行 |
| ADMIT 后 | 已是 Req 的生命周期，沿原 request cancel/P→D 路径；不能被 Host 控制队列再释放一次 |
| 断线、未知 RPC 结果、队列溢出、退出 | fail-closed；保留精确 attempt 与可能在途的所有权，不空状态重启、不重发新 op ID 猜测成功 |

一条消息“已收到”不等于执行完成；transport 补槽要对应真实全组 copy receipt。
resident 补槽必须对应全组已执行 native ADMIT/原安全终态，不能仅依据发送 ADMIT。
重试复用 snapshot 时必须沿原新 epoch/context 失效规则，旧 ready 位不能继承。

## 6. 实施难度与验收门槛

这不是一个安全的十几行开关修改。要同时迁移 rank0 选择的唯一写者、准备命令
投递、scheduler intent 消费和 credit 归还；只移动一处循环会产生双写、误补槽
或 follower 尚未取得描述就执行计划的竞态。应实现一个完整且有界的 TCP TP>1
切片，TP1/legacy 保持原链；不能以未经验证的 fallback 混用两条控制链。

实施前先确认 R15 Direct 修复能否显著降低 Slow 到达率。当前 ~50% Direct 成功
给 Slow 带来的额外输入本身会维持队列，不能把所有排队都算作 Host 独立缺陷。

建议顺序：先以 R15 同窗口确认 Direct、Slow 到达/选择速率和队列年龄变化；
如果仍需要 Host 改造，最小完整架构范围是**选择/prepare/refill 的唯一后台
producer + exact 物理/恢复 credit 回执 + native intent/BIND/ADMIT 消费**。
三者要一起闭合，不仅挪一个计数器或在每处再加一次 poll。可先消除明确的
“已全组 ADMIT、仅待 CLEAR 却仍计入容量”顺序延迟作为独立安全检查，但不能
预先宣称它能解决40秒排队。是否把完整 producer 迁移实施，必须由实测决定。

必须新增 TP2/8 故障测试：PREPARE 重复/延迟、lease plan 晚到、最后 rank 缺失、
copy 完成与 refill 同时发生、取消与补槽竞争、partial BIND、ADMIT 尚未全组执行、
CLEAR 晚到及重试旧消息。断线时不得增发 credit；真实 allocator/共享前缀/Mamba
资源守恒必须验证。独立审核 GO 后再用同配置 c128 运行。

验收看 **Host-ready→selected、selected→grant、copy→BIND、handed→ADMIT、
physical lane 空闲但逻辑 credit 仍占用的时间**，以及整组吞吐/复用正确性。
不以增加 lane、resident 上限、延长期限或改变数据顺序掩盖问题。
