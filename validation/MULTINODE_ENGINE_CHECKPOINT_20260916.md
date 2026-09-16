# 多节点实际接线检查点：2026-09-16

本检查点取代早期“只有 transport library、尚未接入”的状态说明。
开发范围仅为 `dualpd/sglang` 和 `dualpd/slime`，没有修改原仓库或运行 GPU 实验。

## 实现范围

- 多节点必须显式启用；关闭时保留原单节点数据路径。
- 快路径沿用 NIXL HBM→HBM。慢路径两方向均为源 GPU→源节点 memfd DRAM→
  NIXL/UCX READ→目标 GPU，控制文件中只传描述符和完成回执，不传 KV payload。
- D→P 使用 D 本地 Arena；P→D 使用 P 本地 Arena。D/P 的既有 D2H fence 和
  TP durable 屏障完成后释放源 GPU，Host 副本等全组目标接管后才释放。
- 恢复仍使用完整 workset、现有 Radix bind、TP claim/commit；不添加来源优先级。
- 独立 Host I/O worker/transport agent，不把网络注册和 READ 放入 Forward。
- P→D GPU 页索引通过 producer CUDA event 交给后台；D→P 复用 workset broker
  发布前已完成的 CPU 页索引镜像，避免跨 CUDA stream 读取尚未生成的索引。
- 未完成 READ、部分 TP 失败、注册异常、丢 ACK 均保留或隔离物理所有权；不能
  因超时直接释放内存。源释放的全组 ACK 完成前不 prune ledger。
- V1 单逻辑 P 组，支持一个或多个 D 组；相同 TP=1/2/4/8，整组同节点。
  先支持 dense Qwen3 MHA；Mamba/MLA/混合模型与跨主机 TP 集体计算明确拒绝。
- native HiCache/Mooncake/offload 禁用；源节点 Host 阈值、Q 拥堵反馈重算等沿用
  既有策略，不把 CPU 测试当作 TP8 参数调优。

## 审核与测试

Host、P→D/transport、Router/launcher 三部分互相独立审查，结论为：允许实验性
代码集成和远端 smoke，不代表真实 RDMA、吞吐或长期运行验收。

可重跑入口：

```bash
DUALPD_PYTHON=/path/to/pd_multi_node/bin/python bash validation/check_multinode_cpu.sh
```

测试覆盖完整 shard、重复/陈旧回执、取消与物理 fence、失败注册隔离、未启动 rank
回滚、全组 Host 释放、源分配失败、TP peer 容量压力、远端路径不得 mmap、既有
单机 TP/生命周期/Router 回归，以及 launcher/supervisor。GPU 用例在 CPU 模式跳过。
本轮统一入口结果：741 passed、2 个 CUDA 测试 skipped；另有 37 个 launcher/smoke/
supervisor unittest 通过，Bash 语法和 `git diff --check` 通过。

## 八项不变量复核

| 项目 | 代码依据 | 尚需实机验证 |
|---|---|---|
| 唯一所有者 | 原 ledger CAS + node/engine/export/read epoch | 多请求所有权计数守恒 |
| P→D Direct 释放 | 沿用既有全组 NIXL commit | 跨机 GPU fence |
| P→D Host 释放 | 全组源 D2H durable 后沿用释放 | 压力下 P HBM及时回落 |
| D→P Host 释放 | 全组 D2H+descriptor durable | 慢工具不占 D HBM |
| 独立进度 | 原 I/O worker、独立 Host NIXL agent | 计算/传输重叠与控制 FS 延迟 |
| TP 原子性 | rank0 路由、逐 shard receipt、全组 commit | TP8 partial failure/cancel |
| 父 KV 正确性 | MHA layout/页映射检查、完整 workset、Radix bind | 两轮 KV/输出与完整重算对照 |
| 修改门禁 | CPU 回归+独立交叉审核 | 用户在远端先 smoke 再长测 |

## 远端交接

入口在 Slime `tools/dualpd/multinode.sh`，详见同目录 `MULTINODE.md`。
先编辑示例 JSON、校验共享控制目录，再启动两个节点的 worker、Router，执行 smoke。
共享 POSIX 元数据是最小兼容方案，需要跨节点锁与时钟同步；不宣称 NFS 延迟足以
满足 1 秒 Direct deadline。全量实验前还需强制 P→D Host 背压、多请求并发及 TP8
异常恢复验证。当前没有新的多节点吞吐数字。
