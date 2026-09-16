# 多节点开发阶段检查记录（2026-09-16）

状态：**基础模块及预检查已完成；完整多节点推理尚未接通，不可启动正式实验。**

范围仅 `dualpd/sglang` 和 `dualpd/slime`。SGLang 基线为
`89af4ab56dede8705a444d503040da829a9c6f50`；Slime 基线为 `d2df7f1`
（原 main 的独立开发分支，附既有硬件采集工具）。不修改原仓库、原环境或根目录工具。

## 已交付

- 默认关闭的多节点配置校验；TP=8 表示 P/D 各自一台机器内8卡组成一个组，等TP、DP1、PP1。
- 现有 NIXL GPU Direct 网络能力审查：无需另写 GPU 数据传输协议，但未实测 UCX/GDR。
- 共享控制目录的后台 watcher 轮询分支：远端写入不依赖本机 inotify；默认本机行为不变。
- 独立 NIXL `DRAM -> remote VRAM READ` 库：完整 shard、layout、token count、export epoch、
  read attempt 校验；失败保持物理 fence；全部TP确认且目标绑定提交后方可释放Host。
- 有界 READ registry：防止异常/GC丢失在途handle，同一attempt幂等，未知DMA不得退休。
- 源注册失败清理的quarantine：注销失败必须继续保留mapping，不能释放仍可能使用的内存。
- Slime Bash入口、TP8示例配置、元数据O_EXCL/hardlink/rename可见性和跨主机flock探针。

## 本地验证

使用 `pd_multi_node` 环境，`CUDA_VISIBLE_DEVICES=''`，禁用pytest第三方自动插件。
未运行任何GPU或跨节点网络实验。

- 联合测试 **518 passed**，17条既有CPU/弃用warning：新增配置、watcher、remote Host，
  加既有lifecycle、TP、Host async/event、slow congestion和Slime early-claim回归。
- Slime配置/预检查 **16 passed**；涵盖生成的环境与引擎配置校验的跨仓库契约。
- Bash语法与 `git diff --check` 通过。
- 独立子agent审计：**GO，仅限离线基础模块/预检查**；完整运行 **NO-GO**。
  审计修正了混合TP read attempt、把DMA receipt误当作目标ownership commit的风险。

## 未完成且必须完成后才能启动

1. D→P改为D源节点Arena分配，不再取得P本机memfd grant。
2. P→D接收和D→P恢复：按源node选择本地mmap或远程READ，并完整接入原workset/Radix/aux metadata。
3. Arena eviction pin、源进程生命周期、全TP commit/取消、断链/丢ACK的权威状态接线。
4. Hybrid/Mamba具体state布局接线（不能只有attention KV）。
5. 真正的worker/router/harness启动管理、全体ready屏障及只回收自身进程的supervisor。
6. 用户远端的共享控制语义、GPU KV逐shard一致性、UCX/RDMA及吞吐验收。

历史检查点：此处描述最初尚未接线的阶段，后续已实现引擎接线；当前状态见
`MULTINODE.md`。这里的 `integrated=false` 是当时的值，不是当前能力声明。
不能因CPU测试通过就修改此标记或绕过门禁。后续实现详见
`MULTINODE_INTEGRATION_PLAN.md` 和 `MULTINODE_HOST_DATAPLANE.md`。
