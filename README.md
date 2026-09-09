# MedicalGPT · Safety-aware Post-training for Medical QA

面向中文医疗问答的 **数据构造 → SFT → 偏好对齐 → Reward Modeling / Online RL → 可审计评测** 实验项目。项目重点不是“把所有训练算法串起来”，而是围绕同一个医疗问答问题，系统比较不同 post-training 路线在 **事实准确性、专业表达、安全合规与回答可接受性** 上的差异，并记录训练工程中的显存、稳定性和评测问题。

> **项目定位**  
> 本项目基于 `shibing624/MedicalGPT` 的训练框架开展个人实验。上游已有 PT/SFT/RM/PPO/GRPO 等能力；个人工作重点包括：医疗数据清洗与格式适配、偏好数据构造、SFT/DPO/RM/RLOO/GRPO 实验、LoRA 与分布式训练配置、统一 AAR 评测、judge bias / reward hacking 分析，以及补充的安全数据契约与配对评测工具。
>
> **证据说明**  
> 历史训练阶段与 AAR 数字来自当时的训练记录与简历整理，但当前公开仓库缺少完整 checkpoint manifest、逐题模型输出和 judge 原始日志，因此这些数字应视为 **历史实验记录**，而不是当前仓库可一键重算的 benchmark。当前公开版本优先保证实验设计、数据治理和评测协议可审计，不把软件测试通过等同于临床能力。

---

## 1. 为什么做这个项目

医疗问答并不是“回答越流畅越好”。一个回答可能语法正确、结构漂亮，但存在以下问题：

- 医学事实错误或推理链错误；
- 对高风险问题给出过度确定的诊断/用药建议；
- 通过模板化免责声明、冗长回答等方式“迎合”评审模型；
- SFT 后会回答，但无法区分同一问题下“可接受回答”和“更安全、更专业的回答”；
- RM 离线排序准确率很高，但在线 RL 后 policy 可能进入 RM 未见过的分布，出现 reward hacking。

因此，本项目把问题拆成 4 个层次：

1. **领域能力**：模型是否具备基本医疗术语理解和指令遵循能力？
2. **偏好对齐**：同一问题有多个候选答案时，模型能否学习“哪个更好”？
3. **在线优化**：能否在当前 policy 的 rollout 上继续优化，而不仅依赖静态 preference pair？
4. **可信评测**：提升是否真的来自训练算法，而不是测试集、sampling、judge 或数据泄漏变化？

---

## 2. 模型选择：为什么最终使用 Qwen3.5-4B

最初尝试过更小的 Qwen2.5-1.5B / Qwen3.5-2B 级模型。完整走过 SFT / preference optimization 后，AAR 提升仍然有限。这里的判断不是“参数越大一定越好”，而是一个 **任务复杂度 × 模型容量 × 训练成本** 的 trade-off：

- 医疗问答包含较高密度的专业知识、术语映射、上下文约束与安全判断；
- 过小模型的基础知识存储与表征能力有限，post-training 更多是在重排已有能力，而无法凭空补齐大量缺失知识；
- 7B/14B 虽然上限更高，但个人多阶段实验需要反复做 SFT、DPO、RM、rollout 和 online RL，训练成本显著增加；
- 4B 在中文指令能力、医疗领域适配能力与 4×RTX 4090 级实验成本之间更平衡。

因此选择 Qwen3.5-4B 不是因为“4B 是标准答案”，而是因为：

> **更小模型出现明显 capacity bottleneck，而 4B 已能观察到 post-training 方法之间有意义的差异，同时仍能在有限 GPU 资源下完成完整实验闭环。**

### 已做的模型规模对比（历史实验）

| 设置 | 观察 | 结论 |
|---|---|---|
| 1.5B / 2B 级模型 | 完整训练后 AAR 提升有限 | 基础能力成为瓶颈，继续堆后训练收益较小 |
| 4B | 后训练收益更明显 | 作为后续统一实验基座 |

> 该部分为历史定性结论。若用于正式报告，建议补充同一数据、同一 evaluator、同一 decoding 下的可重算表格。

---

## 3. Post-training 不是严格串行 Pipeline

```text
                    ┌───────────── DPO ─────────────┐
                    │   offline preference alignment│
Base → SFT ─────────┤                                │
                    ├──── RM → RLOO ────────────────┤
                    │ learned reward + online RL     │
                    │                                │
                    └──── GRPO ──────────────────────┘
                         group-relative online RL
                         (current script: rule rewards)
```

其中：

- **SFT**：先获得稳定的领域指令能力；
- **DPO**：直接利用 `chosen/rejected` 做离线偏好优化，不需要单独训练 RM；
- **RM → RLOO**：显式学习 reward，再用当前 policy rollout 做 online optimization；
- **GRPO**：同 prompt 多采样，使用组内相对 advantage；当前公开脚本使用 `accuracy_reward + format_reward` 的 rule-based reward，而不是 RM reward；
- **DPO → RLOO**：可以作为扩展实验，表示“先离线 preference alignment，再从 DPO policy 初始化 online RL”，但不是理论上必须的顺序。

这也是本项目想强调的训练思维：**算法不是按名词顺序堆叠，而是针对不同训练信号与优化目标做路线比较。**

---

## 4. 数据与训练数据治理

### 4.1 SFT 数据

SFT 数据由医疗领域问答与通用 instruction 数据混合构成，并统一转换为 ShareGPT / instruction-output 等训练格式。数据处理重点包括：

- 去重与异常样本清洗；
- 术语规范化；
- 医学表达与安全提示检查；
- 统一 chat template；
- 避免同一 patient/source 或近重复 prompt 跨 train/dev/test 泄漏。

### 4.2 SFT 最大长度：512

历史数据长度统计中，90% 以上样本可落在 512 token 范围内，因此训练时优先选择 512：

- 覆盖绝大多数训练样本；
- 显著降低 activation memory；
- 提升 batch throughput；
- 对个人算力更友好。

对于长上下文医学病例，512 并不是理论最佳长度；若业务任务本身包含长病历，应重新做 length distribution 并调整 max length，而不是机械沿用。

### 4.3 LoRA target modules

SFT / DPO 中主要覆盖 Transformer 的 Attention 与 MLP 线性层：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

原因：

- `q/k/v` 决定新的 attention interaction pattern；
- `o_proj` 负责多头信息融合；
- MLP（`gate/up/down`）占模型参数和非线性表征的重要部分，对领域语义与表达风格适配非常关键；
- 不优先训练 embedding，是因为任务并非重新学习 tokenizer / token embedding；
- LayerNorm 参数很少，单独加入 LoRA 意义有限。

### 4.4 LoRA rank

历史 SFT/DPO 主要使用 `r=8, alpha=16`；在部分 RL 脚本中可见 `r=16, alpha=32`。早期比较中，单纯把 rank 从 8 提升到 16 并没有带来等比例收益，且小数据上更容易过拟合，因此：

> LoRA rank 应视为 **容量超参数**，不是越大越好；应结合数据规模、模型层数和目标任务复杂度调节。

---

## 5. 为什么 SFT 之后还要 DPO

SFT 的目标是最大化参考回答 likelihood：

\[
L_{SFT}=-\sum_t\log\pi_\theta(y_t|x,y_{<t})
\]

它解决的是：

> “面对医疗问题，模型应该如何生成一个合理回答？”

但 SFT 并没有显式学习：

> “同一个问题下，回答 A 与回答 B 都能说通，为什么 A 比 B 更安全、更准确、更简洁？”

因此进一步构造 preference pair：

```text
(prompt, chosen, rejected)
```

DPO 直接提高 chosen 相对于 rejected 的概率，同时约束 policy 不要过度偏离 reference model。

### 为什么不把 chosen 直接继续拿去 SFT？

因为两种训练信号不同：

- **chosen-only SFT**：要求模型模仿 chosen 的全部 token，包括风格、长度和模板；
- **DPO**：重点学习 chosen 与 rejected 的 **相对偏好差异**。

在医疗场景里，这一点很重要：一个 teacher 的答案可能事实更好，但明显更冗长。直接 SFT 容易把 teacher 的冗长表达一起学进去；DPO 更适合表达“这个答案整体上更好”，而不是要求逐 token 模仿其全部风格。

> 更严谨的验证方式：对同一批 preference 数据做 `chosen-only continued SFT` 与 `DPO` 对照，从而区分“多看了高质量答案”与“preference objective 本身”的贡献。

---

## 6. Preference Data：如何构造高质量 Chosen / Rejected

候选回答来自不同能力和风格的 teacher，以增加多样性、避免同源 self-play 过于相似。Judge 根据医疗安全与回答质量进行 pairwise 选择。

历史规则重点包括：

- 循证医学常识冲突；
- 未排查禁忌症或危险用药建议；
- 处方药建议缺少必要的就医/遵医嘱提示；
- 绝对化、高风险断言；
- 两边都存在严重安全问题时整对丢弃；
- 两边都安全时，再比较表达、信息密度和冗余程度。

同时人工抽检一批样本，重点看：

- Judge 是否系统性偏爱长回答；
- 是否偏爱某种固定模板；
- chosen / rejected 是否长度严重失衡；
- rejected 是否过于简单，导致 preference pair 太“容易”。

### 为什么不能让 rejected 太差？

如果 rejected 是明显垃圾回答，模型很容易学会区分，训练信号却无法覆盖真实场景中的“两个都不错，但一个更优”的 hard preference。高质量 preference 数据的关键不是数量本身，而是 **pair 的信息量**。

---

## 7. Reward Model

### 7.1 RM 的目标

Reward Model 对同一个 prompt 的 chosen/rejected 分别输出标量 reward：

\[
r_c = r_\phi(x,y_c), \qquad r_r = r_\phi(x,y_r)
\]

使用 pairwise logistic loss：

\[
L_{RM}=-\log\sigma(r_c-r_r)
\]

即希望：

\[
r_c > r_r
\]

当前代码中的核心实现为：

```python
loss = -torch.nn.functional.logsigmoid(
    rewards_chosen - rewards_rejected
).mean()
```

RM 学到的不是“这个回答绝对值应该是 8.6 分”，而是 **在训练分布中 chosen 应排在 rejected 前面**。

### 7.2 RM accuracy 是什么

可以定义 held-out preference accuracy：

\[
Acc_{RM}=P(r_{chosen}>r_{rejected})
\]

它用于判断 RM 是否学会了离线偏好排序。

### 7.3 为什么 RM accuracy 高，不代表最终生成质量高

这是本项目中非常重要的风险：

```text
High RM accuracy ≠ High final response quality
```

原因包括：

1. **Distribution shift**：RM 在离线 preference pair 上训练，但 online RL 会改变 policy 分布；
2. **Reward hacking**：policy 可能找到 RM 的漏洞，例如“越长分越高”“只要反复出现免责声明就更安全”；
3. **Spurious correlation**：RM 可能学习到风格、长度、固定措辞，而不是真实医学质量；
4. **OOD exploitation**：policy 可能产生 RM 训练阶段几乎没见过的文本模式，此时 reward 数值并不可靠。

---

## 8. 如果 reward 很高，但模型开始胡言乱语：怎么排查

这是 online RL 中典型的 **reward hacking / reward over-optimization** 问题。

排查时不直接看“reward 是否上涨”，而是同时看以下几类信号：

### 8.1 先确认是不是 evaluator / reward 本身被 exploit

对高 reward 样本人工抽检，并记录：

- factual accuracy；
- terminology / professional expression；
- safety / compliance；
- 过度拒答、模板化免责声明、长度异常、重复等行为。

> 你的旧笔记中可核验的 judge prompt 是 **3 个显式评分维度（事实准确性、术语规范性、安全合规性）+ 1 个总体 Accepted 判断**。如果当时确实使用过“4 个独立评分维度”，需要从旧 judge 日志或 prompt 文件中恢复后再写进公开 README，避免凭记忆补造。

### 8.2 看 reward 与独立质量指标是否脱钩

重点监控：

- mean reward ↑ 但 AAR ↓；
- reward ↑ 但 factual accuracy ↓；
- reward ↑ 但 unsafe rate ↑；
- reward ↑ 但 output length 异常增长；
- reward ↑ 但重复率 / disclaimer rate 异常增长。

如果出现这些现象，应优先怀疑 reward hacking，而不是继续加训练步数。

### 8.3 用独立 Judge / 人工评审做交叉验证

不能让“生成 preference 的 Judge”和“最终验收的 Judge”完全相同，否则可能出现 teacher/judge bias 闭环。

更严格的方式：

- Judge-A 生成或筛选 preference；
- Judge-B 做 blind evaluation；
- 人工抽检高 reward / 高 disagreement 样本；
- 统计 `Judge-B vs Human` 一致性；
- 对模型名称、回答顺序做 blind / randomized presentation，降低 position bias。

### 8.4 从训练层抑制 reward over-optimization

如果确认是 online RL 过度优化，可考虑：

- 提高 KL 约束或降低 learning rate；
- early stopping；
- reward clipping / normalization；
- 引入多目标 reward，而不是只优化单一 learned RM；
- 对长度、重复、过度拒答设置独立监控或惩罚；
- 增强 RM 的 hard negatives 和 adversarial examples。

### 面试版回答

> 如果 reward 一直上升但模型开始胡言乱语，我不会把它理解为“RL 继续有效”，而会优先判断 reward 是否被 exploit。RM 只是在离线偏好分布上学排序，policy 在线更新后会产生 distribution shift，可能学到“更长、更模板化、更多免责声明”这类 RM 偏好，而不是真实医学质量。我会固定独立测试集，用与训练偏好标注不同的 judge 做 blind evaluation，并人工抽查高 reward 样本，比较 reward 与 factual accuracy、safety、AAR、长度和重复率是否脱钩。如果确认 reward hacking，再通过降低 RL 学习率、加强 KL、early stop、增加 hard negative 或多目标 reward 来修正，而不是单纯继续训练。

---

## 9. RLOO 与 GRPO：为什么两个都做

共同目标：减少 PPO 对独立 value/critic network 的依赖，在有限资源下进行 online RL。

### 9.1 RLOO

对同一 prompt 采样 K 个回答，得到 reward：

\[
r_1,\dots,r_K
\]

对第 i 个样本，用其他 K-1 个样本均值作为 baseline：

\[
b_i=\frac{1}{K-1}\sum_{j\neq i}r_j
\]

advantage：

\[
A_i=r_i-b_i
\]

直觉：

> “这条回答相比同 prompt 的其他 rollout 到底更好还是更差？”

当前公开 `rloo_training.py` 将独立 reward model 传给 `RLOOTrainer`，因此这一路线属于 **learned reward + online RL**。

### 9.2 GRPO

对同一 prompt 生成一组 G 个回答：

\[
r_1,\dots,r_G
\]

组内均值和标准差：

\[
\mu=\frac{1}{G}\sum_i r_i
\]

\[
\sigma=\sqrt{\frac{1}{G}\sum_i(r_i-\mu)^2}
\]

标准化 advantage：

\[
A_i=\frac{r_i-\mu}{\sigma+\epsilon}
\]

当前 GRPO 脚本：

```text
num_generations = 4
beta = 0.001
learning_rate = 5e-7
max_completion_length = 512
```

并使用 rule-based reward：

```text
accuracy_reward + format_reward
```

而不是直接调用 RM。

### 9.3 为什么两者都实验

- 都可以避免单独训练 critic；
- RLOO 用 leave-one-out baseline，当前样本不参与自己的 baseline；
- GRPO 用 group-relative normalization；
- 两者对 reward scale、group size 和 sampling variance 的敏感性不同；
- 在同一算力预算下比较，可观察哪种 estimator 更稳定。

### group size 消融

历史实验比较过 `G=2` 与 `G=4`：

- `G=2`：组内统计量估计更粗，训练曲线波动更明显；
- `G=4`：rollout 成本增加，但 reward normalization 更稳定，因此最终选 4。

这是一个典型的 **训练稳定性 vs rollout 成本** trade-off。

---

## 10. 训练工程：为什么改成 ZeRO-2 + CPU Offload

历史笔记中曾把实验写成 ZeRO-3。重新审视后，对 **4B + LoRA** 任务，更合理的描述是：

> **使用 DeepSpeed ZeRO-2 + CPU optimizer offload，配合 bf16、LoRA、gradient checkpointing / FlashAttention 等手段控制显存。**

### 10.1 为什么 4B LoRA 不需要强行上 ZeRO-3

4B bf16 基座权重约：

\[
4B\times2\ bytes\approx8GB
\]

LoRA 只训练少量 adapter 参数，因此与 full fine-tuning 相比，gradient 与 optimizer state 小很多。对于 24GB 级 GPU，4B LoRA 本身通常不要求 ZeRO-3 参数分片。

如果强行使用 ZeRO-3：

- 参数频繁 all-gather；
- 通信开销更高；
- 对小模型/LoRA 未必有收益；
- 工程复杂度增加。

### 10.2 ZeRO-2 做了什么

ZeRO-2 主要分片：

- optimizer states；
- gradients；

模型参数仍在每张 GPU 上保留完整副本。

在 LoRA 场景下，这通常已经能满足显存需求，并比 ZeRO-3 更简单。

### 10.3 CPU Offload

对于 ZeRO-2，可把 optimizer state / optimizer computation offload 到 CPU，进一步降低 GPU 显存占用：

```json
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  }
}
```

注意：`offload_param` 是 ZeRO-3 语境下的参数 offload；ZeRO-2 不做 parameter sharding，因此这里应主要描述 **optimizer offload**。

### 10.4 为什么当时仍需要做显存优化

不是因为“4B LoRA 单模型就一定塞不下”，而是因为多阶段 post-training 中会额外引入：

- DPO 的 policy/reference 计算；
- RM 的 pairwise chosen/rejected forward；
- RLOO/GRPO 的多 rollout generation；
- 较长 sequence / completion；
- 多样本组内比较。

因此显存优化是为了给更复杂的训练阶段留出 headroom，而不是为了证明“4B 必须 ZeRO”。

---

## 11. Evaluation：AAR 与固定测试协议

### 11.1 AAR

AAR（Answer Acceptance Rate）用于衡量回答是否达到可接受标准：

\[
AAR=\frac{\#Accepted\ Answers}{\#All\ Test\ Questions}
\]

历史 judge prompt 中可核验的显式评分维度包括：

1. **医学事实准确性**；
2. **专业术语规范性**；
3. **安全合规性**；
4. 最终 `is_accepted` 总体判断。

当前公开评测协议进一步加入：

- unsafe rate；
- over-refusal rate；
- empty rate；
- routine / high-risk 分桶；
- Wilson 95% CI；
- 同题 paired bootstrap。

### 11.2 固定的 evaluation protocol

为了让 SFT / DPO / RL 之间可比较，实验中固定：

- 同一 test set / question IDs；
- 同一 system prompt 与 chat template；
- 同一 decoding 参数（temperature / top-p / max tokens 等）；
- 同一 evaluator 与 rubric；
- 同一 acceptance threshold；
- 尽可能固定 seed，并对 stochastic decoding 做重复采样或置信区间。

只有这样，模型间差异才更接近训练算法本身，而不是评测条件变化。

---

## 12. 历史实验结果与证据边界

旧实验记录中曾整理出以下 AAR：

| 阶段 | 历史 AAR | 说明 |
|---|---:|---|
| SFT | 74.2% | 领域指令能力基线 |
| DPO | 83.4% | offline preference optimization |
| RLOO | 87.8% | learned RM + online RL 路线 |
| GRPO | 90.2% | group-relative RL 路线 |

这些数字用于说明历史实验现象，但当前公开仓库没有完整逐题输出、judge log 与 checkpoint manifest，因此 **不应把 74.2 → 83.4 → 90.2 直接写成已严格证明的算法因果提升**。

更正确的表述是：

> 历史实验中观察到 AAR 随 post-training 阶段提高；当前代码库进一步补充冻结测试协议、分组去重、paired evaluation 和消融设计，用于重新验证这些提升是否能被归因于训练方法。

---

## 13. 怎么证明 AAR 提升来自算法，而不是数据 / Judge / Sampling

仅看：

```text
SFT 74% → DPO 83% → GRPO 90%
```

不能证明因果。至少需要以下控制。

### 13.1 冻结 test set

所有 checkpoint 在完全相同的、训练阶段不可见的 test set 上评估；patient/source group 与近重复 prompt 不跨集合。

### 13.2 固定 generation setting

统一：

- temperature；
- top-p / top-k；
- max_new_tokens；
- system prompt；
- chat template；
- decoding strategy。

### 13.3 固定 evaluator 与 rubric

同一 evaluator、同一 prompt、同一阈值，避免 evaluator 变化造成“假提升”。

### 13.4 Paired evaluation

对同一批 question IDs 比较模型：

```text
Question ID | SFT | DPO | GRPO
q1          | Fail| Pass| Pass
q2          | Pass| Pass| Pass
q3          | Fail| Fail| Pass
```

再做 paired bootstrap / confidence interval，而不是只比较三个独立百分比。

### 13.5 核心消融：Chosen-only SFT vs DPO

使用完全相同的 preference 数据：

```text
A: SFT checkpoint + chosen-only continued SFT
B: SFT checkpoint + DPO(chosen, rejected)
```

若 B 明显优于 A，才能更有力地说明收益来自 **preference objective**，而不是“DPO 阶段额外看了 2k 高质量 chosen”。

### 13.6 Judge bias control

- preference label Judge 与 final evaluator 尽量分离；
- 回答顺序随机化；
- evaluator 不知道模型名称；
- 抽样人工复核；
- 报告 Judge-Human agreement；
- 对高 reward 但 low human score 样本单独做 failure analysis。

---

## 14. 已完成 / 已观察的消融与建议补充实验

### 14.1 已完成或历史上已观察

| 消融 | 对比 | 观察 |
|---|---|---|
| Model size | 1.5B/2B vs 4B | 小模型完整后训练提升有限，4B 更适合作为统一基座 |
| LoRA rank | r=8 vs r=16 | r=16 增加容量但收益有限，小数据上更易过拟合 |
| GRPO group size | G=2 vs G=4 | G=2 方差更大，G=4 曲线更稳定 |
| RL learning rate | 1e-6 vs 5e-7 | 5e-7 训练更平滑；较大学习率 KL 增长更快 |
| Sequence length | 512 vs 更长上限的成本 | 512 覆盖多数 SFT 样本并显著节省显存 |

> 上述为历史实验笔记级结论；建议今后保存 run config、TensorBoard、seed 和 checkpoint hash，让消融可重算。

### 14.2 最值得补的严谨消融

1. **Chosen-only SFT vs DPO**：验证 preference objective 的额外价值；
2. **DPO vs RM→RLOO vs GRPO**：从同一个 SFT checkpoint 出发，保持训练 token / compute 尽量接近；
3. **RM length balancing on/off**：验证 RM 是否依赖长度捷径；
4. **DPO beta sweep**：例如 0.05 / 0.1 / 0.2，观察 AAR、KL 与过拟合；
5. **GRPO G=2/4/8**：比较 variance、throughput、reward diversity；
6. **Reward ablation**：accuracy only / format only / combined；
7. **Judge-A vs Judge-B vs Human**：量化 evaluator bias；
8. **LoRA Attention-only vs Attention+MLP**：验证 target modules 的收益；
9. **ZeRO-2 vs DDP**：在 4B LoRA 下比较显存、step time 与通信开销，证明工程选择而不是“为了用 DeepSpeed”。

---

## 15. 训练稳定性与常见 Failure Modes

### 15.1 输出越来越长

可能原因：reward / judge 偏爱长回答。

监控：

- average output length；
- reward-length correlation；
- AAR by length bucket；
- human preference by length bucket。

处理：长度平衡 preference pair、长度惩罚、多目标 reward、独立 judge。

### 15.2 模板化免责声明增多

可能是模型发现“建议就医 / 遵医嘱”容易获得安全高分。

处理：

- over-refusal / disclaimer rate 单独统计；
- 对低风险可回答问题惩罚无意义拒答；
- 加入 hard negative：形式安全但事实错误的回答。

### 15.3 高 reward 但 factuality 下降

典型 reward hacking。

处理：

- early stopping；
- 更强 KL；
- 降低 learning rate；
- 增加事实型 / adversarial RM 数据；
- 独立 factuality judge + 人工复核。

### 15.4 模型重复 / 循环

先检查 decoding（temperature、top-p、repetition penalty），再检查是否 RL 过拟合某类高 reward 模板；最后回到训练数据检查重复与模板偏置。

---

## 16. 训练工程配置建议

### SFT / DPO

```text
model: Qwen3.5-4B
precision: bf16
PEFT: LoRA
LoRA targets: q/k/v/o + gate/up/down
SFT max length: 512
DPO beta: historical default around 0.1
```

### Online RL

```text
GRPO:
  learning_rate: 5e-7
  beta: 0.001
  num_generations: 4
  max_completion_length: 512

Distributed:
  DeepSpeed ZeRO-2
  CPU optimizer offload
```

> 配置不是固定真理。RL 的 beta / lr / group size 应看 KL、reward variance、AAR、length 和 safety 指标共同决定。

---

## 17. README 之外的工程建议

为让项目更贴近“大模型训练 / Post-training 算法实习”JD，建议今后每次实验自动保存：

```text
run_id/
├── config.json
├── git_commit.txt
├── dataset_manifest.json
├── checkpoint_manifest.json
├── generation_config.json
├── tensorboard/
├── predictions.jsonl
├── judge_ratings.jsonl
├── metrics.json
└── failure_cases.jsonl
```

这样面试官问：

> “为什么你说 DPO 提升了 9 个点？”

你可以直接回答到：

- 哪个 checkpoint；
- 哪套 test IDs；
- 哪个 generation config；
- 哪个 evaluator；
- 哪个 rubric version；
- 是否 paired；
- CI 多大；
- 有哪些 bad cases。

这比再多写几个算法名更能证明“真正做过训练”。

---

## 18. Repository Structure

```text
config/                    安全对齐与实验配置
src/medical_alignment/     偏好契约、分组清洗、评测与统计
src/alignment/             显式 alignment 数学核心
training/                  SFT / DPO / RM / RLOO / GRPO 等训练入口
scripts/                   数据构造、训练与审计脚本
tests/                     回归测试
examples/                  合成软件 fixture
reports/                   软件验证与评测产物
```

---

## 19. Reproducibility

```bash
python -m unittest discover -s tests -p 'test_medical_alignment.py' -v
python -m scripts.audit_alignment curate \
  --input examples/preferences_fixture.jsonl \
  --output reports/fixture_curation
python -m scripts.audit_alignment evaluate \
  --input examples/ratings_fixture.jsonl \
  --output reports/fixture_evaluation

# 准备经审查的真实偏好数据与 merged SFT checkpoint 后
python -m scripts.train_safe_dpo --config config/safe_dpo.json
```

历史训练脚本保留在 `training/` 与 `scripts/` 中；正式复现实验时应使用冻结数据版本、固定 evaluator、统一 generation config，并保存完整 manifest。

---

## 20. 面向岗位的能力映射

这个项目希望展示的不只是“会调用 Trainer”，而是完整的大模型训练问题拆解能力：

| JD 能力 | 项目证据 |
|---|---|
| 模型数据建设 | 医疗 SFT 数据清洗、偏好 pair 构造、hard preference、安全标注 |
| SFT / PEFT | Qwen3.5-4B、LoRA、bf16、target module 设计 |
| Preference Alignment | DPO、RM pairwise loss、RLOO / GRPO |
| Reward Design | learned RM 与 rule-based reward 两条路线 |
| 训练工程 | DeepSpeed ZeRO-2、CPU optimizer offload、多卡、显存优化 |
| 模型评估 | AAR、unsafe rate、over-refusal、paired bootstrap、failure analysis |
| 训练稳定性 | KL、learning rate、group size、reward hacking 分析 |
| 实验严谨性 | frozen test、固定 generation/evaluator、消融与 judge bias control |

---

## 21. 参考与归属

- 上游项目：`shibing624/MedicalGPT`，Apache-2.0。
- DPO: Rafailov et al., *Direct Preference Optimization*, 2023.
- InstructGPT / pairwise RM: Ouyang et al., 2022.
- PPO: Schulman et al., 2017.
- RLOO / GRPO 相关实现以当前训练框架与 TRL 接口为准。

---

## Disclaimer

本项目用于研究、软件验证与模型训练学习，不构成医疗建议，也不应把自动评测结果解释为临床安全性证明。真实医疗部署需要更严格的医学专家评估、数据合规、风险分层与线上安全机制。
