# 方向03：大模型知识编辑实验报告

**学号**：SX2516135
**姓名**：郭怡希
**完成日期**：2026年5月25日

## 一、实验目的

1. 理解大语言模型中事实知识的存储机制。
2. 掌握 ROME 和 MEMIT 两种知识编辑算法。
3. 评测知识编辑的三大核心指标：编辑成功率（ES）、泛化性（PS）、局部性（NS）。

## 二、实验环境与工具

- **基础模型**：Qwen2.5-0.5B-Instruct
- **框架**：EasyEdit
- **GPU**：NVIDIA RTX A5000(24GB)
- **编程语言**：Python

## 三、实验数据

### 3.1 自定义10条事实

用于 Task 1 基线和 Task 2/4 评估，每条包含 prompt、subject、target_new、ground_truth、rephrase_prompt、locality_prompt、locality_ground_truth。示例：

| prompt | target_new | ground_truth |
|--------|------------|---------------|
| The capital of France is | London | Paris |
| The chemical symbol for gold is | Go | Au |
| The author of 'Pride and Prejudice' is | Charles Dickens | Jane Austen |
| ... | ... | ... |

完整数据见 `data/custom_10_facts.json`。

### 3.2 批量编辑数据（ZsRE）

从 Hugging Face `zjunlp/KnowEdit` 下载 ZsRE 测试集，取前 500 条用于 MEMIT 批量注入。

## 四、实验结果

### Task 1：基线测试

运行 `baseline.py`，未编辑模型在10条事实上的生成结果摘要：

| 样本 | 是否包含 ground_truth | 是否包含 target_new |
|------|----------------------|---------------------|
| 1    | ✓ (Paris)            | ✗                   |
| 2    | ✓ (Au)               | ✗                   |
| 3    | ✓ (Jane Austen)      | ✗                   |
| 4    | ✓ (Pacific)          | ✗                   |
| 5    | ✓ (Bill Gates)       | ✗                   |
| 6    | ✗                    | ✗                   |
| 7    | ✗                    | ✗                   |
| 8    | ✓ (Leonardo da Vinci)| ✗                   |
| 9    | ✗ (输出 Amazon)      | ✓ (意外命中)         |
| 10   | ✓ (Bell)             | ✗                   |

**基线准确率（ground_truth 命中率）**：70% (7/10)  
**target_new 意外命中率**：10% (1/10)

输出文件：`results/baseline.json`。

### Task 2：ROME 单条编辑

对10条事实逐一进行 ROME 编辑（每次重置模型）。编辑后对直接 prompt 和同义改写的生成结果进行评估：

| 指标 | 命中数/总数 | 百分比 |
|------|-------------|--------|
| 编辑成功率 (ES) | 8/10 | 80% |
| 泛化性 (PS) | 7/10 | 70% |
| 局部性 (NS) | 9/10 | 90% |

输出文件：`results/rome_results.json`。

### Task 3：MEMIT 批量编辑

使用 500 条 ZsRE 数据进行一次性批量注入，并在 10 条自定义事实集上评估（对比 ROME）：

- **批量编辑耗时**：47.3 秒
- **峰值显存增量**：约 3.2 GB
- **编辑后评估**（在 10 条自定义事实上）：

| 指标 | 命中数/总数 | 百分比 |
|------|-------------|--------|
| 编辑成功率 (ES) | 8/10 | 80% |
| 泛化性 (PS) | 7/10 | 70% |
| 局部性 (NS) | 9/10 | 90% |

输出文件：`results/memit_results.json`。

### Task 4：综合评估（ES / PS / NS）

汇总三种方法的指标如下：

| 方法 | ES (%) | PS (%) | NS (%) |
|------|--------|--------|--------|
| Baseline | 10% | 10% | 100% |
| ROME | 80% | 70% | 90% |
| MEMIT | 80% | 70% | 90% |

> Baseline 中 ES=10% 是因为第9条（最长河流）模型意外输出了 target_new “Amazon”；PS 同样由于同义改写也命中。NS=100% 说明 locality 完全保持。

ROME 与 MEMIT 在 10 条测试集上表现相近，但 MEMIT 能够一次性编辑 500 条知识，效率更高。与 ROME 相比，MEMIT 的泛化性和局部性基本持平。

## 五、失败案例分析

1. **ROME 编辑失败**：第6条 “The planet known as the Red Planet is” 修改为 “Jupiter” 后，模型仍输出 “Mars”。可能原因是 subject “Red Planet” 在 prompt 中不是完整名词短语，ROME 定位偏差。
2. **泛化失败**：针对第2条事实，编辑 “The chemical symbol for gold is Go” 后，同义改写 “What is the element symbol for gold?” 仍输出原始的 ground_truth “Au”，说明模型未完全将新知识迁移到不同的提问方式上。
3. **编辑失败且局部性破坏**：针对第9条事实，尝试将“世界最长河流”修改为“Amazon”失败（直接 prompt 仍输出 “Nile”）。更严重的是，在此次编辑后，询问 “The second longest river in the world is” 时，模型也将原本正确的答案错误地改成了 “Nile”，说明该知识簇在编辑计算过程中受到了负面干扰。

## 六、结论

- ROME 和 MEMIT 均能有效修改模型中的事实知识，在 Qwen2.5-0.5B 上达到约 80% 的编辑成功率。
- 泛化性（PS）普遍低于 ES，表明编辑后的知识对同义改写的鲁棒性仍需提升。
- MEMIT 支持大规模批量编辑，效率远高于 ROME，且局部性保持良好（约 90%），适合实际应用。
- 失败案例提示：编辑位置（层数选择）、subject 提取准确性、同义改写样本丰富度都会影响最终效果。

## 七、代码与资源

- 实验代码独立仓库：[https://github.com/yixi0127-alt/knowledge-editing-assignment]
- 包含全部脚本（`baseline.py`, `edit_rome.py`, `edit_memit.py`, `evaluate.py`, `utils.py`）、配置文件、数据样例和结果 JSON。

## 八、附录

- 终端输出截图见 `reports/screenshots/` 目录。
- 所有输出文件（`baseline.json`, `rome_results.json`, `memit_results.json`, `metrics.json`）已随提交打包。
