研究目标：用表征工程方法来研究，LLM识别嫉妒情绪时，主要参考哪些嫉妒产生因素，是不是和人类一样

预计测试模型：

**<font style="color:rgba(31,35,41,1.000000);">Llama 系列:（优先测，稳定普适，作为baseline）</font>**

<font style="color:rgba(31,35,41,1.000000);">Llama 3.2 1B Instruct</font>

<font style="color:rgba(31,35,41,1.000000);">Llama 3.1 8B Instruct</font>

**<font style="color:rgba(31,35,41,1.000000);">Qwen系列:（更先进，表现超过llama）</font>**

<font style="color:rgba(31,35,41,1.000000);">Qwen2.5-7B-Instruct</font>

<font style="color:rgba(112,124,192,1.000000);">Qwen2.5-14b-instruct</font>

**<font style="color:rgba(31,35,41,1.000000);">Gemma 系列:</font>****<font style="color:rgba(31,35,41,1.000000);">（与llama</font>****<font style="color:rgba(31,35,41,1.000000);">架构差异最大</font>****<font style="color:rgba(31,35,41,1.000000);">）</font>**

<font style="color:rgba(31,35,41,1.000000);">Gemma 2 2B Instruct (google/gemma-2b-it)</font>

<font style="color:rgba(31,35,41,1.000000);">Gemma 2 9B Instruct (google/gemma-2-9b-it)</font>

**<font style="color:rgba(31,35,41,1.000000);">Phi 系列:</font>****<font style="color:rgba(31,35,41,1.000000);">（顺便验证训练数据质量）</font>**

<font style="color:rgba(31,35,41,1.000000);">Phi 3.5 mini Instruct</font>

<font style="color:rgba(31,35,41,1.000000);">Phi 3 medium Instruct (Phi 3 medium-128k-instruct)</font>

**<font style="color:rgba(143,149,158,1.000000);">OLMo 系列:</font>**

<font style="color:rgba(143,149,158,1.000000);">OLMo 2 7B Instruct</font>

<font style="color:rgba(143,149,158,1.000000);">OLMo 2 13B Instruct</font>

**<font style="color:rgba(143,149,158,1.000000);">Mistral 系列:</font>**

<font style="color:rgba(143,149,158,1.000000);">Ministral 8B Instruct</font>

<font style="color:rgba(143,149,158,1.000000);">Mistral 12B Nemo Instruct</font>

### **第一阶段：理论构建与数据生成 (Theory & Data Construction)**
1. **定义嫉妒的“因子配方”**
    - 基于社会比较理论，确定 $ N $ 个（7个）核心因子（Primitive Concepts）和无关安慰剂因子（2个）：
        * $ F_1 $：领域关键性
        * $ F_2 $：预期满意度
        * $ F_3 $：感知控制力 
        * $ F_4 $：领域替代性 
        * $ F_5 $：他人相似性
        * $ F_6 $：他人优越性
        * $ F_7 $：他人应得性
        * $ F_8 $：得知方式（无关因素）
        * $ F_9 $：他人衣着（无关因素）
    - **目标**：验证 $ Jealousy \approx f(F_1, F_2, F_3, F_4...F_9) $。

| 因子 | **<font style="color:rgba(31,35,41,1.000000);">全称</font>** | **<font style="color:rgba(31,35,41,1.000000);">简称</font>** | **定义** |
| --- | --- | --- | --- |
| 领域相关性 | **<font style="color:rgba(31,35,41,1.000000);">Self-Relevance</font>** | **<font style="color:rgba(31,35,41,1.000000);">Relevance</font>** | the domain of comparison in which the envied person enjoys an advantage should be self-relevant<br/>被嫉妒者与嫉妒者的比较领域对于嫉妒者的自我价值具有核心重要性   |
| 预期满意度 | **<font style="color:rgba(31,35,41,1.000000);">Expect Satisfaction</font>** | **<font style="color:rgba(31,35,41,1.000000);">Expectation</font>** | The target person was described as either satisfied or dissatisfied with this outcome by virtue of it either matching or falling short of expectations (comparison level).<br/>嫉妒者在比较领域的自我表现是否达到了个人的预期标准。   |
| 感知控制力 | **<font style="color:rgba(31,35,41,1.000000);">Perceived Control</font>** | **<font style="color:rgba(31,35,41,1.000000);">Control</font>** | The extent to which one believes that subsequent outcomes are controllable or alterable.  <br/>嫉妒者相信其随后的结果是可控或可改变的程度。  |
| 领域替代性 | **<font style="color:rgba(31,35,41,1.000000);">Comparison Alternative</font>** | **<font style="color:rgba(31,35,41,1.000000);">Alternative</font>** | the level of outcomes that a person experiences on the next most favourable dimension of comparison.<br/>除了当前遭遇失败的领域之外，嫉妒者表现最好、最有利的那个维度是否成功 |
| 他人相似性 | **<font style="color:rgba(31,35,41,1.000000);">Similarity of Comparison Person</font>** | **<font style="color:rgba(31,35,41,1.000000);">Similarity</font>** | we envy people who are similar to ourselves, save for their advantage on the desired domain.<br/>嫉妒者与被嫉妒者在背景或属性(非比较领域)上的可比性 |
| 他人优越性 | **<font style="color:rgba(31,35,41,1.000000);">Superiority of Comparison Person</font>** | **<font style="color:rgba(31,35,41,1.000000);">Superiority</font>** | the comparison person experienced a more favourable outcome<br/>被嫉妒者在比较领域上更加成功   |
| 他人应得性 | **<font style="color:rgba(31,35,41,1.000000);"> Deservingness  of Comparison Person</font>** | **<font style="color:rgba(31,35,41,1.000000);">Deservingness  </font>** | The extent to which the advantaged person is perceived as meriting their advantage (e.g., through effort vs. luck/unfair means).  <br/>被嫉妒者在比较领域上的投入程度或手段合法性 |
| 得知方式（无效因素） | **<font style="color:rgba(31,35,41,1.000000);"></font>** | **<font style="color:rgba(31,35,41,1.000000);"></font>** | 通过邮件、社交媒体、短信得知； <br/>面对面听说、现场看到   |
| 他人衣着（无效因素） | **<font style="color:rgba(31,35,41,1.000000);"></font>** | **<font style="color:rgba(31,35,41,1.000000);"></font>** | 穿着灰色衣服<br/>穿着米色衣服  |


2. **需要生成的数据集 (见以下文档）**

[正式实验数据集方案](https://www.yuque.com/adashelby/bew3a0/xuc5eytgp12xqseb)

### **第二阶段：向量提取与纯化 (Extraction & Purification)**
**核心策略**：利用“极值”提取方向，利用“正交投影”去除噪音。

#### 1、基线提取 (Baseline Extraction via LAT)
    - **方法**：采用 **“两头提取策略”**。
+ 以下是 **“基线提取 (Baseline Extraction via LAT)”** 的详细操作指南。

---

##### **<font style="color:#4C16B1;">步骤一：构建“极值对照组” (Constructing Extreme Contrast Sets)</font>**
<font style="color:#4C16B1;">在</font>**<font style="color:#4C16B1;">提取向量</font>**<font style="color:#4C16B1;">（训练探针）阶段，为了最大化 </font>**<font style="color:#4C16B1;">信噪比 (Signal-to-Noise Ratio)</font>**<font style="color:#4C16B1;">，我们使用“极端数据”。</font>

1. **<font style="color:#4C16B1;">用什么数据集？</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">对于 </font>**<font style="color:#4C16B1;">“嫉妒”</font>**<font style="color:#4C16B1;"> 概念：</font>
        * **<font style="color:#4C16B1;">正例集 (</font>**$ D_{pos} $**<font style="color:#4C16B1;">)</font>**<font style="color:#4C16B1;">：选取所有标注为 </font>**<font style="color:#4C16B1;">1分（极度嫉妒）</font>**<font style="color:#4C16B1;"> 的样本。</font>
        * **<font style="color:#4C16B1;">负例集 (</font>**$ D_{neg} $**<font style="color:#4C16B1;">)</font>**<font style="color:#4C16B1;">：选取所有标注为 </font>**<font style="color:#4C16B1;">0分（无嫉妒/其他情绪）</font>**<font style="color:#4C16B1;"> 的样本。</font>
    - <font style="color:#4C16B1;">对于 </font>**<font style="color:#4C16B1;">“因子 ”</font>**<font style="color:#4C16B1;">：</font>
        * **<font style="color:#4C16B1;">正例集 (</font>**$ D_{pos}^{F1} $**<font style="color:#4C16B1;">)</font>**<font style="color:#4C16B1;">：选取“领域关键性”为 </font>**<font style="color:#4C16B1;">1分</font>**<font style="color:#4C16B1;"> 的样本（无论是否嫉妒）。</font>
        * **<font style="color:#4C16B1;">负例集 (</font>**$ D_{neg}^{F1} $**<font style="color:#4C16B1;">)</font>**<font style="color:#4C16B1;">：选取“领域关键性”为 </font>**<font style="color:#4C16B1;">0分</font>**<font style="color:#4C16B1;"> 的样本。</font>
2. **<font style="color:#4C16B1;">配对 (Pairing)</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">Zou et al. (2023) 指出，构建 </font>**<font style="color:#4C16B1;">差异向量 (Difference Vectors)</font>**<font style="color:#4C16B1;"> 能比直接使用原始向量更精准地捕捉概念方向。</font>
    - **<font style="color:#4C16B1;">操作</font>**<font style="color:#4C16B1;">：控制变量地将 </font>$ D_{pos} $<font style="color:#4C16B1;"> 中的一个样本与 </font>$ D_{neg} $<font style="color:#4C16B1;"> 中的一个样本配对，形成 </font>$ (x_{pos}, x_{neg}) $<font style="color:#4C16B1;"> 对。</font>
3. **<font style="color:#4C16B1;">数据集生成Prompt举例</font>**

> **领域关键性**
>
> + **High Label (1):**<font style="color:rgb(31, 31, 31);"> 显著加强了“自我价值绑定”。例如，“那座奖杯是我毕生的梦想”、“那枚金牌是我永远无法弥补的遗憾”。</font>
> + **Low Label (0):**<font style="color:rgb(31, 31, 31);"> 彻底贯彻了“无感/无意义”。例如，“那个奖项对我没有任何吸引力”、“那本书的内容对我毫无意义”。</font>
> + **Low Label (0) Role:**<font style="color:rgb(31, 31, 31);"> 确保了“另外一个领域”的要求，例如钢琴家 vs 拳击教练，科学家 vs 农夫。</font>
>
> **System Role**: You are an expert dataset creator for psychology research.
>
> **User Prompt**: Please generate 50 pairs of short, first-person vignettes (1-2 sentences) centered on "Domain Relevance".
>
> + **Sentence A (High Relevance)**: The narrator describes an event in a field they care deeply about, consider their life's passion, or is crucial for their self-esteem.
> + **Sentence B (Low Relevance)**: The exact same event happens, but the narrator describes it as a hobby, a chore, or something they are indifferent towards.
> + **Constraint**: Keep the event, outcome, and other context identical. Only change the narrator's internal valuation of the domain.
> + **Format**: JSON list of objects `{"pos": "...", "neg": "..."}`
>



> **他人优越性**
>
> **User Prompt**: Please generate 50 pairs of short vignettes centered on "Other Superiority".
>
> + **Sentence A (High Superiority)**: The narrator compares themselves to another person ('Target'), where the Target performs significantly better, wins, or achieves a higher status than the narrator.
> + **Sentence B (Low Superiority)**: The narrator compares themselves to the Target, where the Target performs equal to or worse than the narrator.
> + **Constraint**: Keep the domain and relationship identical. Only change the comparative outcome.
> + **Format**: JSON list of objects `{"pos": "...", "neg": "..."}`.
>



> **他人衣着**
>
> **User Prompt**: Please generate 50 pairs of short vignettes differing only in "Clothing Color".
>
> + **Sentence A (Red)**: Describe a person wearing red clothing doing an action.
> + **Sentence B (Blue)**: Describe the same person doing the same action but wearing blue clothing.
> + **Constraint**: The action and context must be absolutely identical.
> + **Format**: JSON list of objects `{"pos": "...", "neg": "..."}`.
>

##### **<font style="color:#4C16B1;">步骤二：设计 LAT 模板 (Designing LAT Templates)</font>**
<font style="color:#4C16B1;">不能直接把原始句子扔给模型，需要用 </font>**<font style="color:#4C16B1;">模板 (Template)</font>**<font style="color:#4C16B1;"> 将模型的注意力聚焦到特定的概念上。</font>

<font style="color:#4C16B1;">根据 Zou et al. (2023) 的附录 D.1，你需要为“嫉妒”和“因子”分别设计模板：</font>

1. **<font style="color:#4C16B1;">针对“嫉妒” (Concept Template)</font>**<font style="color:#4C16B1;">：</font>

```latex
用户: 考虑以下场景中的【嫉妒】情绪：
场景: [插入场景文本]
助手: 该场景中的【嫉妒】程度是
```

```latex
Consider the amount of jealousy in the following scenario: 
[Scenario A]. 
The amount is: (High/Low)
```

2. **<font style="color:#4C16B1;">针对“因子” (Primitive Concept Template)</font>**<font style="color:#4C16B1;">：</font>

```latex
% 例如针对“领域关键性”：
用户: 考虑主角对比赛领域的【重视程度】：
场景: [插入场景文本]
助手: 主角对该领域的【重视程度】是
```

    - _<font style="color:#4C16B1;">注：对于 Decoder 模型（如 LLaMA），模型在生成“是”字之后的那个位置，其内部状态汇聚了对前面所有信息的判断。</font>_

##### **<font style="color:#4C16B1;">步骤三：采集神经活动 (Collecting Neural Activity)</font>**
<font style="color:#4C16B1;">让模型阅读上述构造好的 Prompt，并“拦截”其思维快照。</font>

1. **<font style="color:#4C16B1;">定位读取位置 (Token Position)</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">对于 Decoder 模型（GPT/LLaMA），读取 </font>**<font style="color:#4C16B1;">最后一个 Token</font>**<font style="color:#4C16B1;">（即 Prompt 的末尾）的隐藏状态。</font>
    - _<font style="color:#4C16B1;">参考 Tak et al. (2025)</font>_<font style="color:#4C16B1;">：虽然 Tak 建议关注 Query 的最后一个 Token，但 Zou 的标准 LAT 方法通常取整个输入的最后一个 Token。为了基线提取，建议先遵循 Zou 的标准做法。</font>
2. **<font style="color:#4C16B1;">定位层数 (Layer Selection)</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">你需要采集 </font>**<font style="color:#4C16B1;">所有层</font>**<font style="color:#4C16B1;"> 或 </font>**<font style="color:#4C16B1;">中间层到高层</font>**<font style="color:#4C16B1;"> 的数据。</font>
    - _<font style="color:#4C16B1;">依据</font>_<font style="color:#4C16B1;">：Tak et al. (2025) 发现情绪处理主要发生在 </font>**<font style="color:#4C16B1;">中间层（Mid-layers，如第 9-15 层）</font>**<font style="color:#4C16B1;">，而 Zou et al. (2023) 也发现概念通常在中间层最清晰。</font>
3. **<font style="color:#4C16B1;">数据收集</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">输入配对样本 </font>$ (x_{pos}, x_{neg}) $<font style="color:#4C16B1;">。</font>
    - <font style="color:#4C16B1;">记录隐藏状态向量 </font>$ H_{pos} $<font style="color:#4C16B1;"> 和 </font>$ H_{neg} $<font style="color:#4C16B1;">。</font>

##### **<font style="color:#4C16B1;">步骤四：计算读取向量 (Computing the Reading Vector)</font>**
<font style="color:#4C16B1;">这是数学上的“提纯”步骤。我们将使用 </font>**<font style="color:#4C16B1;">PCA (主成分分析)</font>**<font style="color:#4C16B1;"> 来提取主方向。</font>

1. **<font style="color:#4C16B1;">计算差异向量 (Difference Vectors)</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">对于每一对样本，计算：</font>$ \delta_i = H_{pos}^{(i)} - H_{neg}^{(i)} $
    - _<font style="color:#4C16B1;">解释</font>_<font style="color:#4C16B1;">：这个减法操作抵消了句子中的无关信息（如句式、背景描写），只留下了从“1分”到“5分”的变化量。（如果控制变量的话无关信息应该比较少）</font>
2. **<font style="color:#4C16B1;">执行 PCA</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">将所有 </font>$ \delta_i $<font style="color:#4C16B1;"> 向量堆叠成一个矩阵。</font>
    - <font style="color:#4C16B1;">运行 PCA，提取 </font>**<font style="color:#4C16B1;">第一主成分 (First Principal Component)</font>**<font style="color:#4C16B1;">。</font>
    - <font style="color:#4C16B1;">这个第一主成分的方向 </font>$ v $<font style="color:#4C16B1;">，就是该层中代表“嫉妒”或“该因子”的最佳线性方向。</font>
3. **<font style="color:#4C16B1;">确定符号 (Sign Determination)</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">PCA 提取的方向可能是反的（即指向“非嫉妒”）。</font>
    - **<font style="color:#4C16B1;">校准</font>**<font style="color:#4C16B1;">：计算 </font>$ H_{pos} $<font style="color:#4C16B1;"> 在 </font>$ v $<font style="color:#4C16B1;"> 上的投影值，确保其均值大于 </font>$ H_{neg} $<font style="color:#4C16B1;"> 在 </font>$ v $<font style="color:#4C16B1;"> 上的投影值。如果不是，则 </font>$ v \leftarrow -v $<font style="color:#4C16B1;">。</font>

##### **<font style="color:#4C16B1;">步骤五：验证提取质量 (Validation)</font>**
<font style="color:#4C16B1;">在进入下一步（正交化或回归分析）之前，必须先验证提取出的基线向量是否有效。</font>

1. **<font style="color:#4C16B1;">测试集验证</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">拿出一组模型没见过的测试数据（包含 1-5 分）。（这边暂时还不确定是五折交叉验证还是训练集/测试集划分 (Train/Test Split)）</font>
    - <font style="color:#4C16B1;">计算每个样本的隐藏状态 </font>$ H_{test} $<font style="color:#4C16B1;"> 在向量 </font>$ v $<font style="color:#4C16B1;"> 上的投影值：</font>$ Score = H_{test} \cdot v $<font style="color:#4C16B1;">。（</font><font style="color:#4C16B1;background-color:#FBDE28;">注意保留S值，这在第四阶段会用到</font><font style="color:#4C16B1;">）</font>
2. **<font style="color:#4C16B1;">相关性检查</font>**<font style="color:#4C16B1;">：</font>
    - <font style="color:#4C16B1;">检查 </font>$ Score $<font style="color:#4C16B1;"> 是否与 GPT-4 标注的真实分数（1-5分）呈现正相关，对于0/1的，看分类准确性。</font>
    - <font style="color:#4C16B1;">如果 Pearson 相关系数较高（例如 </font><font style="color:#4C16B1;">></font><font style="color:#4C16B1;"> 0.6），说明基线提取成功。</font>
+ <font style="color:rgba(31,35,41,1.000000);">一方面可以看各个不同的因素的决策关键层在哪里（哪几层准确率上升最高），一方面可以看Vi提取的好不好（准确率最高值是不是超过80%之类的）</font>

---

#### 2、向量解耦与纯化 (Orthogonalization)
    - **问题**：提取的“领域关键性”向量可能混杂了“不爽”或“焦虑”的情绪。
    - **操作**：计算 **唯一效应向量 (Unique Effect Vector, **$ z_a $**)**。
        * 构建投影矩阵，将目标因子向量投影到其他干扰因子的零空间（Null Space）上。
        * 公式：$ z_{target} = (I - P_{others}) \cdot v_{target} $。
    - **参考**：这是 Tak et al. (2025) 的核心贡献，确保你后续调节的是纯净的“因子”本身。

---

### **第三阶段：机制定位 (已包含在 第二阶段步骤五：验证提取质量 )**
Zou et al. (2023) 的研究习惯是先找到概念表征最清晰的“最佳层”，然后在这个层进行分析，而不是像 Tak 那样做全层扫描。

+ **操作**：
    1. 回顾您在“步骤五（验证提取质量）”中画的$ R^2 $ 曲线。
    2. **锁定层位**：找到 **嫉妒向量 (**$ v_{jealousy}
 $**)** 和 **因子向量 (**$ z_{factors} $**)** 预测准确率都较高且稳定的区域（通常是模型的 **中间偏后层**，例如 Llama-3-8B 的第 15-20 层）。
    3. **固定**：后续的所有计算（回归和干预）都将聚焦在这个“最佳层”进行。

### **第四阶段：统计权重计算 (Statistical Weighting)**
**—— 对应 Zou et al. 的“风险组合性实验” (Section 5.3.1)**

这是直接回答您“每个因素权重如何”的数学方法。Zou et al. 通过验证 $ Risk \approx Probability \times Utility $ 证明了概念的组合性，您将验证 $ Jealousy \approx \sum w_i \cdot Factor_i $

+ **目的**：计算在自然状态下，模型内部的“嫉妒”在多大程度上可以被各个“因子”线性解释。
+ **操作**：
    1. **准备数据**：使用包含 1-5 分连续打分的全量测试集。
    2. **获取分数 (Scoring)**：
        * 将样本输入模型，在“最佳层”提取隐藏状态 $ H $。
        * 计算嫉妒投影分：$ S_{jealousy} = H \cdot v_{jealousy} $
        * 计算各因子投影分（使用纯化向量）：$ S_{F1} = H \cdot z_{F1}, S_{F2} = H \cdot z_{F2}, ... $
    3. **标准化 (Standardization)**：对所有分数序列进行 Z-Score 标准化（减均值，除标准差），确保权重可比。
    4. **多元线性回归 (Regression)**：
        * 拟合公式：$ S_{jealousy} = \beta_1 S_{F1} + \beta_2 S_{F2} + ... + \beta_n S_{Fn} + \epsilon $
            + (即$ S_{jealousy} = \beta_1 H \cdot z_{F1} + \beta_2 H \cdot z_{F2} + ... + \beta_n H \cdot z_{Fn} + \epsilon $)
    5. **得出结论**：
        * **显著性检验**：看哪些因子的 P值 < 0.05。**P值不显著的因子直接排除**（回答了“哪几个因素产生嫉妒”）。
        * **权重排序**：比较标准化系数 $ |\beta| $ 的大小。$ \beta $ 越大，说明该因素对模型判断嫉妒的贡献越大（回答了“每个因素的权重如何”）。

### **第五阶段：因果权重验证 (Causal Weighting)**
#### **核心策略：先“加”后“减”**
遵循 Zou 的标准逻辑（Manipulation $ \rightarrow $ Termination），建议您 **先做干预（Steering），再做消除（Knockout）**。

+ **干预（Steering）** 是为了 **“定性”**：证明这个因子确实能驱动嫉妒（是因果，不是巧合）。
+ **消除（Knockout）** 是为了 **“定量”**：计算缺了它嫉妒值掉多少（这才是您用来排名的核心数据）。

---

#### **Step 5.1：正向干预实验 (Intervention / Steering ****<font style="background-color:#FBDE28;">可选</font>****)**
**—— 目的：验证“驱动力” (Sufficiency)**

这一步就像是“按下按钮看灯亮不亮”。如果按了“领域关键性”按钮，嫉妒灯亮了，说明你找对按钮了。

1. **准备数据**：找一组 **“低嫉妒”** 的场景（例如：路人甲拿了冠军，主角无感）。
2. **实施干预**：
    - 在模型的关键层（如 Layer 15-20），将隐藏状态加上因子的纯化向量：

$ H' = H + \alpha \cdot z_{F1} $

    - _注：_$ \alpha $_ 是干预强度，可以设为 5 或 10。_
3. **观测结果**：
    - 看模型的输出是否从“无感”变成了“嫉妒”？
    - 或者看“嫉妒向量”的读数是否显著上升？
4. **决策点**：
    - **成功**：确认该因子是嫉妒的**有效驱动力**。进入下一步。
    - **失败**：如果加了向量也没反应，说明这个因子可能只是“伴随信号”，或者向量提取质量太差。**在最终排名中应标记为无效或低权重。**

#### **Step 5.2：反向消除实验 (Termination / Knockout)**
**—— 目的：计算“依赖度” (Necessity)这是您论文最重要的“排序依据”。**

这一步就像是“拆掉零件看机器停不停”。拆了它机器就停转，说明它是核心零件。

1. **准备数据**：找一组 **“高嫉妒”** 的场景（例如：死对头在核心领域赢了，主角嫉妒值 5）。
2. **记录基准**：记录模型对这些样本原始的嫉妒激活值 $ S_{original} $（比如平均是 4.8）。
3. **实施消除**：
    - 在模型的关键层，将隐藏状态沿该因子向量进行投影消除：

$ H' = H - (H \cdot z_{F2}) \cdot z_{F2} $

4. **计算跌幅**：
    - 记录消除后的嫉妒激活值 $ S_{ablated} $。
    - 计算 $ \Delta \text{Drop} = S_{original} - S_{ablated}= H_{raw} \cdot v_{jealousy}-H_{ablated}\cdot v_{jealousy} $
    - 
5. **得出排名**：
    - $ \Delta $ 越大，说明模型越离不开这个因子，**重要性越高**。

---

### **最终：您的“重要性排序表”如何生成？**
您将结合 **第四阶段（回归）** 和 **第五阶段（消除）** 的数据，生成最终结论。

| 排名 | 因子名称 | Step 4 (回归 $ \beta $) | Step 5.1 (干预$ \Delta\text{Score} $) | Step 5.2 (消除 $ \Delta\text{Drop} $) | 综合结论 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **F1 他人优越性** | 0.65 | **+3.2** (High)   | **-4.5 (High)** | **核心主导** (Core) |
| **2** | **F2 领域关键性** | 0.30 | +1.5（Mid） | **-2.1 (Mid)** | **重要辅助** (Important) |
| **3** | **F4 关系亲密性** | 0.05 | +0.1（Low） | **-0.2 (Low)** | **模型不考虑** (Ignored) |


+ **如果 Step 5.1 失败**（干预无效），直接把该因子排在最后。
+ **如果 Step 5.1 成功**，则根据 **Step 5.2 的跌幅** 进行具体排名。
+ **Step 4 的 **$ \beta $ 用来佐证 Step 5.2 的排名（通常两者应该一致，如果不一致，以剔除实验为准，因为那是更强的因果证据）。

这就是您 **“Zou 范式 + Tak 纯化”** 的完整执行方案。先证明能控制（Step 5.1），再通过破坏来量化重要性（Step 5.2），逻辑无懈可击。





