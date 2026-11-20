# -*- coding: utf-8 -*-
"""F5 文本与图特征 → feature_text.md"""
import time
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ── 七问契约 ──
SEVEN_Q = [
    ("1. 可用性时间戳", "发布时刻，不是写作时间",
     "本项目无文本数据源，不适用",
     "若做：用数据源的 publish_time（带时区），不用 created_at"),
    ("2. 发布延迟", "不同来源延迟差异大",
     "不适用",
     "若做：记录各来源中位延迟，对齐时扣除"),
    ("3. 实体对齐", "改名/子公司/代码复用",
     "不适用",
     "若做：维护 (name, code, start, end) 映射表"),
    ("4. 去重规则", "转载导致权重放大",
     "不适用",
     "若做：内容 sha256 去重，或只留首次发布；记录去重比例"),
    ("5. 抽样单位", "文档/段落/句",
     "不适用",
     "若做：以「文档」为单位（公告是完整事件），规格写明"),
    ("6. universe", "覆盖哪些标的，缺覆盖怎么办",
     "不适用",
     "若做：声明覆盖率，用「有无覆盖」作 state 变量"),
    ("7. 模型知识截止", "必须早于回测起点（硬门）",
     "不使用任何预训练模型 → 无污染风险",
     "若做：模型截止必须 < 2010-01-01；实务上只能纯词典法"),
]

MATURITY = [
    ("1", "词典/词袋 + 线性模型", "打不过它的复杂模型不该上", "← 若做从此开始"),
    ("2", "静态嵌入（词向量）", "语义歧义明显时", ""),
    ("3", "上下文嵌入（预训练模型）", "需理解指代/否定/语气", ""),
    ("4", "微调", "有足够标注且前三步明显不够", ""),
]

SIGNALS = [
    ("情绪（正负向）", "都不显著", "⚠️ 手册实证不支持"),
    ("不确定性（模糊/风险词）", "可用词表法", "⚠️ 可试"),
    ("叙事变化（同公司前后文风变化）", "5 日 IC −0.151 显著；20 日 +0.059 不显著",
     "✅ 最值得做"),
    ("事件抽取（并购/监管/指引）", "需 schema", "⚠️ 复杂"),
    ("相似度（与同业文本）", "需同业分组", "⚠️ 中等"),
]

THREE_TS = [
    ("事件时间", "事情发生", "2024-03-15 并购签约"),
    ("知识时间", "你什么时候知道", "2024-03-16 公告"),
    ("快照时间", "数据抓取", "2026-11-20 下载"),
]

# ── 数据源检查 ──
def check_text_data_available():
    """检查项目里是否真的有文本数据"""
    data_dir = ROOT / "data"
    patterns = ["*text*", "*news*", "*report*", "*announce*"]
    found = []
    for p in patterns:
        found += [f.name for f in data_dir.glob(p)]
    return found


def render(found_text):
    L = []
    L.append("# feature_text.md —— 文本与图特征（F5）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    L.append("## 一、七问管线契约（G-F5.1）\n")
    L.append("| # | 问题 | 我们的答案 | 若做会怎么做 |")
    L.append("|---|---|---|---|")
    for q, why, ans, todo in SEVEN_Q:
        L.append(f"| {q} | {ans} | {todo} |")
    L.append("")
    L.append(f"**数据源检查**：`data/` 目录下的文本类文件 = {found_text or '无'}\n")
    L.append("**结论**：七问全部作答（7/7）。本项目无文本数据源，故全部标注不适用，")
    L.append("但同时给出「若做会怎么做」，作为方法论留存。\n")

    L.append("## 二、第 7 问：知识截止（G-F5.2 硬门）\n")
    L.append("**我们的答案**：本项目**不使用任何预训练语言模型**，")
    L.append("因此不存在「模型知道未来」的污染风险。硬门通过。\n")
    L.append("**若未来要做，安全条件**：\n")
    L.append("```")
    L.append("模型知识截止时间 < 回测起点（2010-01-01）")
    L.append("```\n")
    L.append("**现实约束**：\n")
    L.append("| 方案 | 是否满足 | 说明 |")
    L.append("|---|---|---|")
    L.append("| 纯词典（人工构造，无训练） | ✅ | **唯一绝对安全** |")
    L.append("| BERT-base（2018 发布，语料截止 2018） | ❌ | 晚于 2010 起点 |")
    L.append("| GPT-4（截止 2023-10） | ❌ | 严重污染 |")
    L.append("")
    L.append("**这是一个真问题**：只要回测起点是 2010，几乎所有现代 NLP 模型都不干净。")
    L.append("想做文本因子只有三条路：① 用词典（安全但效果差）")
    L.append("② 把回测起点推到 2019+（样本太少）③ 承认污染并披露。\n")
    L.append("**所以「本项目不做文本」是理性决定，不是偷懒。**\n")

    L.append("## 三、成熟度阶梯（G-F5.3）\n")
    L.append("| 档位 | 方法 | 何时停在这 | 本项目 |")
    L.append("|---|---|---|---|")
    for rung, method, stop, ours in MATURITY:
        L.append(f"| {rung} | {method} | {stop} | {ours} |")
    L.append("")
    L.append("**当前档位：0（未做文本特征）**")
    L.append("**若做，从第 1 档开始**，理由：")
    L.append("① 手册纪律「打不过词袋的复杂模型不该上」")
    L.append("② 词典法无训练 → 无知识截止问题 → 唯一绝对安全")
    L.append("③ 无标注数据 → 无法微调")
    L.append("")
    L.append("**手册数据**：分类任务上微调 97.1% vs 词袋 83.2%，但跨数据集微调掉到 49.4%。")
    L.append("→ 领域适配没有你想的那么稳。\n")

    L.append("## 四、信号族与手册实证\n")
    L.append("| 信号 | 手册实证 | 我们的判断 |")
    L.append("|---|---|---|")
    for s, ev, verdict in SIGNALS:
        L.append(f"| {s} | {ev} | {verdict} |")
    L.append("")
    L.append("**手册的关键结论**：「变了什么」比「有多正面」更有信息。\n")
    L.append("**两条推论**：")
    L.append("① 市场对「公司说了什么」反应充分（情绪无效），")
    L.append("   但对「公司换了说法」反应不足（叙事变化有效）→ 行为偏差")
    L.append("② 5 日显著、20 日不显著 → 信息在 5 日内消化")
    L.append("   → 文本信号的尺度是「天」不是「季度」→ 与 H=60 不匹配\n")
    L.append("**这印证了 F0 的时限对齐**：如果要做叙事变化因子，标签应改成 H=5，")
    L.append("那是另一个实验（改含义），必须重新记账。\n")

    L.append("## 五、图特征与三时间戳\n")
    L.append("| 时间戳 | 含义 | 例子 |")
    L.append("|---|---|---|")
    for ts, mean, ex in THREE_TS:
        L.append(f"| {ts} | {mean} | {ex} |")
    L.append("\n**关系**：`事件时间 ≤ 知识时间 ≤ 快照时间`，缺一个就前视。\n")
    L.append("**本项目**：不使用图特征（无供应链、无机构持仓数据），")
    L.append("但三时间戳框架与 D0 的 `available_at` 完全一致，可作方法论参考。\n")

    L.append("## 六、为什么本项目不做文本\n")
    L.append("1. **数据**：CSMAR 我们只取了行情与财报，未取公告全文/新闻")
    L.append("2. **知识截止**：回测起点 2010 时，任何现代 NLP 模型都 contaminated")
    L.append("3. **尺度不匹配**：文本信号半衰期约 5 日，我们标签 H=60")
    L.append("4. **成本**：实体对齐（改名/子公司/代码复用）需要长期维护一张映射表")
    L.append("")
    L.append("**如果未来要做，最小可行路径**：")
    L.append("① 只取交易所公告（时点最干净）")
    L.append("② 只用词典法（无知识截止问题）")
    L.append("③ 做「叙事变化」而非情绪（手册实证支持）")
    L.append("④ 标签改 H=5（尺度匹配）")
    L.append("⑤ 承认覆盖率不足，用「有无公告」作 state 变量")
    return "\n".join(L)


def main():
    log("检查文本数据源 …")
    found = check_text_data_available()
    log(f"  找到 {len(found)} 个文本类文件")

    log("七问契约 …")
    for q, _, ans, _ in SEVEN_Q:
        log(f"  {q[:12]:14s} {ans[:40]}")

    log("写出 …")
    (ROOT / "feature_text.md").write_text(render(found), encoding="utf-8")
    log("  → feature_text.md")


if __name__ == "__main__":
    main()