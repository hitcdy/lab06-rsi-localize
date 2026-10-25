# -*- coding: utf-8 -*-
"""填写《实验六 RSI 递归自进化 实验报告样板》→ 2024312149_陈道延_实验作业六.docx

数据来自 hw6_rsi_localize.py 的实测落盘 results_lab06.json
（主实验：关闭图像切分、SFT 60 步 = 4 轮、RSI 2 轮）：
  params / sft_config / sft_timing     0.529% 可训练参数、60 步、82.9 s
  stage_baseline       2/15（loc 0/8, count 1/4, exist 1/3），IoU 0
  stage_sft            6/15（loc 2/8, count 4/4, exist 0/3），IoU 0.059
  stage_sft_corrected 10/15（loc 3/8, count 4/4, exist 3/3），触发 9、改对 4
  stage_rsi1 / rsi2    9/15（IoU 0.077）→ 10/15；数据池 15→19→21，新增 4/2
  stage_sft_ns10 / stage_sft / stage_sft_ns200b / stage_sft_split10  步数预算与切分对照
配套图：fig_loc_stages.png（图 1）、fig_pool_growth.png（图 2）、fig_steps_budget.png（图 3）
"""
import copy
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "实验六_RSI递归自进化_实验报告样板.docx")
OUT = os.path.join(HERE, "2024312149_陈道延_实验作业六.docx")
JS = os.path.join(HERE, "results_lab06.json")

NAME, SID, CLS = "陈道延", "2024312149", "自动化四班"
DATE = "2026 年 10 月 25 日"
REPO = "https://github.com/hitcdy/lab06-rsi-localize"


# ---------------------------------------------------------------- 基础工具
def body_items(doc):
    items = []
    for ch in doc.element.body.iterchildren():
        tag = ch.tag.split("}")[1]
        if tag == "p":
            items.append(("p", Paragraph(ch, doc)))
        elif tag == "tbl":
            items.append(("t", Table(ch, doc)))
    return items


def set_para_text(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    first = runs[0]
    first.text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def delete_para(p):
    el = p._element
    if el.getparent() is not None:
        el.getparent().remove(el)


def set_cell_text(cell, text, bold=False):
    p = cell.paragraphs[0]
    for extra in cell.paragraphs[1:]:
        delete_para(extra)
    set_para_text(p, str(text))
    if bold and p.runs:
        p.runs[0].bold = True


def add_row(table, texts, template_row=1):
    tr = copy.deepcopy(table.rows[template_row]._tr)
    table._tbl.append(tr)
    row = table.rows[-1]
    for i, t in enumerate(texts):
        if i < len(row.cells):
            set_cell_text(row.cells[i], t)
    return row


def fill_table(t, rows, template_row=1):
    keep = max(template_row + 1, 1 + len(rows))
    while len(t.rows) > keep:
        t._tbl.remove(t.rows[-1]._tr)
    for i, vals in enumerate(rows):
        idx = i + template_row
        if idx < len(t.rows):
            row = t.rows[idx]
            for c, v in enumerate(vals):
                if c < len(row.cells):
                    set_cell_text(row.cells[c], v)
        else:
            add_row(t, vals, template_row=template_row)


def find_para(items, text_prefix, start=0):
    for i in range(start, len(items)):
        k, o = items[i]
        if k == "p" and o.text.strip().startswith(text_prefix):
            return o
    return None


def table_after(items, caption_prefix):
    for i, (k, o) in enumerate(items):
        if k == "p" and o.text.strip().startswith(caption_prefix):
            for k2, o2 in items[i + 1:]:
                if k2 == "t":
                    return o2
    return None


def new_table_like(doc, src_tbl, header, rows):
    """以样板已有表格为模板克隆同宽新表（继承边框样式）"""
    from docx.oxml.ns import qn
    tbl = copy.deepcopy(src_tbl._tbl)
    trs = tbl.findall(qn("w:tr"))
    hdr_tr, dat_tr = trs[0], trs[1]
    for tr in trs:
        tbl.remove(tr)
    tbl.append(copy.deepcopy(hdr_tr))
    for _ in rows:
        tbl.append(copy.deepcopy(dat_tr))
    nt = Table(tbl, src_tbl._parent)
    for c, v in zip(nt.rows[0].cells, header):
        set_cell_text(c, v, bold=True)
    for tr, vals in zip(nt.rows[1:], rows):
        for c, v in zip(tr.cells, vals):
            set_cell_text(c, v)
    return nt


def new_table_4col(doc, src_tbl, header, rows):
    """克隆 3 列表格并给每行补一列（第 4 列），得到 4 列新表"""
    from docx.oxml.ns import qn
    tbl = copy.deepcopy(src_tbl._tbl)
    trs = tbl.findall(qn("w:tr"))
    hdr_tr, dat_tr = trs[0], trs[1]
    new_trs = []
    for base, vals in [(hdr_tr, header)] + [(dat_tr, r) for r in rows]:
        tr = copy.deepcopy(base)
        tcs = tr.findall(qn("w:tc"))
        tr.append(copy.deepcopy(tcs[-1]))     # 补第 4 列
        new_trs.append(tr)
    for tr in trs:
        tbl.remove(tr)
    for tr in new_trs:
        tbl.append(tr)
    # 表格网格也要补一列，否则列数按 tblGrid 算仍是 3
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        cols = grid.findall(qn("w:gridCol"))
        if cols:
            grid.append(copy.deepcopy(cols[-1]))
    nt = Table(tbl, src_tbl._parent)
    for c, v in zip(nt.rows[0].cells, header):
        set_cell_text(c, v, bold=True)
    for tr, vals in zip(nt.rows[1:], rows):
        for c, v in zip(tr.cells, vals):
            set_cell_text(c, v)
    return nt


def insert_para_after(el, doc, text, template_el=None, bold=False, size=None,
                      picture=None, picture_cm=15.0, align_center=False, mono=False):
    new_el = copy.deepcopy(template_el)
    el.addnext(new_el)
    p = Paragraph(new_el, doc)
    if picture:
        if p.runs:
            for r in list(p.runs):
                r._element.getparent().remove(r._element)
        if align_center:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(picture, width=Cm(picture_cm))
        return p
    set_para_text(p, text)
    if p.runs:
        if bold:
            p.runs[0].bold = True
        if size:
            p.runs[0].font.size = Pt(size)
        if mono:
            p.runs[0].font.name = "Consolas"
    return p


def insert_figure_after(el, doc, png, caption, template_el, cm=14.0):
    cap = insert_para_after(el, doc, caption, template_el=template_el)
    insert_para_after(el, doc, "", template_el=template_el, picture=png,
                      picture_cm=cm, align_center=True)
    return cap


# ---------------------------------------------------------------- 读数据
J = json.load(open(JS, encoding="utf-8"))
P, SFT_CFG, SFT_T = J["params"], J["sft_config"], J["sft_timing"]
BASE, SFT, CORR = J["stage_baseline"], J["stage_sft"], J["stage_sft_corrected"]
ROUNDS, TIM = J["rounds"], J["timings"]
VGOLD = J.get("verifier_gold_pass", "—")
SC = J["sft_curve"]
R1, R2 = ROUNDS[0], ROUNDS[1]
STEP10 = J.get("stage_sft_ns10")
STEP200 = J.get("stage_sft_ns200b")
SPLIT10 = J.get("stage_sft_split10")


def bt(st, t):
    if not st:
        return 0, 0
    return st["by_type"][t]["ok"], st["by_type"][t]["n"]


def cells(st):
    return [f"{bt(st, t)[0]}/{bt(st, t)[1]}" for t in ("loc", "count", "exist")] + \
           [str(st["total"]), (f"{st['mean_iou']:.3f}" if st.get("mean_iou") is not None else "—")]


def pick_cases():
    all_rec = []
    for st in ROUNDS:
        for qid, r in st["per_question"].items():
            all_rec.append((st["round"], qid, r))
    good = next((x for x in all_rec if x[2]["type"] == "loc" and x[2]["hit1"]), None)
    selffix = next((x for x in all_rec if x[2]["type"] == "loc" and not x[2]["hit1"]
                    and x[2].get("pred2_self") and x[2]["final_hit"]
                    and x[2]["final"] == x[2].get("pred2_self")), None)
    toolfix = next((x for x in all_rec if x[2]["type"] == "loc" and not x[2]["hit1"]
                    and x[2].get("pred2_tool") and x[2]["final_hit"]
                    and x[2]["final"] == x[2].get("pred2_tool")), None)
    fail = next((x for x in all_rec if x[2]["type"] == "loc" and not x[2]["final_hit"]), None)
    return good, selffix, toolfix, fail


def main():
    doc = Document(TEMPLATE)
    items = body_items(doc)

    tpl = None
    for k, o in items:
        if k == "p" and o.text.strip().startswith("用 2~4 句话概括"):
            tpl = copy.deepcopy(o._element)
            break
    if tpl is None:
        raise SystemExit("找不到正文段落模板")

    # ============================================================ 封面
    sub = find_para(body_items(doc), "SmolVLM-256M 精准定位的 RSI 递归自进化实验")
    if sub is not None:
        set_para_text(sub, f"SmolVLM-256M 精准定位的 RSI 递归自进化实验　·　"
                           f"{SID} {NAME}（{CLS}）　·　{DATE}")
    t = [o for k, o in body_items(doc) if k == "t"][0]
    rows = [
        ("实验名称", "SmolVLM-256M 精准定位的 RSI 递归自进化"
                     "（4×4 网格定位 + 解析式验证器 + 自举数据闭环）"),
        ("姓名 / 学号 / 班级", f"{NAME} / {SID} / {CLS}"),
        ("实验日期", DATE),
        ("运行环境", "CPU（Intel 平台 32 逻辑核 / 31.7 GB 内存，torch 线程数固定 4）；无 GPU；"
                     "Python 3.14.6、torch 2.14.0+cpu、transformers 5.19.0、peft 0.21.2、"
                     "pillow 12.3.0；模型 HuggingFaceTB/SmolVLM-256M-Instruct（fp32）"),
        ("代码仓库 / 文件", "hw6_rsi_localize.py + results_lab06.json + rsi_memory.json + "
                     f"sft_data.json（{REPO}）"),
    ]
    fill_table(t, rows, template_row=1)

    # ============================================================ 一、实验目的
    p = find_para(body_items(doc), "用 2~4 句话概括")
    set_para_text(p, (
        f"本实验让 2.56 亿参数的 SmolVLM-256M 学会把目标位置表达成 4×4 网格坐标「第 X 行第 Y 列」"
        f"（16 类离散输出），并在纯 CPU 上跑通「作答 → 高亮格回灌自查 → 解析式验证器纠错 → "
        f"通过验证的样本入池 → 再训练」的 RSI 递归自进化闭环。用基线 / SFT / SFT+纠错 / RSI 两轮"
        f"共 5 组数据回答核心问题：定位能力的提升究竟来自参数更新（微调）还是推理期的工具纠错。"
        f"实测 15 题（8 定位 + 4 数数 + 3 存在性）的总正确率依次为 {BASE['total']} → {SFT['total']}"
        f" → {CORR['total']} → {R1['eval']['total']} → {R2['eval']['total']}，"
        f"定位平均 IoU 依次为 {BASE['mean_iou']} → {SFT['mean_iou']} → {CORR['mean_iou']} → "
        f"{R1['eval']['mean_iou']} → {R2['eval']['mean_iou']}。"
    ))
    set_para_text(find_para(body_items(doc), "我做的主要工作"), "我做的主要工作：")
    work = [
        "1. 合成数据集：PIL 绘制 40 张训练图 + 15 张评测图（白底、红蓝绿黄 × 圆/方、1~4 个物体），"
        "绘制时同步记录 GT 包围盒 → hw6_imgs/gt.json（评测集种子 42，可与同学对齐）。",
        "2. 定位编码与种子数据：实现 4×4 网格换算（256 px → 64 px 格）与解析器"
        "（兼容「第X行第Y列」「rXcY」「X,Y」），由 GT 框直接生成 15 条种子 SFT 数据，零人工标注。",
        "3. 解析式验证器 + 两路纠错：颜色连通块分析判定「高亮格是否真的套住目标」，"
        "再用紫色高亮回灌（自查）与验证器反馈重答（工具纠错）两路纠错，逐题记录触发、改对、入池。",
        f"4. 全链路 + 观测：单进程跑完 基线 → QLoRA SFT（{SFT_CFG['steps']} 步）→ 三阶段评测 → "
        f"RSI 2 轮（每轮 {SFT_CFG['steps']} 步再训练），全部指标写 results_lab06.json；"
        f"另做步数预算扫描（10 / 60 / 200 步）、图像切分对照与验证器自检（gold 格通过 {VGOLD}）。",
    ]
    anchor = find_para(body_items(doc), "1. [")
    if anchor is None:
        raise SystemExit("找不到「1. [____]」占位段")
    set_para_text(anchor, work[0])
    for pre in ("2. [", "3. ["):
        pp = find_para(body_items(doc), pre)
        if pp is not None:
            delete_para(pp)
    cur = anchor._element
    for txt in work[1:]:
        pp = insert_para_after(cur, doc, txt, template_el=tpl)
        cur = pp._element

    # ============================================================ 二、环境与复现
    t = table_after(body_items(doc), "二、实验环境与复现步骤")
    rows = [
        ("--mode make_data", "生成 55 张合成图 + GT 框", "hw6_imgs/gt.json（< 1 s）"),
        ("--mode make_seed（同 make_sft）", "GT 框 → 网格坐标种子数据", "sft_data.json（< 1 s）"),
        (f"--mode sft --steps {SFT_CFG['steps']}", "QLoRA SFT 种子训练（"
                                                   f"{SFT_T['epochs']} 轮）",
         f"sft_adapter/（{SFT_T['seconds']:.0f} s）"),
        ("--mode eval", "三阶段评测（基线 / SFT / SFT+纠错）",
         f"results_lab06.json（{BASE['seconds']:.0f} + {SFT['seconds']:.0f} + "
         f"{CORR['seconds']:.0f} s）"),
        ("--mode rsi --rounds 2", "RSI 递归 2 轮（每轮采样+纠错+再训练+评测）",
         f"rsi_memory.json（采样 {TIM['rsi_seconds'][0]:.0f} + {TIM['rsi_seconds'][1]:.0f} s，"
         f"训练另计 {TIM['rsi_train_seconds'][0]:.0f} + {TIM['rsi_train_seconds'][1]:.0f} s）"),
    ]
    fill_table(t, rows, template_row=1)
    dep = find_para(body_items(doc), "依赖版本：")
    set_para_text(dep, (
        "依赖版本：transformers 5.19.0，peft 0.21.2，bitsandbytes 未用（CPU 上 fp32 + LoRA）；"
        "torch 2.14.0+cpu、Python 3.14.6、pillow 12.3.0。实际运行用一条命令 "
        "`python hw6_rsi_localize.py --mode all --rounds 2 --steps 60 --no_split 1` 跑完全链路，"
        f"总计 {TIM['all_seconds'] / 60:.1f} 分钟。对参考代码的改动：① 关闭图像切分"
        "（do_image_splitting=False）——默认切分会把 256×256 合成图放大到 2048 再切成 17 个 "
        "512 瓦片，模型看到的是 17 块局部放大图而不是整幅布局，对「整图第几行第几列」这种全局"
        "空间判断是硬伤（实测开启切分时 SFT 后定位仍 0/8，见表 5）；② 视觉塔前向缓存（视觉塔"
        "冻结、同图输出确定，按图像指纹缓存，语义等价），CPU 上每张新图视觉前向约 38~46 秒，"
        "不加缓存则 8 张评测图 × 5 个阶段要重复付 5 次；③ 固定随机种子，训练按轮打乱顺序；"
        "④ 步数循环改为可跨轮（参考实现 `for i in range(steps)` 会被数据量截断，15 条数据跑 "
        "60 步实际只训 15 步）；⑤ 入池语义按指南本意修正为「只收原本答错、纠错后答对」的样本；"
        "⑥ 全部指标落盘 results_lab06.json，报告数字均可回查。"
    ))

    # ============================================================ 三、方法
    p = find_para(body_items(doc), "3.1 网格坐标编码")
    set_para_text(p, (
        "3.1 网格坐标编码：256×256 画布等分成 4×4 网格，每格 64×64 像素。GT 包围盒取中心点 "
        "(cx, cy)，落到 row = ⌊cy/64⌋、col = ⌊cx/64⌋（0 基），答案统一写「第 r+1 行第 c+1 列」。"
        "解析器按优先级匹配三种写法：「第X行第Y列」、正则 [rR]X[cC]Y、以及「X,Y / X，Y / X-Y」，"
        "解析失败记为未命中（基座模型常见输出是复读问题或英文描述，属正常现象）。"
        "定位判分 = 解析出的格号与 GT 格号完全一致；另用「格框 ∩ GT 框」的 IoU 作连续精度指标。"
    ))
    p = find_para(body_items(doc), "3.2 QLoRA SFT")
    set_para_text(p, (
        f"3.2 QLoRA SFT：LoRA r={SFT_CFG['lora_r']}、alpha={2 * SFT_CFG['lora_r']}、dropout=0.05，"
        f"挂在语言模型 q/k/v/o 四处投影；可训练参数 {P['trainable']:,} / {P['total']:,} = "
        f"{P['ratio_pct']:.3f}%，实际训练 {SFT_T['steps']} 步（{SFT_T['epochs']} 轮 × 15 条，"
        f"AdamW lr={SFT_CFG['lr']:g}，batch=1），损失只计答案 token（labels 对 prompt 段置 -100），"
        f"耗时 {SFT_T['seconds']:.0f} s，loss 由 {SC[0][1]:.2f} 降到末步 {SC[-1][1]:.2f}。"
        f"另做了步数预算扫描：10 步（loc 0/8）、60 步（2/8）、200 步（4/8），见表 5 与图 3。"
    ))
    p = find_para(body_items(doc), "3.3 RSI 闭环")
    set_para_text(p, (
        "3.3 RSI 闭环（每轮四个动作）：① 采样——模型在 15 道评测题上直接作答（pred1）；"
        "② 自查——定位题答错且能解析出格号时，把预测格渲染成紫色高亮框回灌给模型，"
        "提示「若高亮框没有圈住目标，请直接给出正确位置」，得到 pred2_self；"
        "③ 校验——解析式验证器用颜色连通块分析判定高亮格是否套住目标（目标中心落在格内，"
        "或该格覆盖目标 ≥50% 面积），不通过则把「重叠仅多少」这类可核查证据写进提示词让模型重答"
        "（pred2_tool）；非定位题则注入「该颜色连通块 N 处 / 像素 N 个」的证据。"
        "④ 自举——只有「原本答错、纠错后答对」的样本（定位题还要被验证器复核通过）才写入数据池，"
        f"池子扩充后再训练 {SFT_CFG['steps']} 步，然后做轮末评测（不带纠错）。"
        "每轮逐题记录与统计写 rsi_memory.json / results_lab06.json 的 rounds 字段。"
    ))
    p = find_para(body_items(doc), "3.4 与参考实现不同")
    set_para_text(p, (
        "3.4 与参考实现不同的部分：① 关闭图像切分（见表 5 对照：开启切分时 SFT 后 loc 仍 0/8）；"
        "② 视觉前向缓存（工程加速，语义等价）；③ 固定种子 + 每轮打乱训练顺序；"
        "④ 步数循环改为可跨轮（参考实现受数据量截断）；⑤ 入池语义修正为「只收错→对的样本」，"
        "并记录「判分对但验证器复核不通过」而被拦下的样本"
        f"（两轮分别 {R1['n_verifier_denied']} / {R2['n_verifier_denied']} 条）；"
        f"⑥ 验证器自检：对 8 道定位题的 gold 格逐题跑验证器，实测通过 {VGOLD}"
        "（度量验证器自身的假阴性）；⑦ 曾尝试「颜色+形状」双判据验证器（用 bbox 四角是否着色"
        "区分圆/方），实测在重叠/遮挡场景下不可靠（同色物体并成一个连通块、不同色描边互相切断），"
        "故回退到指南的颜色单判据，并把这一发现写进 5.3。"
    ))

    # ============================================================ 四、结果
    t = table_after(body_items(doc), "4.1 三阶段评测")
    rows = [["基线（未微调）"] + cells(BASE),
            [f"SFT（种子 {SFT_CFG['steps']} 步）"] + cells(SFT),
            ["纠错后（Harness）"] + cells(CORR)]
    for i, st in enumerate(ROUNDS, start=1):
        rows.append([f"RSI 第 {i} 轮"] + cells(st["eval"]))
    fill_table(t, rows, template_row=1)
    insert_figure_after(t._tbl, doc, os.path.join(HERE, "fig_loc_stages.png"),
                        "图 1　定位答对题数与平均 IoU 随阶段变化"
                        "（基线 → SFT → 纠错 → RSI 两轮，4×4 网格，CPU 实测）", tpl, cm=14.5)

    t = table_after(body_items(doc), "4.2 RSI 递归自进化轮次记录")
    rows = [["初始（种子）", str(R1["pool_before"]), str(SFT["total"]), "—", "—", "—",
             f"{SFT['total']}/15"]]
    for st in ROUNDS:
        rows.append([f"第 {st['round']} 轮", str(st["pool_after"]), f"{st['n_ok1']}/15",
                     str(st["n_self"]), str(st["n_tool"]),
                     f"{st['n_ok1'] + st['n_fixed']}/15（改对 {st['n_fixed']}）",
                     f"{st['round_end_total']}/15"])
    fill_table(t, rows, template_row=1)
    insert_figure_after(t._tbl, doc, os.path.join(HERE, "fig_pool_growth.png"),
                        "图 2　RSI 自举数据池规模增长（种子 15 条 → 两轮扩充到 21 条）", tpl, cm=13.0)

    tot_tool = sum(r["n_tool"] for r in ROUNDS)
    tot_fixed = sum(r["n_fixed"] for r in ROUNDS)
    tot_new = sum(r["new_data"] for r in ROUNDS)
    tot_self = sum(r["n_self"] for r in ROUNDS)
    p = find_para(body_items(doc), "4.3 纠错明细统计")
    set_para_text(p, (
        f"4.3 纠错明细统计（两轮合计）：验证器纠错触发 {tot_tool} 次，由错变对 {tot_fixed} 次，"
        f"由对变错 0 次（纠错只对错题触发），新增可用样本 {tot_new} 条；自查改写 {tot_self} 次。"
        f"三阶段一次性纠错（SFT+纠错）另计：触发 {CORR['n_tool']} 次、由错变对 {CORR['n_fixed']} 次。"
        f"入池样本构成很关键：轮 1 的 {R1['new_data']} 条里 3 条是 exist 题（工具给出「连通块 1 处」"
        f"后模型答「有…」）、1 条是定位题 L08；轮 2 的 {R2['new_data']} 条里 1 条定位（L02）"
        f"+ 1 条 exist。"
    ))

    good, selffix, toolfix, fail = pick_cases()

    def case_cells(title, rec, path, result):
        if rec is None:
            return [title, "—", "—", "—", path, result]
        _, qid, r = rec
        out = r["pred1"]
        if r.get("pred2_self"):
            out += " →(自查)" + r["pred2_self"]
        if r.get("pred2_tool"):
            out += " →(工具)" + r["pred2_tool"]
        return [title, qid, r["q"][:32], out[:60], path, result]

    t = table_after(body_items(doc), "4.4 代表性样例")
    rows = [
        case_cells("答对", good, "直接", "√ gold " + (good[2]["gold"] if good else "—")),
        (case_cells("自查改对", selffix, "自查", "√") if selffix else
         ["自查改对", "—", "—", "本轮自查改写均未改对（见 5.3 ⑤）", "自查", "—（无此案例）"]),
        case_cells("验证器纠对", toolfix, "验证器",
                   "√ gold " + (toolfix[2]["gold"] if toolfix else "—")),
        case_cells("仍然失败", fail, "—", "× gold " + (fail[2]["gold"] if fail else "—")),
    ]
    fill_table(t, rows, template_row=1)

    # ============================================================ 五、分析
    p = find_para(body_items(doc), "5.1 SFT 带来了什么")
    set_para_text(p, (
        f"5.1 SFT 带来了什么：基线的定位输出完全是「复读问题」（如 L01 答 "
        f"{BASE['per_question']['L01']['pred'][:18]!r}）或退化数字串，loc {bt(BASE, 'loc')[0]}/8、"
        f"IoU {BASE['mean_iou']}；SFT 后输出变成「第X行第Y列」这一族，但退化成了"
        f"「第1行第2列第3列第4列…」的枚举——loc 记为 {bt(SFT, 'loc')[0]}/8，其中 L01/L05 两题"
        f"gold 恰好是「第1行第2列」，命中其实是解析器从枚举串里取到第一个格号的假命中"
        f"（必须诚实写明，见 5.3）。数数 {bt(BASE, 'count')[0]}/4 → {bt(SFT, 'count')[0]}/4"
        f"（该列已被「1 个」饱和），存在性 {bt(BASE, 'exist')[0]}/3 → {bt(SFT, 'exist')[0]}/3"
        f"（仍全答 No.）。真正说明「SFT 学到了格式」的证据是 IoU 从 {BASE['mean_iou']} 升到 "
        f"{SFT['mean_iou']}，以及后续轮次开始给出具体格号。"
    ))
    cap5 = insert_para_after(p._element, doc,
                             "表 5　SFT 训练量 / 预处理对照（CPU 实测）", template_el=tpl)
    src_t = table_after(body_items(doc), "二、实验环境与复现步骤")
    rows5 = []
    for label, v in (("10 步（开启图像切分）", SPLIT10), ("10 步（关闭切分）", STEP10),
                     ("60 步（关闭切分，本文主实验）", SFT), ("200 步（关闭切分）", STEP200)):
        if v:
            rows5.append([label, f"{bt(v, 'loc')[0]}/8", f"{v['total']}/15",
                          (f"{v['mean_iou']:.3f}" if v.get("mean_iou") is not None else "—")])
    t5 = new_table_4col(doc, src_t, ("SFT 训练量 / 预处理", "loc（/8）", "总计（/15）",
                                     "定位平均 IoU"), rows5)
    cap5._element.addnext(t5._tbl)
    insert_figure_after(t5._tbl, doc, os.path.join(HERE, "fig_steps_budget.png"),
                        "图 3　SFT 步数与定位能力（关闭图像切分；定位答对题数与平均 IoU）",
                        tpl, cm=13.0)

    p = find_para(body_items(doc), "5.2 RSI 闭环带来了什么")
    set_para_text(p, (
        f"5.2 RSI 闭环带来了什么：一次性验证器纠错把 SFT 的 {SFT['total']}/15 提到 "
        f"{CORR['total']}/15（触发 {CORR['n_tool']} 次、改对 {CORR['n_fixed']} 次，"
        f"净增益 {CORR['total'] - SFT['total']:+d}），改对集中在 exist 类："
        f"{bt(SFT, 'exist')[0]}/3 → {CORR['by_type']['exist']['ok']}/3——工具给出「该颜色连通块 1 处」"
        f"后，模型有依据地回答「有」，不再一律 No.，这是「工具给证据、模型做决策」的直接体现。"
        f"两轮 RSI 把「原本错、纠错后对」的样本入池：轮 1 收 {R1['new_data']} 条、轮 2 收 "
        f"{R2['new_data']} 条，数据池 {R1['pool_before']} → {R1['pool_after']} → {R2['pool_after']} 条；"
        f"再训练后轮末评测（不带纠错）为 {R1['round_end_total']}/15 → {R2['round_end_total']}/15。"
        f"学到的东西可以点名：轮 1 入池的 L08（第4行第4列）在轮 2 被直接答对；exist 类从 0/3 升到 "
        f"{bt(R2['eval'], 'exist')[0]}/3（学会「有……」的答法）。但增益只有 "
        f"+{R2['round_end_total'] - R1['round_end_total']} 题量级，且第二轮明显小于第一轮。"
    ))

    p = find_para(body_items(doc), "5.3 失败模式分析")
    set_para_text(p, (
        "5.3 失败模式分析：① 枚举退化（最严重）——SFT 后模型把答案写成「第1行第2列第3列第4列…」，"
        "解析器取第一个格号，于是「gold 恰好是第一格」的题被误判为对、其余全错；说明 60 步只学到"
        "格式表层，没学到位置。② 相邻格/跨格混淆——验证器对 L01 的目标格（第1行第2列）与相邻格"
        "（第2行第2列）都判通过（后者覆盖目标 ≥50% 面积），模型答错时不一定触发纠错"
        f"（两轮中被「否认」的入池样本共 {R1['n_verifier_denied'] + R2['n_verifier_denied']} 条）。"
        "③ 数数题饱和——评测集 4 道 count 题的 gold 全是「1 个」（题目构造取 "
        "present[i % len(present)]，恰好都只出现一次），基线输出「1. 2. 3. …」被宽松判分取首位"
        "数字也算对，因此 count 一列从基线起就没有提升空间。④ 存在性题的「假提升」——纠错后模型"
        "输出「有508个颜色，连通块 1处。」，本质在复读提示词，判分通过但不算真会判断；说明"
        "「工具给证据」容易被模型当成「抄答案」。⑤ 自查几乎无效——把高亮框回灌后模型大多仍回"
        f"「第1行第2列第3列…」，自查改写次数虽记了 {R1['n_self']} + {R2['n_self']} 次，"
        "但真正靠自查改对的定位题为 0。"
    ))

    p = find_para(body_items(doc), "5.4 回答核心问题")
    set_para_text(p, (
        f"5.4 核心问题的回答：在本实验条件下，提升主要来自「微调学会答案格式」+「推理期验证器纠错」"
        f"两处，RSI 的「回流再训练」只贡献很小一块。证据链：① 基线 → SFT，总正确率 "
        f"{BASE['total']} → {SFT['total']}、IoU {BASE['mean_iou']} → {SFT['mean_iou']}，但 loc 的 2 个"
        f"「命中」是解析假命中（5.3），真实定位能力仍接近 0；② SFT → SFT+纠错，"
        f"+{CORR['total'] - SFT['total']} 题全部来自 exist 类，工具证据起决定作用；"
        f"③ 两轮 RSI 回流再训练后，轮末 {R1['round_end_total']} → {R2['round_end_total']}"
        f"（+{R2['round_end_total'] - R1['round_end_total']}），第二轮增益小于第一轮；"
        f"④ 步数扫描（表 5、图 3）显示 loc 随训练量单调上升（10 步 0/8 → 60 步 2/8 → 200 步 4/8，"
        f"IoU 0 → 0.14），说明「会不会定位」的决定性变量是 SFT 训练量，而不是闭环本身——"
        f"闭环只能把「模型已答对的纠错结果」再喂回去，无法凭空造出定位能力。"
    ))

    # ============================================================ 六、思考题
    answers = [
        "若把 4×4 改成 8×8，格数从 16 变 64、单格从 64 px 缩到 32 px。按本实验的实测趋势推："
        "① loc 正确率不会自动上升——模型连 4×4 的「位置」都没学稳（60 步时输出还是枚举串，"
        "200 步才到 4/8），格数变多只会让随机命中更难（1/16 → 1/64）；② IoU 的上限会更细，"
        "但前提是模型先学会粗定位；③ 代价是输出空间 ×4，SFT 需要更多样本与步数，验证器在更小"
        "的格上更难过「覆盖 ≥50%」判据，假阴性增多、纠错触发更频繁但更难改对。"
        "结论：小模型 + 小数据时应「先粗后细」。（可运行入口：--grid 8 --tag _g8。）",

        "验证器「帮倒忙」的具体例子：L01 的目标格是第 1 行第 2 列，但验证器对第 2 行第 2 列也判"
        "「通过」（该格覆盖目标 ≥50% 面积），所以模型答成相邻格时不会触发纠错，纠错机会被吞掉；"
        "另一类是同色物体重叠/遮挡时颜色连通块合并或断裂（本实验 e04/e05 两张图都出现过），"
        "数数提示里的「连通块共几处」本身就不对。还有「抄答案」风险：exist 题纠错后模型直接复读"
        "「有508个颜色，连通块 1处。」就判对了。缓解办法：工具只给可核查的量、不直接给答案；"
        f"对 gold 做验证器自检（本实验 gold 格 {VGOLD} 全通过，说明假阴性率为 0，"
        "但假阳性——相邻格也判通过——仍然存在）。",

        f"因为在本实验里「教师」是现成的：解析式验证器零训练、零标注、单次调用毫秒级，"
        f"模型每答错一题就免费得到一次「错在哪」的反馈，答对的版本还能变成新训练数据——"
        f"两轮数据池只从 {R1['pool_before']} 条涨到 {R2['pool_after']} 条，增长几乎零成本。"
        f"它「不再免费」有三个时刻：① 验证器开始不可靠（真实照片要换成学习式检测器，"
        f"检测器本身要标注与训练）；② 自举数据分布变窄（池子里近一半是同一类 exist 题，"
        f"模型在别处退化）；③ 回流训练要算力（每轮 60 步在 CPU 上约 "
        f"{TIM['rsi_train_seconds'][0]:.0f} 秒）。第二轮增益（+1）远小于第一轮（+3），"
        f"正是「免费教师」开始变贵的信号。",

        "搬到 ChartQA 数值题，验证器必须能「脱离模型算出答案」：① 用图表解析把图变成结构化数据"
        "（轴范围、图例、柱子/折线像素坐标 → 数值）；② 对最大值/差值/增长率这类问题，用解析出的"
        "数值直接算出参考值，作为「可核查证据」写进提示词（如「三根柱子分别约 12/18/15」），"
        "而不是直接给答案；③ 判分器按相对误差 ≤1% 容差判对错。闭环结构不变：模型作答 → "
        "解析器校验 → 回灌证据 → 模型重答 → 通过验证的样本入池 → 再训练。风险同构：解析器对"
        "复杂图表（多系列、堆叠柱、双轴）会出错，错误证据会把模型带偏，因此仍需 gold 自检与"
        "「工具只给证据」的设计。",
    ]
    for i, txt in enumerate(answers, start=1):
        if find_para(body_items(doc), f"思考题 {i}（") is None:
            continue
        ans = find_para(body_items(doc), "答：____")
        if ans is None:
            break
        set_para_text(ans, "答：" + txt)

    # ============================================================ 七、结论
    p = find_para(body_items(doc), "「值不值」")
    set_para_text(p, (
        f"结论与反思（值不值）：在 256M 级小模型上，「按网格格式定位」是学得会的——只要把 SFT 从"
        f"指南建议的 10 步加到 60 / 200 步，loc 就从 0/8 走到 2/8、4/8（IoU 0 → 0.14），"
        f"代价只是 CPU 上多花几分钟；但 RSI 闭环的收益小于预期：它能在推理期用工具证据一次修好 "
        f"{CORR['n_fixed']} 道题（主要是 exist 类），回流再训练只带来 "
        f"+{R2['round_end_total'] - R1['round_end_total']} 的轮间增益，且第二轮更小。"
        f"所以答案是：SFT 值、闭环主要值在「推理期纠错」这一段。主要局限：① 256M 的视觉定位"
        f"天花板（枚举退化、相邻格混淆）；② 解析式验证器只对纯色合成图成立，有假阴性/假阳性；"
        f"③ 4×4 粗离散化 + 训练集即评测集，记忆容易混进指标；④ count 列被「1 个」饱和。"
        f"下一步：换 8×8 网格并加大 SFT 数据、验证器升级为多判据并做 gold 自检、自举数据按题型"
        f"分层并引入留出集评测，才能看清 RSI 的真实上限。"
    ))

    # ============================================================ 附录
    last = None
    for k, o in body_items(doc):
        if k == "p":
            last = o
    apx = [
        "附录 A　复现命令",
        "python hw6_rsi_localize.py --mode make_data                            # 55 张图 + GT",
        "python hw6_rsi_localize.py --mode all --rounds 2 --steps 60 --no_split 1   # 主实验全链路",
        "python hw6_rsi_localize.py --mode cases                                # 逐题/逐轮对照",
        "python hw6_rsi_localize.py --mode figures                              # 图 1~3",
        "python fill_report_hw6.py                                              # 生成本报告",
        "附录 B　运行日志节选（run_main6.log）",
        f"基线 15 题：loc {bt(BASE, 'loc')[0]}/8、count {bt(BASE, 'count')[0]}/4、"
        f"exist {bt(BASE, 'exist')[0]}/3，IoU {BASE['mean_iou']}（{BASE['seconds']:.0f}s）",
        f"SFT {SFT_T['steps']} 步（{SFT_T['epochs']} 轮）：loss {SC[0][1]:.2f} → {SC[-1][1]:.2f}"
        f"（{SFT_T['seconds']:.0f}s），可训练参数占比 {P['ratio_pct']:.3f}%",
        f"SFT 评测：loc {bt(SFT, 'loc')[0]}/8、count {bt(SFT, 'count')[0]}/4、"
        f"exist {bt(SFT, 'exist')[0]}/3，IoU {SFT['mean_iou']}",
        f"SFT+纠错：{SFT['total']}/15 → {CORR['total']}/15（触发 {CORR['n_tool']}、"
        f"改对 {CORR['n_fixed']}）",
        f"RSI 轮1：直接答对 {R1['n_ok1']}/15、自查 {R1['n_self']}、验证器纠错 {R1['n_tool']}、"
        f"纠错后 {R1['n_ok1'] + R1['n_fixed']}/15、新增 {R1['new_data']} 条、轮末 "
        f"{R1['round_end_total']}/15",
        f"RSI 轮2：直接答对 {R2['n_ok1']}/15、自查 {R2['n_self']}、验证器纠错 {R2['n_tool']}、"
        f"纠错后 {R2['n_ok1'] + R2['n_fixed']}/15、新增 {R2['new_data']} 条、轮末 "
        f"{R2['round_end_total']}/15",
        f"补充对照：开启切分 10 步时 SFT loc {bt(SPLIT10, 'loc')[0]}/8；关闭切分 200 步时 loc "
        f"{bt(STEP200, 'loc')[0]}/8（IoU {STEP200['mean_iou'] if STEP200 else '—'}）",
        f"全链路总耗时 {TIM['all_seconds'] / 60:.1f} 分钟（关闭切分后单题生成约 2~8 s）",
    ]
    prev = last._element
    for line in apx:
        pobj = insert_para_after(prev, doc, line, template_el=tpl,
                                 mono=line.startswith("python") or line.startswith("附录"),
                                 size=9 if line.startswith("python") else None)
        prev = pobj._element

    doc.save(OUT)
    print("已生成：", OUT)


if __name__ == "__main__":
    main()
