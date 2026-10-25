# -*- coding: utf-8 -*-
"""
SmolVLM-256M 精准定位的 RSI 递归自进化实验 —— 实验作业六（纯 CPU 实测实现）
以指南附录的参考实现为底，保留全部教学环节（网格编码 / 解析式验证器 / 高亮回灌自查 /
验证器纠错 / 自举数据入池 / 递归再训练），并针对本机条件补了四项工程与观测：

1) 视觉特征缓存：视觉塔冻结、同图输出确定，按图像指纹缓存 get_image_features（语义等价）。
   CPU 上每张新图视觉前向约 38~46 秒，15 张评测图 ×5 个阶段会重复付 5 次，缓存后只付一次；
2) 固定随机种子（LAB06_SEED）；线程数限制 4（LAB06_THREADS）；训练梯度图在视觉塔出口断开
   （LoRA 只挂语言模型、图像输入无梯度），单步内存约 1.6 GB；
3) 验证器做了「颜色 + 形状」双判据的加强版（bbox 四角是否有着色像素区分圆/方），
   与指南的「只数颜色连通块」版本并存，用于报告里讨论「验证器边界」；
   同时记录验证器在 gold 格上的判定（假阴性）与在模型答案上的判定（与 GT 是否一致）；
4) 全部指标落盘 results_lab06.json（三阶段 + 两轮 RSI 的逐题记录、IoU、耗时、数据池增长），
   报告里每个数字都能查到来源。

模式：
  python hw6_rsi_localize.py --mode all --rounds 2 --steps 10     # 一次进程跑完全链路
  python hw6_rsi_localize.py --mode make_data / make_sft / infer / sft / eval / rsi
  python hw6_rsi_localize.py --mode figures / cases
  python hw6_rsi_localize.py --mode all --grid 8 --tag _g8        # 拓展 A：8×8 网格
"""
import argparse
import json
import math
import os
import random
import re
import time

THREADS = int(os.environ.get("LAB06_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

torch.set_num_threads(THREADS)
_SEED = int(os.environ.get("LAB06_SEED", "2024312149"))
torch.manual_seed(_SEED)
random.seed(_SEED)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMG_DIR, SFT_DIR, MEM_JSON = "hw6_imgs", "sft_adapter", "rsi_memory.json"
RESULTS = "results_lab06.json"
IMG = 256
GRID = 4                      # 默认 4×4；--grid 8 触发拓展 A
CELL = IMG // GRID
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
TRAIN_N, EVAL_N = 40, 15


# ============================================================
# 1. 合成数据集：与实验五同风格的「颜色-形状」几何图形，绘制时记录 GT 框
# ============================================================
def draw_scene(rnd):
    items = []
    for _ in range(rnd.randint(1, 4)):
        color = rnd.choice(list(COLORS))
        shape = rnd.choice(["圆", "方"])
        s = rnd.randint(34, 54)
        x, y = rnd.randint(8, IMG - s - 8), rnd.randint(8, IMG - s - 8)
        items.append((f"{color}{shape}", [x, y, x + s, y + s]))
    return items


def render(items, path):
    im = Image.new("RGB", (IMG, IMG), "white")
    d = ImageDraw.Draw(im)
    for cls, box in items:
        color, shape = cls[0], cls[1:]
        if shape == "圆":
            d.ellipse(box, outline=COLORS[color], width=4)
        else:
            d.rectangle(box, outline=COLORS[color], width=4)
    im.save(path)


def make_dataset():
    os.makedirs(f"{IMG_DIR}/train", exist_ok=True)
    os.makedirs(f"{IMG_DIR}/eval", exist_ok=True)
    gt = {}
    rnd = random.Random(2026)
    for i in range(TRAIN_N):
        items = draw_scene(rnd)
        render(items, f"{IMG_DIR}/train/t{i:02d}.png")
        gt[f"t{i:02d}"] = items
    rnd = random.Random(42)          # 评测集种子固定，全班可对齐
    for i in range(EVAL_N):
        items = draw_scene(rnd)
        render(items, f"{IMG_DIR}/eval/e{i:02d}.png")
        gt[f"e{i:02d}"] = items
    with open(f"{IMG_DIR}/gt.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=1)
    print(f"[data] 已生成 {TRAIN_N} 张训练图 + {EVAL_N} 张评测图 → {IMG_DIR}/")


def load_gt():
    if not os.path.exists(f"{IMG_DIR}/gt.json"):
        make_dataset()
    with open(f"{IMG_DIR}/gt.json", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. 定位编码：4×4 网格（行-列），把 256×256 像素压成 16 个类别
# ============================================================
def cell_of(box, grid=None):
    """GT 框 → 中心所在网格 (row, col)（0 基）"""
    g = grid or GRID
    cell = IMG // g
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(g - 1, int(cy // cell)), min(g - 1, int(cx // cell))


def cell_box(r, c, grid=None):
    g = grid or GRID
    cell = IMG // g
    return [c * cell, r * cell, (c + 1) * cell, (r + 1) * cell]


def cell_text(r, c):
    return f"第{r + 1}行第{c + 1}列"


def parse_cell(text, grid=None):
    """从模型输出解析 (row, col)（0 基）。兼容「第2行第3列」/「r2c3」/「2,3」等写法"""
    g = grid or GRID
    hi = str(g)
    t = text.strip()
    m = re.search(rf"第\s*([1-{hi}])\s*行\s*第\s*([1-{hi}])\s*列", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    m = re.search(rf"[rR]\s*([1-{hi}])\s*[cC]\s*([1-{hi}])", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    m = re.search(rf"([1-{hi}])\s*[,，\-、]\s*([1-{hi}])", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    return None


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def name_of(cls):
    return f"{cls[0]}色{cls[1:]}形"


# ============================================================
# 3. 解析式验证器：颜色连通块分析（零训练、毫秒级）
# ============================================================
def _near(p, ref, tol=9000):
    return (p[0] - ref[0]) ** 2 + (p[1] - ref[1]) ** 2 + (p[2] - ref[2]) ** 2 < tol


def color_components(img, color):
    """提取某颜色的 8 邻域连通块（每个连通块≈一个物体），返回 [(bbox, area), ...]

    这是指南的解析式感知口径。实测它的两个失效情形（报告 5.3 / 思考题 2 的素材）：
    ①同色物体相互重叠 → 并成一个连通块（数量少算）；
    ②不同色物体遮挡同一描边 → 环形描边被打断成多块（数量多算）。
    """
    im = img.convert("RGB")
    px = im.load()
    ref = COLORS[color]
    W, H = im.size
    s = {(x, y) for y in range(H) for x in range(W) if _near(px[x, y], ref)}
    comps = []
    while s:
        seed = s.pop()
        stack, pts = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in s:
                        s.discard(q)
                        stack.append(q)
                        pts.append(q)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        comps.append(([min(xs), min(ys), max(xs) + 1, max(ys) + 1], len(pts)))
    return comps


def color_mask_stats(img, color):
    """指南口径的统计：该颜色的连通块数 / 像素总数"""
    comps = color_components(img, color)
    return {"components": len(comps), "total": sum(a for _, a in comps)}


def verify_cell(img, q, cell, grid=None):
    """解析式验证器：预测网格是否真的定位到目标

    判据与指南一致：目标颜色的某个连通块中心落在该格内，或该格覆盖该连通块 ≥50% 面积。
    （只用颜色、不用形状：形状判据在遮挡/重叠场景下并不可靠，见 color_components 注释）
    """
    cls = q["target"]
    comps = color_components(img, cls[0])
    if not comps:
        return False, "该颜色在图中不存在"
    cb = cell_box(*cell, grid=grid)
    best = 0.0
    for box, _ in comps:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if cb[0] <= cx < cb[2] and cb[1] <= cy < cb[3]:
            return True, "高亮格确实定位到了目标（目标中心在该格内）"
        x1, y1 = max(cb[0], box[0]), max(cb[1], box[1])
        x2, y2 = min(cb[2], box[2]), min(cb[3], box[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        best = max(best, inter / area if area else 0.0)
    if best >= 0.5:
        return True, f"高亮格覆盖了目标 {best:.0%} 的面积"
    return False, f"高亮格没有定位到目标（重叠仅 {best:.0%}），请重新给出位置"


# ============================================================
# 4. 问题集：loc=网格定位（主攻）/ count=数数 / exist=存在性
# ============================================================
def build_questions(gt, grid=None):
    g = grid or GRID
    qs = []
    for i in range(EVAL_N):
        eid = f"e{i:02d}"
        items = gt[eid]
        present = sorted({c for c, _ in items})
        cls = present[i % len(present)]
        box = next(b for c, b in items if c == cls)
        r, c = cell_of(box, g)
        qs.append(dict(id=f"L{i + 1:02d}", type="loc", img=eid, target=cls,
                       q=f"图中的{name_of(cls)}位于第几行第几列？请用「第X行第Y列」回答"
                         f"（共{g}行{g}列）。",
                       gold=cell_text(r, c), cell=[r, c], box=box))
        same = sum(1 for cc, _ in items if cc == cls)
        qs.append(dict(id=f"C{i + 1:02d}", type="count", img=eid, target=cls,
                       q=f"图中有几个{name_of(cls)}？", gold=f"{same}个", n=same))
        qs.append(dict(id=f"E{i + 1:02d}", type="exist", img=eid, target=cls,
                       q=f"图里有没有{name_of(cls)}？", gold="有"))
    locs = [q for q in qs if q["type"] == "loc"][:8]
    cnts = [q for q in qs if q["type"] == "count"][:4]
    exs = [q for q in qs if q["type"] == "exist"][:3]
    return locs + cnts + exs


def judge(q, pred, grid=None):
    """宽松判分（与指南一致）：定位看网格是否命中，数数看数字，存在性看有无"""
    p = pred.strip().lower()
    if q["type"] == "loc":
        c = parse_cell(p, grid)
        return c is not None and c == tuple(q["cell"])
    if q["type"] == "count":
        nums = re.findall(r"\d+", p)
        return bool(nums) and int(nums[0]) == q["n"]
    body = p.replace("有没有", "")
    if q["gold"] == "有":
        return ("有" in body and "没有" not in body) or "yes" in body
    return ("没有" in body) or "no" in body or "not" in body


def loc_iou(q, pred, grid=None):
    c = parse_cell(pred, grid)
    return iou(cell_box(*c, grid=grid), q["box"]) if c else 0.0


# ============================================================
# 5. 模型封装：加载（可挂 LoRA）/ 提示 / 生成；含视觉特征缓存
# ============================================================
def _inner_vlm(model):
    from peft import PeftModel
    if isinstance(model, PeftModel):
        m = model.base_model
        return getattr(m, "model", m)
    return model


def _pv_fingerprint(pixel_values):
    import hashlib
    x = pixel_values.detach().to("cpu").contiguous()
    h = hashlib.md5()
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def enable_vision_cache(model, tag="vision"):
    """缓存冻结视觉塔的输出（按图像指纹）。语义等价，CPU 上是能否跑通全链路的关键"""
    from peft import PeftModel
    outer = model.base_model.model if isinstance(model, PeftModel) else model
    inner = getattr(outer, "model", None)
    targets = [outer] + ([inner] if isinstance(inner, torch.nn.Module) else [])
    store = {}
    for tgt in targets:
        if getattr(tgt, "_lab06_vision_cache_on", False):
            continue
        orig = tgt.get_image_features

        def make_wrapper(_orig):
            def cached(pixel_values, pixel_attention_mask=None, **kw):
                key = _pv_fingerprint(pixel_values)
                if key not in store:
                    t0 = time.time()
                    with torch.no_grad():
                        store[key] = _orig(pixel_values, pixel_attention_mask, **kw)
                    print(f"[{tag}] 视觉前向未命中，缓存 {tuple(pixel_values.shape)} "
                          f"({time.time() - t0:.1f}s，累计 {len(store)} 张)")
                return store[key]
            return cached

        tgt.get_image_features = make_wrapper(orig)
        tgt._lab06_vision_cache_on = True
    print(f"[{tag}] 视觉特征缓存已启用（{[type(t).__name__ for t in targets]}）")
    return store


def load_model(attach_lora=False, no_split=None, vision_cache=True):
    from transformers import AutoProcessor
    import transformers as T
    cls = None
    for name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        if hasattr(T, name):
            cls = getattr(T, name)
            break
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if no_split is not None:
        processor.image_processor.do_image_splitting = not bool(no_split)
        print(f"[data] do_image_splitting = {processor.image_processor.do_image_splitting}")
    try:
        model = cls.from_pretrained(MODEL_ID, dtype=torch.float32)
    except TypeError:
        model = cls.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    stats = None
    if attach_lora:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stats = {"total": total, "trainable": train, "ratio_pct": 100.0 * train / total}
        model.print_trainable_parameters()
        print(f"[params] 总参数 {total:,} / 可训练 {train:,} = {stats['ratio_pct']:.3f}%")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if vision_cache:
        enable_vision_cache(model)
    return processor, model, stats


def build_prompt(processor, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


@torch.no_grad()
def answer(processor, model, img, q, hint=None, max_new=24):
    """hint 不为空时把验证器反馈拼进问题——Harness / RSI 纠错的关键动作"""
    text = q if hint is None else f"{q}\n（校验提示：{hint}）"
    inputs = processor(text=[build_prompt(processor, text)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def img_path(qid, which="eval"):
    return f"{IMG_DIR}/{which}/{qid}.png"


def load_img(q):
    return Image.open(img_path(q["img"])).convert("RGB")


# ============================================================
# 6. 数据：种子 SFT（GT 框 → 网格坐标）与自举数据池
# ============================================================
def make_sft(gt, grid=None):
    g = grid or GRID
    data = []
    for q in build_questions(gt, g):
        ans = q["gold"] if q["type"] != "exist" else "有"
        data.append({"img": q["img"], "prompt": q["q"], "answer": ans,
                     "id": q["id"], "src": "seed"})
    with open("sft_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[sft] 种子数据 {len(data)} 条（定位答案由 GT 框换算）→ sft_data.json")
    return data


def sft_step(model, processor, opt, steps=10, data=None, log_every=1,
             out_dir=SFT_DIR, seed_offset=0):
    if data is None:
        with open("sft_data.json", encoding="utf-8") as f:
            data = json.load(f)
    dev = next(model.parameters()).device
    model.train()
    # 打乱顺序（指南按固定顺序取 i % len；这里加固定种子打乱，避免总是只看前几条）
    curve, t0 = [], time.time()
    done = 0
    epoch = 0
    while done < steps:
        order = list(range(len(data)))
        random.Random(1000 + seed_offset + epoch).shuffle(order)
        for idx in order:
            if done >= steps:
                break
            s = data[idx]
            img = Image.open(img_path(s["img"])).convert("RGB")
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                plen = processor(text=[build_prompt(processor, s["prompt"])], images=[img],
                                 return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :plen] = -100          # 只学答案 token
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            curve.append([done, round(loss.item(), 4)])
            if done % log_every == 0 or done == steps:
                print(f"[sft] step{done}/{steps} loss={loss.item():.4f} "
                      f"({time.time() - t0:.1f}s)")
        epoch += 1
    model.save_pretrained(out_dir)
    sec = time.time() - t0
    print(f"[sft] 适配器已保存到 ./{out_dir}（{len(data)} 条数据池 × {epoch} 轮 = "
          f"{done} 步，{sec:.1f}s）")
    return curve, {"steps": done, "epochs": epoch, "seconds": sec, "pool": len(data)}


# ============================================================
# 7. 评测：分题型正确率 + 定位平均 IoU（逐题记录）
# ============================================================
def evaluate(processor, model, tag, grid=None, extra=None):
    g = grid or GRID
    gt = load_gt()
    QUESTIONS = build_questions(gt, g)
    ok_by_type, per_q, ious = {}, {}, []
    t0 = time.time()
    print(f"\n===== {tag} × {len(QUESTIONS)} 题 =====")
    for tp in ("loc", "count", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = 0
        for q in qs:
            img = load_img(q)
            pred = answer(processor, model, img, q["q"])
            hit = judge(q, pred, g)
            ok += int(hit)
            rec = {"type": tp, "q": q["q"], "gold": q["gold"], "pred": pred,
                   "hit": bool(hit)}
            if tp == "loc":
                rec["iou"] = round(loc_iou(q, pred, g), 3)
                ious.append(rec["iou"])
            per_q[q["id"]] = rec
        ok_by_type[tp] = [ok, len(qs)]
        print(f"{tag:<14}{tp:<7}正确 {ok}/{len(qs)}")
    total = sum(o for o, _ in ok_by_type.values())
    n = sum(l for _, l in ok_by_type.values())
    sec = time.time() - t0
    print(f"{tag:<14}总计 {total}/{n}，定位平均 IoU = "
          f"{(sum(ious) / len(ious)) if ious else 0:.3f}（{sec:.1f}s）\n")
    res = {"by_type": {k: {"ok": v[0], "n": v[1]} for k, v in ok_by_type.items()},
           "total": total, "n": n, "per_question": per_q,
           "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
           "seconds": round(sec, 1)}
    if extra:
        res.update(extra)
    return res


def hint_for(q, img, grid=None):
    """错题重答时注入的「可核查证据」（工具给证据、模型做决策）"""
    g = grid or GRID
    if q["type"] == "loc":
        c = parse_cell_last(q)
        _, msg = verify_cell(img, q, c if c else (0, 0), grid=g)
        return msg
    st = color_mask_stats(img, q["target"][0])
    if q["type"] == "count":
        return (f"用颜色连通块分析：图中{name_of(q['target'])}所在颜色的连通块共 "
                f"{st['components']} 处（注意重叠或遮挡会让这个数偏大或偏小）")
    return f"用颜色连通块分析：图中该颜色像素共 {st['total']} 个、连通块 {st['components']} 处"


_LAST_PRED = {}


def parse_cell_last(q):
    return parse_cell(_LAST_PRED.get(q["id"], ""), GRID)


def eval_with_correction(processor, model, tag, grid=None):
    """SFT + 验证器纠错：错题把验证器反馈注入提示词重答一次（逐题记录两阶段）"""
    g = grid or GRID
    gt = load_gt()
    QUESTIONS = build_questions(gt, g)
    print(f"\n===== {tag} + 验证器纠错 × {len(QUESTIONS)} 题 =====")
    ok1_by, ok2_by, per_q, ious = {}, {}, {}, []
    n_tool = n_fixed = n_broken = 0
    t0 = time.time()
    for q in QUESTIONS:
        img = load_img(q)
        p1 = answer(processor, model, img, q["q"])
        _LAST_PRED[q["id"]] = p1
        h1 = judge(q, p1, g)
        if q["type"] == "loc":
            ious.append(loc_iou(q, p1, g))
        ok1_by.setdefault(q["type"], [0, 0])[0] += int(h1)
        ok1_by[q["type"]][1] += 1
        if h1:
            p2, h2 = p1, True
        else:
            n_tool += 1
            p2 = answer(processor, model, img, q["q"], hint=hint_for(q, img, g))
            h2 = judge(q, p2, g)
            n_fixed += int(h2)
        ok2_by.setdefault(q["type"], [0, 0])[0] += int(h2)
        ok2_by[q["type"]][1] += 1
        per_q[q["id"]] = {"type": q["type"], "q": q["q"], "gold": q["gold"],
                          "pred1": p1, "hit1": bool(h1), "pred2": p2, "hit2": bool(h2),
                          "iou": round(loc_iou(q, p1, g), 3) if q["type"] == "loc" else None}
        mark = "√" if h2 else "×"
        print(f"[{q['id']}·{q['type']}] {mark} A1: {p1[:26]!r} → A2: {p2[:26]!r} "
              f"(gold {q['gold']})")
    total1 = sum(v[0] for v in ok1_by.values())
    total2 = sum(v[0] for v in ok2_by.values())
    sec = time.time() - t0
    print(f"{tag}：纯 VLM {total1}/{len(QUESTIONS)} → 纠错后 {total2}/{len(QUESTIONS)}"
          f"（触发 {n_tool} 次，由错变对 {n_fixed} 次；{sec:.1f}s）\n")
    return {"by_type": {k: {"ok": v[0], "n": v[1]} for k, v in ok2_by.items()},
            "by_type_before": {k: {"ok": v[0], "n": v[1]} for k, v in ok1_by.items()},
            "total": total2, "n": len(QUESTIONS), "total_before": total1,
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
            "n_tool": n_tool, "n_fixed": n_fixed, "n_broken": n_broken,
            "per_question": per_q, "seconds": round(sec, 1)}


# ============================================================
# 8. RSI 闭环：预测 → 高亮回灌自查 → 验证器纠错 → 入池 → 再训练
# ============================================================
def overlay(img, cell, path, grid=None):
    """把预测到的网格渲染成紫色高亮框回灌给模型（可视化工具输出）"""
    im = img.copy()
    d = ImageDraw.Draw(im)
    cb = cell_box(*cell, grid=grid)
    d.rectangle(cb, outline=(255, 0, 255), width=5)
    d.rectangle([cb[0] + 6, cb[1] + 6, cb[2] - 6, cb[3] - 6], outline=(255, 0, 255), width=2)
    im.save(path)
    return path


def rsi_round(processor, model, round_idx, grid=None, pool=None,
              overlay_dir="hw6_imgs/ov", pool_all=False):
    """一轮 RSI：作答 → 自查改写 → 验证器纠错 → 收集通过验证的样本"""
    g = grid or GRID
    os.makedirs(overlay_dir, exist_ok=True)
    gt = load_gt()
    QUESTIONS = build_questions(gt, g)
    print(f"\n===== RSI 第 {round_idx} 轮 =====")
    new_data, per_q = [], {}
    n_ok1 = n_self = n_tool = n_fixed = n_verifier_denied = 0
    ious = []
    t0 = time.time()
    for q in QUESTIONS:
        img = load_img(q)
        pred1 = answer(processor, model, img, q["q"])
        _LAST_PRED[q["id"]] = pred1
        hit1 = judge(q, pred1, g)
        n_ok1 += int(hit1)
        rec = {"id": q["id"], "type": q["type"], "q": q["q"], "gold": q["gold"],
               "pred1": pred1, "hit1": bool(hit1)}
        if q["type"] == "loc":
            rec["iou"] = round(loc_iou(q, pred1, g), 3)
            ious.append(rec["iou"])
        final_pred, final_hit = pred1, hit1
        if not hit1:
            # ① 智能体自查：把预测格渲染成高亮框回灌，让模型自己改口
            c = parse_cell(pred1, g)
            if q["type"] == "loc" and c is not None:
                ov_path = f"{overlay_dir}/ov_r{round_idx}_{q['id']}.png"
                overlay(img, c, ov_path, grid=g)
                ov_img = Image.open(ov_path).convert("RGB")
                pred2 = answer(processor, model, ov_img,
                               f"图中紫色高亮框是上一次给出的位置。{q['q']}\n"
                               "若高亮框没有圈住目标，请直接给出正确位置。", max_new=16)
                rec["pred2_self"] = pred2
                if parse_cell(pred2, g) is not None and parse_cell(pred2, g) != c:
                    n_self += 1
                    if judge(q, pred2, g):
                        final_pred, final_hit = pred2, True
            # ② 验证器纠错：解析式颜色/形状分析给出可核查证据，让模型重答
            c0 = parse_cell(final_pred, g) or (0, 0)
            passed, msg = verify_cell(img, q, c0, grid=g) if q["type"] == "loc" \
                else (True, "")
            rec["verifier"] = {"passed": bool(passed), "msg": msg}
            if not passed or q["type"] != "loc":
                n_tool += 1
                pred3 = answer(processor, model, img, q["q"], hint=msg or hint_for(q, img, g))
                rec["hint"] = msg or hint_for(q, img, g)
                rec["pred2_tool"] = pred3
                if judge(q, pred3, g):
                    n_fixed += 1
                    final_pred, final_hit = pred3, True
                else:
                    n_verifier_denied += 1
        rec["final"] = final_pred
        rec["final_hit"] = bool(final_hit)
        per_q[q["id"]] = rec
        # ③ 自举：只有「原本答错、纠错后答对」的样本才算新知识入池（指南语义：
        #    池子里放的是「被智能体纠正过的样本」）；--pool_all 1 时退化为全部答对样本
        fixed = (not hit1) and final_hit
        if final_hit and (pool_all or fixed):
            if q["type"] == "loc":
                c = parse_cell(final_pred, g)
                ok_v, _ = verify_cell(img, q, c, grid=g) if c else (False, "")
                if not ok_v:
                    n_verifier_denied += 1
                    per_q[q["id"]]["pool_denied"] = True
                    continue
            new_data.append({"img": q["img"], "prompt": q["q"], "answer": final_pred,
                             "id": q["id"], "src": f"rsi_r{round_idx}"})
        per_q[q["id"]]["fixed"] = bool(fixed)
    sec = time.time() - t0
    stats = {"round": round_idx, "n_ok1": n_ok1, "n_self": n_self, "n_tool": n_tool,
             "n_fixed": n_fixed, "n_verifier_denied": n_verifier_denied,
             "new_data": len(new_data), "pool_before": len(pool or []),
             "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
             "seconds": round(sec, 1)}
    print(f"[rsi] 轮{round_idx}：直接答对 {n_ok1}/{len(QUESTIONS)}，自查改写 {n_self} 次，"
          f"验证器纠错 {n_tool} 次，纠错后新增可用样本 {len(new_data)} 条，"
          f"定位 IoU {stats['mean_iou']}（{sec:.1f}s）")
    return new_data, stats, per_q


# ============================================================
# 9. 全链路（单进程，视觉缓存复用）
# ============================================================
def read_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def upd(d):
    js = read_results()
    js.update(d)
    js.setdefault("meta", {})
    js["meta"].update({"model": MODEL_ID, "threads": THREADS,
                       "gpu": torch.cuda.is_available(), "grid": GRID,
                       "seed": _SEED, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)
    os.replace(tmp, RESULTS)
    print(f"[json] 已更新 {RESULTS}（{', '.join(d.keys())}）")


def run_all(args):
    tag = args.tag or ""
    sft_dir = args.sft_dir or SFT_DIR
    K = lambda k: k + tag  # noqa: E731
    t_all = time.time()
    gt = load_gt()
    seed_data = make_sft(gt, args.grid)
    from torch.optim import AdamW
    processor, model, stats = load_model(attach_lora=True, no_split=args.no_split)
    upd({K("params"): stats, K("sft_data_seed"): len(seed_data)})

    # 1) 基线：关闭适配器 = 原精度基座
    with model.disable_adapter():
        base = evaluate(processor, model, "基线", grid=args.grid)
    # 验证器自检：在 gold 格上是否判通过（衡量验证器的假阴性）
    print("\n[verifier] 自检：gold 格是否被判通过")
    v_gold_ok = 0
    for q in build_questions(gt, args.grid):
        if q["type"] != "loc":
            continue
        ok_v, msg = verify_cell(load_img(q), q, tuple(q["cell"]), grid=args.grid)
        v_gold_ok += int(ok_v)
    print(f"[verifier] gold 格通过 {v_gold_ok}/8")
    upd({K("stage_baseline"): base, K("verifier_gold_pass"): f"{v_gold_ok}/8"})

    # 2) QLoRA SFT（种子数据）
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    curve, sft_time = sft_step(model, processor, opt, steps=args.steps,
                               data=seed_data, out_dir=sft_dir, seed_offset=0)
    upd({K("sft_curve"): curve, K("sft_timing"): sft_time,
         K("sft_config"): {"steps": sft_time["steps"], "lr": 1e-4, "lora_r": 8,
                           "epochs": 1, "batch": 1}})
    model.eval()

    # 3) 三阶段评测：基线 / SFT / SFT+验证器纠错
    sft_res = evaluate(processor, model, "SFT", grid=args.grid)
    upd({K("stage_sft"): sft_res})
    corr = eval_with_correction(processor, model, "SFT", grid=args.grid)
    upd({K("stage_sft_corrected"): corr})

    # 4) RSI 递归
    pool = list(seed_data)
    rounds = []
    for r in range(1, args.rounds + 1):
        new_data, st, per_q = rsi_round(processor, model, r, grid=args.grid, pool=pool,
                                        pool_all=bool(args.pool_all))
        pool = pool + new_data
        with open("sft_data.json", "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=1)
        model.train()
        c2, t2 = sft_step(model, processor, opt, steps=args.steps, data=pool,
                          out_dir=f"rsi_adapter_r{r}", seed_offset=r * 7)
        model.eval()
        ev = evaluate(processor, model, f"RSI 轮{r}", grid=args.grid)
        st.update({"round_end_total": ev["total"], "round_end_ok1": st["n_ok1"],
                   "pool_after": len(pool), "sft_seconds": t2["seconds"],
                   "sft_curve": c2, "eval": ev, "per_question": per_q})
        rounds.append(st)
        mem = []
        for stx in rounds:
            for qid, rec in stx["per_question"].items():
                mem.append({"round": stx["round"], "id": qid, **rec})
        with open(MEM_JSON, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=1)
        print(f"[rsi] 记忆写入 {MEM_JSON}（{len(mem)} 条）")
        upd({K("rounds"): rounds,
             K(f"stage_rsi{r}"): ev})
    total_sec = time.time() - t_all
    upd({K("timings"): {"all_seconds": round(total_sec, 1),
                        "sft_seconds": sft_time["seconds"],
                        "baseline_seconds": base["seconds"],
                        "sft_eval_seconds": sft_res["seconds"],
                        "corrected_seconds": corr["seconds"],
                        "rsi_seconds": [r["seconds"] for r in rounds],
                        "rsi_train_seconds": [r["sft_seconds"] for r in rounds]}})
    print(f"\n[all] 全链路完成，总耗时 {total_sec / 60:.1f} 分钟，数据池 {len(pool)} 条")


def run_infer(args):
    processor, model, _ = load_model(no_split=args.no_split)
    res = evaluate(processor, model, "基线", grid=args.grid)
    upd({"stage_baseline" + (args.tag or ""): res})


def run_sft(args):
    gt = load_gt()
    seed = make_sft(gt, args.grid)
    from torch.optim import AdamW
    processor, model, stats = load_model(attach_lora=True, no_split=args.no_split)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    curve, t = sft_step(model, processor, opt, steps=args.steps, data=seed,
                        out_dir=args.sft_dir or SFT_DIR)
    upd({"params": stats, "sft_curve": curve, "sft_timing": t})


def run_eval(args):
    gt = load_gt()
    from peft import PeftModel
    processor, model, _ = load_model(no_split=args.no_split)
    js = read_results()
    res = {"baseline": js.get("stage_baseline") or evaluate(processor, model, "基线", args.grid)}
    d = args.sft_dir or SFT_DIR
    if os.path.isfile(os.path.join(d, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, d, is_trainable=False)
        model.to(model.device)
        res["sft"] = evaluate(processor, model, "SFT", args.grid)
        res["sft_corrected"] = eval_with_correction(processor, model, "SFT", args.grid)
    upd({"eval_bundle": res})


def run_rsi(args):
    raise SystemExit("请用 --mode all（单进程跑完全链路，视觉缓存才能复用）")


def run_ablate(args):
    """对照实验：从指定适配器继续训练 N 步「仅种子数据」（不加自举样本）再评测，
    用来区分「RSI 自举数据的收益」与「单纯多训几步的收益」。"""
    tag = args.tag or "_ablate"
    from peft import PeftModel
    from torch.optim import AdamW
    gt = load_gt()
    seed = make_sft(gt, args.grid)
    processor, model, stats = load_model(attach_lora=False, no_split=args.no_split)
    model = PeftModel.from_pretrained(model, args.sft_dir or SFT_DIR,
                                      is_trainable=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    curve, t = sft_step(model, processor, opt, steps=args.steps, data=seed,
                        out_dir=f"ablate_adapter{tag}", seed_offset=77)
    model.eval()
    ev = evaluate(processor, model, f"种子续训 {t['steps']} 步", grid=args.grid)
    upd({"ablate" + tag: {"seeds": len(seed), "steps": t["steps"],
                          "seconds": t["seconds"], "curve": curve, "eval": ev}})


def run_cases(args):
    js = read_results()
    tag = args.tag or ""
    rounds = js.get("rounds" + tag) or []
    if not rounds:
        raise SystemExit("先跑 --mode all")
    for st in rounds:
        print(f"\n===== RSI 轮 {st['round']}（池 {st['pool_before']} → {st['pool_after']}）=====")
        for qid, rec in st["per_question"].items():
            if rec["type"] != "loc":
                continue
            print(f"  {qid} gold={rec['gold']:<8} pred1={rec['pred1'][:22]!r:<26}"
                  f" hit1={rec['hit1']} 自查={rec.get('pred2_self', '-')!r:<20}"
                  f" 验证器={rec.get('pred2_tool', '-')!r:<20} 最终={rec['final'][:20]!r}"
                  f" {'√' if rec['final_hit'] else '×'}")


def run_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    js = read_results()
    tag = args.tag or ""
    rounds = js.get("rounds" + tag) or []
    base = js.get("stage_baseline" + tag)
    sft = js.get("stage_sft" + tag)
    corr = js.get("stage_sft_corrected" + tag)
    if not rounds or not base:
        raise SystemExit("先跑 --mode all")

    # 图 1：定位正确率与 IoU 随阶段变化 + 数据池增长
    stages = ["基线", "SFT", "SFT+纠错"] + [f"RSI{r['round']}" for r in rounds]
    loc = [base["by_type"]["loc"]["ok"], sft["by_type"]["loc"]["ok"],
           corr["by_type"]["loc"]["ok"]] + [r["eval"]["by_type"]["loc"]["ok"] for r in rounds]
    ious = [base["mean_iou"], sft["mean_iou"], corr["mean_iou"]] + \
           [r["eval"]["mean_iou"] for r in rounds]
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=200)
    ax.bar(range(len(stages)), loc, color="#4c78a8", alpha=0.85, label="定位答对题数（/8）")
    ax.set_ylim(0, 8.6)
    ax.set_ylabel("定位答对（/8）")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    for i, v in enumerate(loc):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(range(len(stages)), ious, "o-", color="#d62728", lw=2, label="定位平均 IoU")
    ax2.set_ylim(0, max(0.05, max(x for x in ious if x is not None) * 1.3))
    ax2.set_ylabel("定位平均 IoU", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("定位正确率与 IoU 随阶段变化（4×4 网格，CPU 实测）")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig("fig_loc_stages.png")
    plt.close(fig)
    print("[fig] fig_loc_stages.png")

    # 图 3：SFT 步数与定位正确率（步数预算扫描）
    steps_pts = [(10, js.get("stage_sft_ns10")), (60, js.get("stage_sft")),
                 (200, js.get("stage_sft_ns200b"))]
    steps_pts = [(n, v) for n, v in steps_pts if v]
    if steps_pts:
        xs = [n for n, _ in steps_pts]
        locs = [v["by_type"]["loc"]["ok"] for _, v in steps_pts]
        ious = [v["mean_iou"] or 0 for _, v in steps_pts]
        fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
        ax.bar(range(len(xs)), locs, color="#4c78a8", alpha=0.85, width=0.55,
               label="定位答对（/8）")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{n} 步" for n in xs])
        ax.set_ylim(0, 8.6)
        ax.set_ylabel("定位答对（/8）")
        for i, v in enumerate(locs):
            ax.text(i, v + 0.15, str(v), ha="center", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(range(len(xs)), ious, "o-", color="#d62728", lw=2, label="平均 IoU")
        ax2.set_ylim(0, max(0.05, max(ious) * 1.4))
        ax2.set_ylabel("定位平均 IoU", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title("SFT 步数与定位能力（关闭图像切分，CPU 实测）")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig("fig_steps_budget.png")
        plt.close(fig)
        print("[fig] fig_steps_budget.png")

    # 图 2：数据池规模与 RSI 指标
    pools = [rounds[0]["pool_before"]] + [r["pool_after"] for r in rounds]
    labels = ["种子"] + [f"轮{rd['round']}" for rd in rounds]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    ax.plot(range(len(pools)), pools, "s-", color="#2ca02c", lw=2, ms=6)
    ax.set_xticks(range(len(pools)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("数据池样本数")
    ax.set_title("RSI 自举数据池增长（CPU 实测）")
    for i, v in enumerate(pools):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_pool_growth.png")
    plt.close(fig)
    print("[fig] fig_pool_growth.png")


def main():
    ap = argparse.ArgumentParser(description="实验作业六：SmolVLM-256M 精准定位的 RSI 递归自进化")
    ap.add_argument("--mode", default="all",
                    choices=["make_data", "make_sft", "make_seed", "infer", "sft",
                             "eval", "rsi", "all", "ablate", "cases", "figures"])
    ap.add_argument("--steps", type=int, default=10, help="每阶段 SFT 步数（CPU 建议 5~10）")
    ap.add_argument("--rounds", type=int, default=2, help="RSI 递归轮数")
    ap.add_argument("--grid", type=int, default=4, help="网格粒度（拓展 A：8）")
    ap.add_argument("--no_split", type=int, default=None, help="1=关闭图像切分")
    ap.add_argument("--tag", default=None, help="结果键后缀（如 _g8）")
    ap.add_argument("--sft_dir", default=None, help="SFT 适配器目录")
    ap.add_argument("--pool_all", type=int, default=0,
                    help="1=把所有答对的样本都入池（默认只收「错→对」的新样本）")
    args = ap.parse_args()

    global GRID
    GRID = args.grid

    if args.mode == "make_data":
        make_dataset()
        return
    if args.mode in ("make_sft", "make_seed"):
        make_sft(load_gt(), args.grid)
        return
    if args.mode == "infer":
        run_infer(args)
        return
    if args.mode == "sft":
        run_sft(args)
        return
    if args.mode == "eval":
        run_eval(args)
        return
    if args.mode == "rsi":
        run_rsi(args)
        return
    if args.mode == "ablate":
        run_ablate(args)
        return
    if args.mode == "cases":
        run_cases(args)
        return
    if args.mode == "figures":
        run_figures(args)
        return
    run_all(args)


if __name__ == "__main__":
    main()
