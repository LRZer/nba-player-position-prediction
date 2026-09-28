from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMPLATE = ROOT / "示例：ID3决策树法 .pptx"
SOURCE_MD = ROOT / "大数据技术实验.md"
OUT_PATH = RESULTS / "大数据技术实验_深度学习模型汇报.pptx"

WIDE = 13.333
HIGH = 7.5
FONT = "Microsoft YaHei"

NAVY = RGBColor(20, 45, 82)
BLUE = RGBColor(37, 99, 169)
CYAN = RGBColor(30, 136, 190)
TEAL = RGBColor(25, 148, 132)
ORANGE = RGBColor(230, 126, 34)
RED = RGBColor(196, 58, 72)
INK = RGBColor(31, 41, 55)
GREY = RGBColor(90, 102, 118)
MID = RGBColor(148, 163, 184)
LINE = RGBColor(218, 226, 237)
BG = RGBColor(246, 249, 252)
WHITE = RGBColor(255, 255, 255)
BLUE_L = RGBColor(231, 241, 252)
TEAL_L = RGBColor(229, 246, 242)
ORANGE_L = RGBColor(255, 241, 224)
RED_L = RGBColor(253, 232, 235)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def n4(value: float) -> str:
    return f"{value:.4f}"


def clear_slides(prs: Presentation) -> None:
    slide_id_list = prs.slides._sldIdLst
    for slide_id in list(slide_id_list):
        prs.part.drop_rel(slide_id.rId)
        slide_id_list.remove(slide_id)


def set_fill(shape, fill: RGBColor, line: RGBColor | None = None, width=0.75) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if hasattr(shape, "line"):
        shape.line.color.rgb = line if line is not None else fill
        shape.line.width = Pt(width)


def para_style(paragraph, size=14, bold=False, color=INK, align=None) -> None:
    paragraph.font.name = FONT
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    if align is not None:
        paragraph.alignment = align


def textbox(slide, text, x, y, w, h, size=14, bold=False, color=INK, align=None, valign=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.03)
    tf.margin_right = Inches(0.03)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    if valign:
        tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = text
    para_style(p, size=size, bold=bold, color=color, align=align)
    return box


def bullets(slide, items, x, y, w, h, size=13, color=INK, gap=0.94):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {item}"
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.line_spacing = gap
    return box


def new_slide(prs, section, title, subtitle, page, total):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    top = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(WIDE), Inches(0.22))
    set_fill(top, NAVY)
    side = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.22), Inches(0.13), Inches(HIGH - 0.22))
    set_fill(side, BLUE)
    textbox(slide, section, 0.55, 0.42, 2.6, 0.22, size=9.5, bold=True, color=TEAL)
    textbox(slide, title, 0.55, 0.68, 11.4, 0.42, size=22, bold=True, color=NAVY)
    if subtitle:
        textbox(slide, subtitle, 0.58, 1.13, 11.6, 0.28, size=10.8, color=GREY)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.43), Inches(1.15), Inches(0.045))
    set_fill(accent, ORANGE)
    textbox(slide, f"{page:02d}/{total:02d}", 11.70, 7.07, 0.9, 0.22, size=9, color=GREY, align=PP_ALIGN.RIGHT)
    textbox(slide, "大数据技术实验 · 深度学习模型", 0.55, 7.07, 4.2, 0.22, size=8.5, color=MID)
    return slide


def panel(slide, x, y, w, h, fill=WHITE, line=LINE, rounded=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    set_fill(shp, fill, line)
    return shp


def card(slide, x, y, w, h, title, body, accent=BLUE, body_size=18, note=None):
    panel(slide, x, y, w, h)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.06), Inches(h))
    set_fill(bar, accent)
    textbox(slide, title, x + 0.18, y + 0.12, w - 0.30, 0.24, size=10.4, bold=True, color=GREY)
    textbox(slide, body, x + 0.18, y + 0.42, w - 0.30, 0.36, size=body_size, bold=True, color=NAVY)
    if note:
        textbox(slide, note, x + 0.18, y + 0.83, w - 0.30, 0.24, size=8.7, color=GREY)


def callout(slide, title, items, x, y, w, h, accent=TEAL, fill=WHITE, size=11.5):
    panel(slide, x, y, w, h, fill=fill)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.055), Inches(h))
    set_fill(bar, accent)
    textbox(slide, title, x + 0.18, y + 0.13, w - 0.36, 0.26, size=12.4, bold=True, color=NAVY)
    bullets(slide, items, x + 0.18, y + 0.48, w - 0.36, h - 0.56, size=size)


def table(slide, rows, cols, x, y, w, h, widths=None, font_size=9.5, header=BLUE):
    tbl = slide.shapes.add_table(len(rows) + 1, len(cols), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for i, width in enumerate(widths):
            tbl.columns[i].width = Inches(width)
    for j, col in enumerate(cols):
        cell = tbl.cell(0, j)
        cell.text = col
        set_fill(cell, header)
        cell.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            para_style(p, size=font_size, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for i, row in enumerate(rows):
        row_fill = WHITE if i % 2 == 0 else RGBColor(241, 245, 249)
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = str(val)
            set_fill(cell, row_fill, LINE)
            cell.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                para_style(p, size=font_size, color=INK, align=PP_ALIGN.CENTER)
    return tbl


def image_box(slide, path: Path, x, y, w, h, title=None, caption=None):
    panel(slide, x, y, w, h)
    top_pad = 0.34 if title else 0.10
    bottom_pad = 0.24 if caption else 0.10
    if title:
        textbox(slide, title, x + 0.15, y + 0.11, w - 0.30, 0.22, size=10.4, bold=True, color=NAVY)
    if caption:
        textbox(slide, caption, x + 0.16, y + h - 0.28, w - 0.32, 0.18, size=7.8, color=GREY, align=PP_ALIGN.CENTER)
    if not path.exists():
        textbox(slide, f"缺少图片：{path.name}", x + 0.15, y + h / 2, w - 0.30, 0.25, size=11, color=RED, align=PP_ALIGN.CENTER)
        return None
    with Image.open(path) as img:
        iw, ih = img.size
    box_w = w - 0.24
    box_h = h - top_pad - bottom_pad
    ratio = iw / ih
    if ratio >= box_w / box_h:
        final_w = box_w
        final_h = final_w / ratio
    else:
        final_h = box_h
        final_w = final_h * ratio
    px = x + 0.12 + (box_w - final_w) / 2
    py = y + top_pad + (box_h - final_h) / 2
    return slide.shapes.add_picture(str(path), Inches(px), Inches(py), width=Inches(final_w), height=Inches(final_h))


def process_box(slide, text, x, y, w, h, fill=BLUE_L, color=NAVY, size=10.5):
    shp = panel(slide, x, y, w, h, fill=fill, line=fill)
    tf = shp.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    para_style(p, size=size, bold=True, color=color, align=PP_ALIGN.CENTER)
    return shp


def arrow(slide, x1, y1, x2, y2, color=GREY, width=1.2):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = color
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def quote_box(slide, text, x, y, w, h):
    panel(slide, x, y, w, h, fill=TEAL_L, line=TEAL_L)
    textbox(slide, text, x + 0.18, y + 0.15, w - 0.36, h - 0.22, size=12, bold=True, color=NAVY)


def main() -> None:
    md_text = SOURCE_MD.read_text(encoding="utf-8")
    bayes = read_json(RESULTS / "experiment1_bayes" / "metrics.json")
    kmeans = read_json(RESULTS / "experiment2_kmeans" / "metrics.json")
    id3 = read_json(RESULTS / "experiment3_id3" / "metrics.json")
    c45 = read_json(RESULTS / "experiment4_c45" / "metrics.json")
    dnn = read_json(RESULTS / "experiment5_dnn" / "metrics.json")
    class_metrics = pd.read_csv(RESULTS / "experiment5_dnn" / "class_metrics.csv")

    prs = Presentation(str(TEMPLATE)) if TEMPLATE.exists() else Presentation()
    clear_slides(prs)
    prs.slide_width = Inches(WIDE)
    prs.slide_height = Inches(HIGH)
    total = 34
    page = 1

    # 1
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(WIDE), Inches(0.30))
    set_fill(band, NAVY)
    side = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.30), Inches(0.18), Inches(HIGH - 0.30))
    set_fill(side, BLUE)
    textbox(slide, "大数据技术实验", 0.75, 0.82, 7.5, 0.55, size=30, bold=True, color=NAVY)
    textbox(slide, "深度学习模型流程、模块设计与实验结果汇报", 0.78, 1.42, 8.8, 0.35, size=16, color=GREY)
    textbox(slide, "依据：大数据技术实验.md；参考：ID3决策树法示例PPT", 0.80, 1.88, 7.2, 0.24, size=10.5, color=MID)
    card(slide, 0.85, 2.72, 2.05, 0.92, "输入特征", "45维", ORANGE, note="17维原始 + 28维工程")
    card(slide, 3.20, 2.72, 2.05, 0.92, "Accuracy", pct(dnn["accuracy"]), TEAL, body_size=17)
    card(slide, 5.55, 2.72, 2.05, 0.92, "Macro F1", pct(dnn["macro_f1"]), BLUE, body_size=17)
    card(slide, 7.90, 2.72, 2.35, 0.92, "最终模型", "DNN融合", RED, note="DNN + 年代专家 + ExtraTrees")
    image_box(slide, RESULTS / "supervised_model_comparison_cn.png", 0.85, 4.12, 5.50, 2.35, title="监督分类模型对比")
    image_box(slide, RESULTS / "experiment5_dnn" / "confusion_matrix.png", 6.70, 4.12, 4.80, 2.35, title="最终模型混淆矩阵")
    textbox(slide, f"{page:02d}/{total:02d}", 11.75, 7.07, 0.9, 0.22, size=9, color=GREY, align=PP_ALIGN.RIGHT)
    page += 1

    # 2
    slide = new_slide(prs, "汇报目录", "本次汇报内容", "按照Markdown总结内容完整展开，重点介绍深度学习模型各模块", page, total)
    table(slide, [
        ["1", "整体流程", "从45维特征到最终位置预测"],
        ["2", "输入特征", "76维候选、45维筛选、原始特征与工程特征"],
        ["3", "Residual DNN", "主干网络结构、参数和集成学习"],
        ["4", "监督对比学习", "ContrastiveResidualPositionNet和辅助损失"],
        ["5", "年代专家DNN", "按Year分段训练专家模型"],
        ["6", "ExtraTrees融合", "树模型辅助概率融合与最终输出"],
        ["7", "实验结果", "模型指标、混淆矩阵和消融实验"],
    ], ["序号", "模块", "说明"], 1.05, 1.65, 10.90, 4.58, widths=[0.75, 2.10, 8.05], font_size=11)
    quote_box(slide, "汇报主线：先讲数据如何进入模型，再讲模型如何学习，最后用结果和消融实验证明每个模块的作用。", 1.05, 6.45, 10.80, 0.48)
    page += 1

    # 3
    slide = new_slide(prs, "整体流程", "深度学习模型整体流程", "从输入技术特征到最终预测位置的完整链路", page, total)
    xs = [0.65, 2.28, 3.88, 5.70, 7.72, 9.70, 11.20]
    labels = ["45维技术\n特征", "标准化", "DNN学习\n位置规律", "年代专家\n补充", "ExtraTrees\n概率补充", "概率融合", "输出最终\n位置"]
    fills = [TEAL_L, BLUE_L, BLUE_L, TEAL_L, ORANGE_L, TEAL_L, BLUE_L]
    colors = [NAVY, NAVY, NAVY, NAVY, ORANGE, NAVY, NAVY]
    for i, (x, label) in enumerate(zip(xs, labels)):
        process_box(slide, label, x, 2.45, 1.18 if i != 6 else 1.35, 0.76, fill=fills[i], color=colors[i])
        if i < len(xs) - 1:
            arrow(slide, x + (1.18 if i != 6 else 1.35) + 0.04, 2.83, xs[i + 1] - 0.10, 2.83)
    callout(slide, "流程解释", [
        "45维特征把球员赛季技术表现转换成数值画像。",
        "标准化让不同量纲的特征处在相近范围，便于DNN训练。",
        "DNN学习主要位置边界，年代专家吸收不同时期打法差异。",
        "ExtraTrees提供另一种表格数据判断方式，最终进行概率融合。",
    ], 0.85, 4.35, 11.40, 1.55, accent=TEAL, size=12.0)
    page += 1

    # 4
    slide = new_slide(prs, "任务与合规", "实验任务和数据边界", "预测目标是五个NBA标准位置，同时避免使用身份泄露信息", page, total)
    card(slide, 0.90, 1.70, 2.25, 0.95, "分类目标", "5类", BLUE, note="C / PF / PG / SF / SG")
    card(slide, 3.45, 1.70, 2.25, 0.95, "训练集", "14,981", TEAL, note="最终训练样本")
    card(slide, 6.00, 1.70, 2.25, 0.95, "测试集", "3,746", BLUE, note="统一评估")
    card(slide, 8.55, 1.70, 2.25, 0.95, "验证集", "2,997", ORANGE, note="训练集内划分")
    callout(slide, "允许使用", [
        "原始技术统计：得分、篮板、助攻、命中率、抢断、盖帽等。",
        "技术统计派生指标：每场、每36分钟、比例指标、位置分离指标。",
        "Year用于刻画不同年代打法背景，不属于球员身份信息。",
    ], 1.00, 3.45, 5.15, 1.62, accent=TEAL, size=11.4)
    callout(slide, "不使用", [
        "Player：球员姓名。",
        "Tm：球队字段。",
        "历史位置、球员姓名相关统计、球队相关统计等可能泄露身份的信息。",
    ], 6.70, 3.45, 5.15, 1.62, accent=RED, size=11.4)
    quote_box(slide, "目的：让模型根据技术表现判断位置，而不是记住具体球员或球队。", 1.00, 5.85, 10.85, 0.48)
    page += 1

    # 5
    slide = new_slide(prs, "特征构造", "候选特征的来源", "最初构造76维候选特征：26维原始数值特征 + 50维工程特征", page, total)
    process_box(slide, "26维原始\n数值特征", 1.00, 2.15, 2.10, 0.82, fill=BLUE_L)
    textbox(slide, "+", 3.35, 2.33, 0.3, 0.26, size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    process_box(slide, "50维工程\n派生特征", 3.95, 2.15, 2.10, 0.82, fill=TEAL_L)
    textbox(slide, "=", 6.30, 2.33, 0.3, 0.26, size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    process_box(slide, "76维候选\n特征集合", 6.90, 2.15, 2.10, 0.82, fill=ORANGE_L, color=ORANGE)
    arrow(slide, 9.15, 2.56, 9.78, 2.56)
    process_box(slide, "筛选后\n45维正式输入", 9.90, 2.15, 2.25, 0.82, fill=TEAL_L)
    callout(slide, "构造思路", [
        "原始特征保留基础技术统计和效率信息。",
        "工程特征将总量转换为场均、每36分钟、比例和位置画像。",
        "候选特征较多时会出现重复和冗余，因此后续进行筛选。",
    ], 1.00, 4.10, 10.95, 1.20, accent=BLUE, size=12.0)
    page += 1

    # 6
    slide = new_slide(prs, "特征筛选", "候选特征子集设计", "为减少冗余、保持效果，设计多组不同维度的特征子集重新训练", page, total)
    table(slide, [
        ["76维", "全量候选特征", "作为完整特征基准"],
        ["54维", "增强特征", "加回部分复合画像指标"],
        ["52维", "使用率特征", "加入使用率和负荷相关指标"],
        ["50维", "指数特征", "加入多种综合指数"],
        ["45维", "位置分离核心特征", "最终正式版本"],
        ["42维", "领域紧凑特征", "进一步压缩核心指标"],
        ["35/30维", "重要性Top特征", "按ExtraTrees重要性取Top-k"],
        ["32维", "画像核心特征", "保留更少领域画像"],
    ], ["维度", "特征子集", "设计目的"], 0.90, 1.55, 11.20, 4.65, widths=[0.95, 2.60, 7.65], font_size=10.2)
    quote_box(slide, "所有候选子集都放入完整实验五流程重新训练，而不是只做简单相关性排序。", 0.95, 6.45, 11.00, 0.48)
    page += 1

    # 7
    slide = new_slide(prs, "特征筛选", "特征筛选结果", "最终选择指标稳定、维度更少、解释更清楚的45维版本", page, total)
    image_box(slide, RESULTS / "experiment5_feature_selection_search" / "compact_feature_search_macro_f1.png", 0.80, 1.45, 5.30, 3.40, title="不同特征集Macro F1")
    table(slide, [
        ["76维全量", "0.7253", "0.7257"],
        ["45维正式版", "0.7261", "0.7258"],
        ["35维重要性版", "0.7157", "0.7158"],
        ["32维画像版", "0.7058", "0.7057"],
    ], ["特征集", "Accuracy", "Macro F1"], 6.65, 1.80, 4.20, 1.90, widths=[1.60, 1.25, 1.35], font_size=10.6, header=TEAL)
    callout(slide, "选择45维的原因", [
        "比76维减少31个特征，降幅约40.8%。",
        "Accuracy保持0.7261，Macro F1保持0.7258。",
        "保留位置区分能力强、解释性清楚的核心指标。",
    ], 6.65, 4.18, 4.75, 1.28, accent=ORANGE, size=11.2)
    page += 1

    # 8
    slide = new_slide(prs, "特征筛选", "筛选原则总结", "去掉重复、冗余、贡献弱的特征，保留更有篮球含义的输入", page, total)
    callout(slide, "去掉什么", [
        "高度重复的信息，例如总量之间的强相关字段。",
        "在多个子集实验中贡献较弱的派生指标。",
        "维度增加但验证效果下降的复合特征。",
    ], 0.95, 1.65, 5.10, 1.80, accent=RED, size=11.8)
    callout(slide, "保留什么", [
        "效率指标：命中率、有效命中率、罚球命中率。",
        "单位时间指标：每场、每36分钟统计。",
        "比例指标：三分出手占比、罚球率、篮板比例。",
        "相邻位置分离指标：C/PF、SF/PF、SG/PG。",
    ], 6.55, 1.65, 5.10, 1.80, accent=TEAL, size=11.8)
    image_box(slide, RESULTS / "experiment5_feature_selection_search" / "feature_count_vs_macro_f1.png", 2.15, 4.05, 7.80, 2.20, title="特征数量与Macro F1关系")
    page += 1

    # 9
    slide = new_slide(prs, "输入特征", "17维原始数值特征", "从26维原始数值特征中保留17维，作为基础技术结构", page, total)
    table(slide, [
        ["赛季与出场", "Year, Age, G, MP", "体现时代、年龄、出场规模"],
        ["投篮效率", "FG%, 3P%, 2P%, eFG%, FT%", "体现不同投篮方式效率"],
        ["篮板", "ORB, DRB, TRB", "区分内线和外线的重要基础信号"],
        ["组织与防守", "AST, STL, BLK, TOV, PF", "体现组织、防守方式和对抗特征"],
    ], ["类别", "特征", "解释"], 0.95, 1.58, 10.95, 3.25, widths=[1.55, 4.35, 5.05], font_size=10.5, header=BLUE)
    callout(slide, "为什么只保留17维", [
        "保留直接体现位置差异的基础统计。",
        "保留效率类指标，减少纯总量字段的重复影响。",
        "被删除的总量信息多数会通过工程特征重新表达。",
    ], 0.95, 5.15, 10.95, 0.98, accent=TEAL, size=11.5)
    page += 1

    # 10
    slide = new_slide(prs, "输入特征", "为什么不直接使用全部26维原始特征", "部分原始总量字段高度相关，直接全部输入会增加冗余", page, total)
    table(slide, [
        ["得分与出手", "FG, FGA, PTS", "与出场时间、投篮次数强相关"],
        ["三分总量", "3P, 3PA", "通过3P%、ThreePAr、3PA_36重新表达"],
        ["两分总量", "2P, 2PA", "通过2P%、得分效率类指标表达"],
        ["罚球总量", "FT, FTA", "通过FT%、FTr、FTA_36表达"],
    ], ["未直接保留字段", "例子", "处理方式"], 0.95, 1.55, 10.85, 3.25, widths=[2.10, 2.35, 6.40], font_size=10.4, header=ORANGE)
    quote_box(slide, "筛选不是丢掉信息，而是把总量信息转化为更适合模型学习的场均、比例和单位时间指标。", 1.05, 5.20, 10.65, 0.52)
    page += 1

    # 11
    slide = new_slide(prs, "工程特征", "28维工程特征：场均指标", "场均指标把总量转换为每场贡献，更容易比较不同出场规模的球员", page, total)
    table(slide, [
        ["PPG", "PTS / G", "场均得分"],
        ["MPG", "MP / G", "场均时间"],
        ["RPG", "TRB / G", "场均篮板"],
        ["APG", "AST / G", "场均助攻"],
        ["SPG / BPG", "STL / G, BLK / G", "场均抢断、盖帽"],
        ["TPG / FPG", "TOV / G, PF / G", "场均失误、犯规"],
    ], ["特征", "计算方式", "含义"], 0.95, 1.55, 6.00, 3.80, widths=[1.30, 2.20, 2.50], font_size=10.3, header=TEAL)
    callout(slide, "为什么需要场均指标", [
        "总得分、总篮板会受到出场场次影响。",
        "场均指标更接近球员每场比赛的真实角色。",
        "有利于比较出场场次不同的球员。",
    ], 7.35, 1.80, 4.00, 1.80, accent=TEAL, size=11.5)
    callout(slide, "示例解释", [
        "两个球员总得分接近时，场均得分可能不同。",
        "场均时间也能帮助区分主力、轮换和边缘样本。",
    ], 7.35, 4.05, 4.00, 1.15, accent=BLUE, size=11.5)
    page += 1

    # 12
    slide = new_slide(prs, "工程特征", "投篮结构与比例指标", "比例指标用于描述球员技术风格，而不是只看总出手数量", page, total)
    process_box(slide, "ThreePAr\n= 3PA / FGA", 1.20, 2.10, 2.10, 0.85, fill=BLUE_L)
    process_box(slide, "FTr\n= FTA / FGA", 4.00, 2.10, 2.10, 0.85, fill=TEAL_L)
    process_box(slide, "AST_TOV\n= AST / TOV", 6.80, 2.10, 2.10, 0.85, fill=ORANGE_L, color=ORANGE)
    process_box(slide, "ORB_Ratio / DRB_Ratio\n= 篮板结构", 9.60, 2.10, 2.10, 0.85, fill=BLUE_L)
    callout(slide, "指标含义", [
        "ThreePAr表示三分出手占总出手比例，用来衡量外线倾向。",
        "FTr表示罚球出手相对投篮出手的比例，用来衡量造犯规和内线冲击。",
        "AST_TOV衡量组织效率，篮板比例区分进攻篮板和防守篮板特征。",
    ], 1.10, 4.10, 10.70, 1.20, accent=TEAL, size=12.0)
    page += 1

    # 13
    slide = new_slide(prs, "工程特征", "每36分钟指标", "每36分钟指标用于降低出场时间差异对统计总量的影响", page, total)
    table(slide, [
        ["PTS_36", "得分换算到每36分钟", "衡量单位时间得分能力"],
        ["TRB_36", "篮板换算到每36分钟", "衡量单位时间篮板能力"],
        ["AST_36", "助攻换算到每36分钟", "衡量单位时间组织能力"],
        ["STL_36 / BLK_36", "防守数据换算", "衡量抢断、护框倾向"],
        ["3PA_36 / FTA_36", "出手结构换算", "衡量三分和罚球倾向"],
    ], ["特征", "计算思想", "作用"], 0.95, 1.55, 6.60, 3.35, widths=[1.65, 2.65, 2.30], font_size=10.3, header=BLUE)
    callout(slide, "为什么重要", [
        "替补球员总数据可能低，但单位时间效率可能很高。",
        "每36分钟指标让不同上场时间的球员更可比。",
        "它能帮助模型识别球员真实技术风格，而不是只看总量。",
    ], 8.00, 1.90, 3.60, 1.80, accent=ORANGE, size=11.4)
    quote_box(slide, "汇报表述：每36分钟指标相当于把所有球员放到相近时间尺度下比较。", 1.10, 5.45, 10.50, 0.48)
    page += 1

    # 14
    slide = new_slide(prs, "工程特征", "位置分离特征", "位置分离指标用于加强相邻位置的区分能力", page, total)
    table(slide, [
        ["Heightless_Big_Profile", "大个内线画像", "不使用身高，基于篮板、盖帽、内线特征近似描述内线倾向"],
        ["Primary_Guard_Profile", "主控后卫画像", "突出助攻、抢断、外线和低内线指标"],
        ["Center_PF_Separation", "C/PF区分", "加强中锋和大前锋之间的差异"],
        ["SG_PG_Separation", "SG/PG区分", "加强得分后卫和控球后卫之间的差异"],
        ["SF_PF_Separation", "SF/PF区分", "加强小前锋和大前锋之间的差异"],
    ], ["特征", "用途", "解释"], 0.85, 1.48, 11.40, 3.65, widths=[2.50, 1.65, 7.25], font_size=9.6, header=TEAL)
    callout(slide, "相邻位置说明", [
        "C/PF：中锋和大前锋，内线统计容易重叠。",
        "PF/SF：大前锋和小前锋，锋线摇摆特征明显。",
        "PG/SG：控球后卫和得分后卫，双能卫样本容易混淆。",
    ], 0.95, 5.42, 11.10, 0.82, accent=ORANGE, size=11.2)
    page += 1

    # 15
    slide = new_slide(prs, "标准化", "数据标准化处理", "DNN对数值尺度敏感，因此训练前对45维输入进行StandardScaler标准化", page, total)
    textbox(slide, "标准化公式", 1.05, 1.78, 2.00, 0.30, size=15, bold=True, color=NAVY)
    process_box(slide, "x_scaled = (x - mean_train) / std_train", 1.05, 2.30, 5.00, 0.72, fill=BLUE_L, size=14)
    callout(slide, "为什么要标准化", [
        "MP可能是几百到几千，FG%通常在0到1之间。",
        "如果尺度差异太大，DNN训练时数值大的特征可能主导梯度。",
        "标准化后每个特征大致均值为0、标准差为1，训练更稳定。",
    ], 6.70, 1.65, 4.95, 1.85, accent=TEAL, size=11.4)
    callout(slide, "避免数据泄露", [
        "只在训练集上计算均值和标准差。",
        "验证集和测试集只使用训练集得到的参数进行transform。",
        "测试集不参与标准化参数计算。",
    ], 1.05, 4.25, 10.60, 1.18, accent=ORANGE, size=11.7)
    quote_box(slide, "ExtraTrees不需要标准化，因为树模型按阈值分裂，对特征尺度不敏感。", 1.05, 6.00, 10.60, 0.48)
    page += 1

    # 16
    slide = new_slide(prs, "深度学习模型", "深度学习模块总览", "实验五的模型由DNN主干、对比学习成员、年代专家和ExtraTrees融合组成", page, total)
    process_box(slide, "45维输入", 0.85, 2.70, 1.35, 0.70, fill=TEAL_L)
    arrow(slide, 2.30, 3.05, 2.80, 3.05)
    process_box(slide, "Residual DNN\n主干", 2.90, 2.70, 1.55, 0.70, fill=BLUE_L)
    arrow(slide, 4.55, 3.05, 5.05, 2.45)
    arrow(slide, 4.55, 3.05, 5.05, 3.65)
    process_box(slide, "普通Residual\nDNN成员", 5.15, 2.10, 1.75, 0.70, fill=BLUE_L)
    process_box(slide, "Contrastive\nDNN成员", 5.15, 3.30, 1.75, 0.70, fill=TEAL_L)
    arrow(slide, 7.00, 2.45, 7.55, 3.05)
    arrow(slide, 7.00, 3.65, 7.55, 3.05)
    process_box(slide, "全局DNN\n概率", 7.65, 2.70, 1.55, 0.70, fill=ORANGE_L, color=ORANGE)
    arrow(slide, 9.30, 3.05, 9.82, 2.50)
    arrow(slide, 9.30, 3.05, 9.82, 3.60)
    process_box(slide, "年代专家", 9.92, 2.15, 1.45, 0.62, fill=TEAL_L)
    process_box(slide, "ExtraTrees", 9.92, 3.35, 1.45, 0.62, fill=ORANGE_L, color=ORANGE)
    arrow(slide, 11.45, 2.46, 12.00, 3.05)
    arrow(slide, 11.45, 3.66, 12.00, 3.05)
    process_box(slide, "最终概率", 12.08, 2.75, 0.90, 0.60, fill=BLUE_L, size=9)
    callout(slide, "设计思路", [
        "Residual DNN负责学习主要非线性位置边界。",
        "监督对比学习模型作为额外DNN成员参与集成。",
        "年代专家捕捉不同年代打法差异。",
        "ExtraTrees提供与DNN不同的表格模型概率补充。",
    ], 1.00, 5.05, 11.00, 1.12, accent=TEAL, size=11.5)
    page += 1

    # 17
    slide = new_slide(prs, "Residual DNN", "主模型：ResidualPositionNet", "主干模型采用残差全连接神经网络，将45维特征转为五类位置概率", page, total)
    process_box(slide, "输入\n45维", 0.95, 2.15, 1.05, 0.62, fill=BLUE_L)
    arrow(slide, 2.10, 2.46, 2.55, 2.46)
    process_box(slide, "输入层\n45→192", 2.65, 2.06, 1.35, 0.80, fill=TEAL_L)
    arrow(slide, 4.10, 2.46, 4.55, 2.46)
    process_box(slide, "Residual\nBlock ×2", 4.65, 2.06, 1.45, 0.80, fill=BLUE_L)
    arrow(slide, 6.20, 2.46, 6.65, 2.46)
    process_box(slide, "分类层\n192→5", 6.75, 2.06, 1.35, 0.80, fill=TEAL_L)
    arrow(slide, 8.20, 2.46, 8.65, 2.46)
    process_box(slide, "Softmax\n五类概率", 8.75, 2.06, 1.40, 0.80, fill=ORANGE_L, color=ORANGE)
    callout(slide, "结构参数", [
        "输入维度：45。",
        "隐藏层宽度：192。",
        "残差块数量：2。",
        "输出维度：5，对应C / PF / PG / SF / SG。",
        "Dropout：0.15。",
    ], 1.05, 4.05, 5.10, 1.55, accent=BLUE, size=11.4)
    callout(slide, "模型作用", [
        "学习得分、篮板、助攻、投篮结构之间的复杂组合。",
        "输出五个位置概率，为后续集成和融合提供基础。",
    ], 6.65, 4.05, 5.10, 1.55, accent=TEAL, size=11.4)
    page += 1

    # 18
    slide = new_slide(prs, "Residual DNN", "残差块内部结构", "Residual Block保留原始表示，同时学习需要补充的修正量", page, total)
    process_box(slide, "输入 x", 0.95, 2.30, 0.95, 0.55, fill=BLUE_L)
    arrow(slide, 2.00, 2.58, 2.48, 2.58)
    process_box(slide, "Linear\n192→192", 2.58, 2.15, 1.25, 0.82, fill=TEAL_L)
    arrow(slide, 3.93, 2.58, 4.30, 2.58)
    process_box(slide, "BatchNorm\nReLU/GELU", 4.40, 2.15, 1.35, 0.82, fill=BLUE_L)
    arrow(slide, 5.85, 2.58, 6.22, 2.58)
    process_box(slide, "Dropout\n0.15", 6.32, 2.15, 1.05, 0.82, fill=ORANGE_L, color=ORANGE)
    arrow(slide, 7.47, 2.58, 7.84, 2.58)
    process_box(slide, "Linear +\nBatchNorm", 7.94, 2.15, 1.25, 0.82, fill=TEAL_L)
    arrow(slide, 9.29, 2.58, 9.66, 2.58)
    process_box(slide, "x + F(x)", 9.76, 2.15, 1.20, 0.82, fill=BLUE_L)
    callout(slide, "残差连接含义", [
        "普通层学习F(x)，残差层输出x + F(x)。",
        "模型不是完全重写特征，而是在原表示上做修正。",
        "这样可以保留已有信息，使训练更稳定。",
    ], 1.05, 4.25, 5.20, 1.35, accent=TEAL, size=11.6)
    callout(slide, "各组件作用", [
        "Linear学习特征组合，BatchNorm稳定训练。",
        "非线性激活提升表达能力。",
        "Dropout随机丢弃部分神经元，降低过拟合。",
    ], 6.65, 4.25, 5.20, 1.35, accent=BLUE, size=11.6)
    page += 1

    # 19
    slide = new_slide(prs, "Residual DNN", "训练参数设计", "Residual DNN成员采用统一结构和训练策略，保证集成成员可比较", page, total)
    table(slide, [
        ["width", "192", "隐藏表示宽度"],
        ["blocks", "2", "残差块数量"],
        ["dropout", "0.15", "抑制过拟合"],
        ["learning_rate", "8e-4", "控制参数更新步长"],
        ["weight_decay", "1e-4", "正则化权重"],
        ["batch_size", "512", "批训练样本数"],
        ["optimizer", "AdamW", "优化神经网络参数"],
        ["scheduler", "CosineAnnealingLR", "学习率随训练逐渐调整"],
    ], ["参数", "取值", "说明"], 1.05, 1.48, 10.80, 4.55, widths=[2.25, 2.20, 6.35], font_size=10.2, header=BLUE)
    page += 1

    # 20
    slide = new_slide(prs, "集成学习", "普通Residual DNN集成", "训练3个不同随机种子的Residual DNN，对概率进行加权平均", page, total)
    for i, (seed, x) in enumerate([(42, 1.30), (7, 3.45), (2026, 5.60)]):
        process_box(slide, f"Residual DNN\nseed={seed}", x, 2.00, 1.45, 0.84, fill=BLUE_L)
        arrow(slide, x + 0.72, 2.92, 5.80, 4.00, color=MID, width=0.9)
    process_box(slide, "概率加权平均\n得到普通DNN集成概率", 4.70, 4.08, 2.35, 0.82, fill=ORANGE_L, color=ORANGE)
    callout(slide, "为什么要集成", [
        "单个神经网络会受随机初始化和训练批次顺序影响。",
        "多个模型输出概率再平均，可以降低偶然误差。",
        "最终使用概率平均而不是硬投票，能保留类别不确定性。",
    ], 1.00, 5.35, 10.95, 0.95, accent=TEAL, size=11.5)
    page += 1

    # 21
    slide = new_slide(prs, "监督对比学习", "ContrastiveResidualPositionNet", "在Residual DNN基础上增加投影头，形成带监督对比学习训练目标的新模型成员", page, total)
    process_box(slide, "45维输入", 0.95, 2.40, 1.20, 0.60, fill=BLUE_L)
    arrow(slide, 2.25, 2.70, 2.70, 2.70)
    process_box(slide, "Residual DNN\n隐藏表示", 2.80, 2.25, 1.60, 0.88, fill=TEAL_L)
    arrow(slide, 4.52, 2.58, 5.10, 2.05)
    arrow(slide, 4.52, 2.82, 5.10, 3.55)
    process_box(slide, "分类头\n输出5类概率", 5.20, 1.72, 1.55, 0.72, fill=BLUE_L)
    process_box(slide, "投影头\n192→128→64", 5.20, 3.30, 1.55, 0.72, fill=ORANGE_L, color=ORANGE)
    callout(slide, "与普通Residual DNN的区别", [
        "普通模型只通过分类头输出位置概率。",
        "Contrastive模型额外输出一个64维对比学习向量。",
        "这个向量只在训练时用于监督对比损失，预测时仍使用分类概率。",
    ], 7.35, 1.70, 4.45, 2.05, accent=TEAL, size=11.2)
    quote_box(slide, "它不是对3个普通DNN再加训练目标，而是另外训练3个新的Contrastive Residual DNN成员。", 1.05, 5.35, 10.65, 0.52)
    page += 1

    # 22
    slide = new_slide(prs, "监督对比学习", "监督对比学习目标", "在普通分类损失之外，要求同类样本更接近、不同类样本更分开", page, total)
    process_box(slide, "同位置样本\n拉近", 1.00, 1.85, 1.50, 0.62, fill=TEAL_L)
    process_box(slide, "不同位置样本\n拉远", 1.00, 3.10, 1.50, 0.62, fill=ORANGE_L, color=ORANGE)
    panel(slide, 3.10, 1.55, 3.10, 2.60)
    textbox(slide, "隐藏空间示意", 3.35, 1.82, 1.50, 0.24, size=11.8, bold=True, color=NAVY)
    for x, y, color in [(3.70, 2.45, TEAL), (4.05, 2.22, TEAL), (4.28, 2.52, TEAL), (5.10, 3.05, BLUE), (5.42, 3.28, BLUE), (4.90, 3.38, BLUE), (4.65, 2.75, ORANGE), (4.82, 2.45, ORANGE)]:
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.16), Inches(0.16))
        set_fill(dot, color)
    callout(slide, "训练损失", [
        "总损失 = 交叉熵损失 + 0.1 × 监督对比损失。",
        "交叉熵损失负责分类正确。",
        "监督对比损失负责让内部表示更容易区分类别。",
    ], 6.85, 1.65, 4.70, 1.55, accent=ORANGE, size=11.5)
    callout(slide, "适用原因", [
        "C/PF、PF/SF、PG/SG这类相邻位置统计上接近。",
        "监督对比学习让同位置样本更靠近，不同位置样本尽量分开。",
    ], 6.85, 3.80, 4.70, 1.28, accent=TEAL, size=11.5)
    page += 1

    # 23
    slide = new_slide(prs, "监督对比学习", "容易混淆的相邻位置", "C/PF、PF/SF、PG/SG表示两个相邻位置之间的边界", page, total)
    table(slide, [
        ["C/PF", "中锋 / 大前锋", "都可能篮板、盖帽和内线得分较高"],
        ["PF/SF", "大前锋 / 小前锋", "锋线摇摆，篮板、投篮、防守角色有重叠"],
        ["PG/SG", "控球后卫 / 得分后卫", "得分型控卫和组织型分卫统计接近"],
        ["SF/SG", "小前锋 / 得分后卫", "外线得分、三分和侧翼防守特征接近"],
    ], ["组合", "含义", "为什么容易混淆"], 1.05, 1.65, 10.80, 3.20, widths=[1.20, 2.35, 7.25], font_size=10.8, header=TEAL)
    quote_box(slide, "监督对比学习和位置分离特征都服务于同一目标：缓解相邻位置之间的混淆。", 1.05, 5.45, 10.80, 0.52)
    page += 1

    # 24
    slide = new_slide(prs, "集成学习", "全局DNN集成结构", "全局DNN集成由两组模型组成：3个普通Residual DNN和3个Contrastive Residual DNN", page, total)
    for i, x in enumerate([0.90, 2.35, 3.80]):
        process_box(slide, f"Residual\nDNN {i + 1}", x, 2.00, 1.05, 0.76, fill=BLUE_L)
        arrow(slide, x + 0.52, 2.82, 5.80, 4.00, color=MID, width=0.8)
    for i, x in enumerate([6.65, 8.10, 9.55]):
        process_box(slide, f"Contrastive\nDNN {i + 1}", x, 2.00, 1.05, 0.76, fill=TEAL_L)
        arrow(slide, x + 0.52, 2.82, 5.80, 4.00, color=MID, width=0.8)
    process_box(slide, "6个模型概率\n加权平均", 4.70, 4.08, 2.20, 0.80, fill=ORANGE_L, color=ORANGE)
    callout(slide, "输出结果", [
        "每个模型都输出P(C)、P(PF)、P(PG)、P(SF)、P(SG)。",
        "最终不是投票，而是对五类概率加权平均。",
        "得到global_probabilities，即全局DNN概率。",
    ], 1.00, 5.35, 10.90, 0.95, accent=TEAL, size=11.5)
    page += 1

    # 25
    slide = new_slide(prs, "年代专家", "为什么加入年代专家DNN", "NBA不同年代位置打法存在差异，单一全局模型容易学到平均规律", page, total)
    callout(slide, "早期内线", [
        "更偏篮下进攻、篮板、盖帽。",
        "三分出手通常较少。",
        "位置分工相对固定。",
    ], 0.95, 1.70, 3.45, 1.55, accent=BLUE, size=11.5)
    callout(slide, "现代内线", [
        "仍有篮板和护框。",
        "可能增加三分、传球和空间属性。",
        "位置边界更加连续。",
    ], 4.95, 1.70, 3.45, 1.55, accent=TEAL, size=11.5)
    callout(slide, "专家模型作用", [
        "全局DNN学习总体规律。",
        "年代专家学习特定年代的局部规律。",
        "二者融合后兼顾稳定性和时代差异。",
    ], 8.95, 1.70, 3.15, 1.55, accent=ORANGE, size=11.2)
    quote_box(slide, "Year只描述赛季年份和时代背景，不是球员身份信息，因此可用于年代分桶。", 1.05, 4.85, 10.85, 0.52)
    page += 1

    # 26
    slide = new_slide(prs, "年代专家", "年代分桶与专家训练", "根据Year划分4个年代段，每个年代单独训练专家DNN集成", page, total)
    table(slide, [
        ["era 0", "Year < 1990", "1980s及以前"],
        ["era 1", "1990 ≤ Year < 2000", "1990s"],
        ["era 2", "2000 ≤ Year < 2010", "2000s"],
        ["era 3", "Year ≥ 2010", "2010s及以后"],
    ], ["年代桶", "划分条件", "说明"], 0.95, 1.55, 5.60, 2.55, widths=[1.15, 2.45, 2.00], font_size=10.8, header=BLUE)
    callout(slide, "每个年代专家内部", [
        "每个年代训练4个DNN成员。",
        "包括2个普通Residual DNN和2个Contrastive Residual DNN。",
        "4个成员权重均为0.25，输出平均得到该年代专家概率。",
    ], 7.05, 1.70, 4.55, 1.85, accent=TEAL, size=11.3)
    process_box(slide, "样本Year\n匹配年代", 1.25, 5.00, 1.60, 0.65, fill=BLUE_L)
    arrow(slide, 2.95, 5.33, 3.55, 5.33)
    process_box(slide, "对应年代\n专家DNN", 3.65, 5.00, 1.70, 0.65, fill=TEAL_L)
    arrow(slide, 5.45, 5.33, 6.05, 5.33)
    process_box(slide, "输出\nera_probabilities", 6.15, 5.00, 2.10, 0.65, fill=ORANGE_L, color=ORANGE)
    page += 1

    # 27
    slide = new_slide(prs, "年代专家", "年代专家与全局DNN融合", "最终DNN概率由全局DNN概率和年代专家概率加权得到", page, total)
    process_box(slide, "global_probabilities\n全局DNN概率", 1.05, 2.15, 2.40, 0.78, fill=BLUE_L)
    process_box(slide, "era_probabilities\n年代专家概率", 1.05, 3.35, 2.40, 0.78, fill=TEAL_L)
    arrow(slide, 3.60, 2.54, 4.35, 3.03)
    arrow(slide, 3.60, 3.74, 4.35, 3.35)
    process_box(slide, "DNN最终概率\n= 0.60×global + 0.40×era", 4.48, 2.88, 3.10, 0.92, fill=ORANGE_L, color=ORANGE, size=11)
    callout(slide, "权重设置", [
        "ERA_EXPERT_WEIGHT = 0.40。",
        "全局DNN占60%，年代专家占40%。",
        "全局模型更稳定，专家模型更贴合特定年代。",
    ], 8.10, 1.80, 3.75, 1.52, accent=TEAL, size=11.3)
    callout(slide, "效果贡献", [
        "global_only_macro_f1 = 0.7001。",
        "era_only_macro_f1 = 0.7122。",
        "融合后DNN ensemble macro_f1 = 0.7167。",
        "消融实验：去掉年代专家后Macro F1下降约0.0176。",
    ], 8.10, 3.80, 3.75, 1.72, accent=BLUE, size=10.8)
    page += 1

    # 28
    slide = new_slide(prs, "ExtraTrees", "ExtraTrees辅助概率模型", "ExtraTrees作为DNN之外的树模型概率补充，适合表格统计特征", page, total)
    callout(slide, "ExtraTrees是什么", [
        "全称Extremely Randomized Trees，极端随机树。",
        "由很多棵随机决策树组成，类似随机森林但随机性更强。",
        "可以直接处理连续表格特征。",
    ], 0.95, 1.70, 5.05, 1.65, accent=ORANGE, size=11.4)
    callout(slide, "为什么它效果不错", [
        "它是多棵树集成，比ID3/C4.5单棵树更稳定。",
        "它直接使用45维增强特征，而不是前四个实验的26维基础特征。",
        "它不需要对连续值做粗离散化，保留更多数值细节。",
    ], 6.50, 1.70, 5.05, 1.65, accent=TEAL, size=11.4)
    card(slide, 2.15, 4.35, 2.25, 0.92, "ExtraTrees Acc", "0.6909", BLUE)
    card(slide, 4.85, 4.35, 2.25, 0.92, "ExtraTrees F1", "0.6899", TEAL)
    card(slide, 7.55, 4.35, 2.25, 0.92, "最终F1", "0.7258", ORANGE)
    page += 1

    # 29
    slide = new_slide(prs, "ExtraTrees", "DNN与ExtraTrees概率融合", "最终不是硬投票，而是把两组五类概率进行加权融合", page, total)
    process_box(slide, "DNN概率\nP_dnn", 1.05, 2.15, 1.55, 0.70, fill=BLUE_L)
    process_box(slide, "ExtraTrees概率\nP_tree", 1.05, 3.35, 1.55, 0.70, fill=ORANGE_L, color=ORANGE)
    arrow(slide, 2.72, 2.50, 4.05, 3.05)
    arrow(slide, 2.72, 3.70, 4.05, 3.40)
    process_box(slide, "最终概率\n= 0.55×DNN + 0.45×ExtraTrees", 4.18, 2.90, 3.25, 0.92, fill=TEAL_L, size=11)
    arrow(slide, 7.56, 3.36, 8.22, 3.36)
    process_box(slide, "取概率最大类别\n作为最终位置", 8.32, 2.98, 2.15, 0.76, fill=BLUE_L)
    callout(slide, "权重来源", [
        "0.45是ExtraTrees概率权重，代码中为AUXILIARY_TREE_WEIGHT。",
        "DNN权重为1 - 0.45 = 0.55。",
        "权重通过验证集搜索确定，测试集不参与权重选择。",
    ], 1.05, 5.10, 10.65, 1.02, accent=TEAL, size=11.5)
    page += 1

    # 30
    slide = new_slide(prs, "最终输出", "最终预测如何产生", "模型输出五个位置概率，选择概率最大的类别作为预测位置", page, total)
    table(slide, [
        ["C", "0.06"],
        ["PF", "0.13"],
        ["PG", "0.58"],
        ["SF", "0.09"],
        ["SG", "0.14"],
    ], ["位置", "最终概率示例"], 1.15, 1.75, 3.10, 2.50, widths=[1.30, 1.80], font_size=12.0, header=BLUE)
    process_box(slide, "概率最大：PG", 5.00, 2.50, 2.00, 0.78, fill=TEAL_L)
    arrow(slide, 7.12, 2.88, 7.82, 2.88)
    process_box(slide, "预测位置\nPG", 7.95, 2.50, 1.65, 0.78, fill=ORANGE_L, color=ORANGE)
    callout(slide, "为什么用概率", [
        "概率能表达样本本身的不确定性。",
        "对PF/SF、SG/SF这类边界模糊样本，概率比硬规则更合适。",
        "后续分析也可以查看不同模型概率是否一致。",
    ], 1.05, 4.55, 10.55, 1.05, accent=TEAL, size=11.7)
    page += 1

    # 31
    slide = new_slide(prs, "实验结果", "最终模型整体指标", "深度学习融合模型在五个实验中表现最好", page, total)
    image_box(slide, RESULTS / "supervised_model_comparison_cn.png", 0.85, 1.45, 6.25, 3.55, title="监督分类模型指标对比")
    table(slide, [
        ["贝叶斯", n4(bayes["accuracy"]), n4(bayes["macro_f1"])],
        ["ID3", n4(id3["accuracy"]), n4(id3["macro_f1"])],
        ["C4.5", n4(c45["accuracy"]), n4(c45["macro_f1"])],
        ["深度学习", n4(dnn["accuracy"]), n4(dnn["macro_f1"])],
    ], ["模型", "Accuracy", "Macro F1"], 7.62, 1.65, 3.95, 2.05, widths=[1.35, 1.20, 1.40], font_size=10.4, header=TEAL)
    callout(slide, "结果结论", [
        f"最终Accuracy = {n4(dnn['accuracy'])}。",
        f"最终Macro F1 = {n4(dnn['macro_f1'])}。",
        "相比基础模型，深度学习模型明显提升。",
    ], 7.62, 4.25, 3.95, 1.20, accent=ORANGE, size=11.2)
    page += 1

    # 32
    slide = new_slide(prs, "实验结果", "混淆矩阵与类别表现", "PG和C识别较稳定，PF/SF/SG是主要混淆区域", page, total)
    image_box(slide, RESULTS / "experiment5_dnn" / "confusion_matrix.png", 0.75, 1.42, 5.30, 3.95, title="最终模型混淆矩阵")
    class_rows = []
    for pos in ["C", "PF", "PG", "SF", "SG"]:
        sub = class_metrics[class_metrics["Position"] == pos].set_index("Metric")["Score"]
        class_rows.append([pos, n4(float(sub["precision"])), n4(float(sub["recall"])), n4(float(sub["f1-score"]))])
    table(slide, class_rows, ["位置", "Precision", "Recall", "F1"], 6.65, 1.58, 4.65, 2.45, widths=[0.70, 1.25, 1.25, 1.25], font_size=10.0, header=BLUE)
    callout(slide, "读图结论", [
        "对角线表示预测正确数量。",
        "PG识别最稳定，F1约0.8605。",
        "SF/SG、C/PF等相邻位置仍存在明显混淆。",
    ], 6.65, 4.45, 4.65, 1.15, accent=TEAL, size=11.2)
    page += 1

    # 33
    slide = new_slide(prs, "消融实验", "模块贡献分析", "通过移除模块观察Macro F1下降，验证模型提升来源", page, total)
    image_box(slide, RESULTS / "experiment5_ablation" / "ablation_delta_macro_f1_cn.png", 0.85, 1.42, 6.70, 4.20, title="中文消融实验结果")
    callout(slide, "主要结论", [
        "DNN主干和DNN集成贡献最大。",
        "工程特征、年代专家也有明显贡献。",
        "ExtraTrees和对比学习带来小幅但稳定提升。",
        "完整模型不是单点技巧，而是多模块组合。",
    ], 8.05, 1.70, 3.70, 2.05, accent=TEAL, size=11.0)
    callout(slide, "关键数值", [
        "单个Residual DNN下降约0.0340。",
        "去掉工程特征下降约0.0235。",
        "去掉年代专家下降约0.0176。",
        "去掉ExtraTrees下降约0.0045。",
    ], 8.05, 4.25, 3.70, 1.55, accent=ORANGE, size=11.0)
    page += 1

    # 34
    slide = new_slide(prs, "总结", "最终总结", "实验五通过合规技术特征和多模块融合，取得本项目最佳分类效果", page, total)
    card(slide, 0.95, 1.55, 2.15, 0.92, "输入", "45维", ORANGE, note="核心技术画像")
    card(slide, 3.40, 1.55, 2.15, 0.92, "主干", "Residual DNN", BLUE, body_size=15)
    card(slide, 5.85, 1.55, 2.15, 0.92, "增强", "年代专家", TEAL, body_size=16)
    card(slide, 8.30, 1.55, 2.15, 0.92, "融合", "ExtraTrees", RED, body_size=16)
    callout(slide, "核心结论", [
        "特征筛选将76维候选特征压缩到45维，效果基本保持。",
        "Residual DNN学习复杂非线性位置边界。",
        "监督对比学习和DNN集成提升内部表示和预测稳定性。",
        "年代专家吸收时代差异，ExtraTrees提供辅助概率补充。",
        f"最终Accuracy为{n4(dnn['accuracy'])}，Macro F1为{n4(dnn['macro_f1'])}。",
    ], 1.00, 3.15, 10.90, 2.35, accent=TEAL, size=11.8)
    quote_box(slide, "一句话总结：最终模型的优势来自“合规技术特征 + 非线性表示 + 集成学习 + 年代专家 + 概率融合”。", 1.00, 6.15, 10.90, 0.52)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"Saved PPT: {OUT_PATH}")
    print(f"Source markdown length: {len(md_text)}")


if __name__ == "__main__":
    main()
