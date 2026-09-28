from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_PATH = RESULTS / "NBA_position_deep_learning_report.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

BG = RGBColor(248, 250, 252)
NAVY = RGBColor(20, 40, 70)
BLUE = RGBColor(46, 111, 164)
GREEN = RGBColor(42, 157, 143)
RED = RGBColor(209, 73, 91)
GREY = RGBColor(90, 102, 117)
LIGHT_BLUE = RGBColor(227, 239, 250)
LIGHT_GREEN = RGBColor(229, 246, 242)
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(20, 20, 20)

FONT = "Microsoft YaHei"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if hasattr(shape, "line"):
        shape.line.color.rgb = color


def set_text_style(paragraph, size=18, bold=False, color=BLACK) -> None:
    paragraph.font.name = FONT
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color


def add_textbox(slide, text, x, y, w, h, size=18, bold=False, color=BLACK, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    set_text_style(p, size=size, bold=bold, color=color)
    if align is not None:
        p.alignment = align
    return box


def add_bullets(slide, items, x, y, w, h, size=17, color=BLACK, bullet=True, line_spacing=1.05):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.line_spacing = line_spacing
        if bullet:
            p.text = f"• {item}"
    return box


def add_title(slide, title: str, subtitle: str | None = None):
    add_textbox(slide, title, 0.55, 0.30, 12.2, 0.45, size=24, bold=True, color=NAVY)
    if subtitle:
        add_textbox(slide, subtitle, 0.58, 0.78, 11.8, 0.30, size=11, color=GREY)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.08), Inches(1.05), Inches(0.045))
    set_fill(line, GREEN)


def add_footer(slide, page: int):
    add_textbox(slide, f"NBA Position Prediction  |  {page}", 10.9, 7.08, 1.8, 0.25, size=9, color=GREY, align=PP_ALIGN.RIGHT)


def add_card(slide, x, y, w, h, title, body, accent=BLUE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    set_fill(shape, WHITE)
    shape.line.color.rgb = RGBColor(226, 232, 240)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.06), Inches(h))
    set_fill(bar, accent)
    add_textbox(slide, title, x + 0.18, y + 0.12, w - 0.3, 0.30, size=13, bold=True, color=NAVY)
    add_textbox(slide, body, x + 0.18, y + 0.48, w - 0.32, h - 0.55, size=18, bold=True, color=BLACK)


def add_table(slide, dataframe: pd.DataFrame, x, y, w, h, font_size=10, header_color=NAVY):
    rows, cols = dataframe.shape
    table = slide.shapes.add_table(rows + 1, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, col in enumerate(dataframe.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        set_fill(cell, header_color)
        for p in cell.text_frame.paragraphs:
            set_text_style(p, size=font_size, bold=True, color=WHITE)
    for i in range(rows):
        for j, col in enumerate(dataframe.columns):
            cell = table.cell(i + 1, j)
            cell.text = str(dataframe.iloc[i, j])
            set_fill(cell, RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(241, 245, 249))
            if hasattr(cell, "line"):
                cell.line.color.rgb = RGBColor(226, 232, 240)
            for p in cell.text_frame.paragraphs:
                set_text_style(p, size=font_size, color=BLACK)
    return table


def add_image(slide, path: Path, x, y, w=None, h=None):
    if not path.exists():
        add_card(slide, x, y, w or 4, h or 2, "缺失图片", path.name, accent=RED)
        return None
    if w is not None and h is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    if w is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    if h is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y))


def add_process_box(slide, text, x, y, w, h, fill=LIGHT_BLUE, color=NAVY):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    set_fill(shape, fill)
    shape.line.color.rgb = RGBColor(203, 213, 225)
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    set_text_style(p, size=13, bold=True, color=color)
    p.alignment = PP_ALIGN.CENTER
    return shape


def add_arrow(slide, x1, y1, x2, y2):
    line = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = GREY
    line.line.width = Pt(1.3)
    return line


def setup_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return slide


def main() -> None:
    metrics = {
        "bayes": read_json(RESULTS / "experiment1_bayes" / "metrics.json"),
        "kmeans": read_json(RESULTS / "experiment2_kmeans" / "metrics.json"),
        "id3": read_json(RESULTS / "experiment3_id3" / "metrics.json"),
        "c45": read_json(RESULTS / "experiment4_c45" / "metrics.json"),
        "dnn": read_json(RESULTS / "experiment5_dnn" / "metrics.json"),
    }
    comparison = pd.read_csv(RESULTS / "model_comparison_summary.csv")
    class_report = pd.read_csv(RESULTS / "experiment5_dnn" / "class_metrics.csv")
    ablation = pd.read_csv(RESULTS / "experiment5_ablation" / "ablation_results.csv")

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    page = 1

    slide = setup_slide(prs)
    add_textbox(slide, "NBA球员位置预测实验汇报", 0.75, 0.72, 11.9, 0.65, size=32, bold=True, color=NAVY)
    add_textbox(slide, "深度学习模型设计与传统模型对比", 0.78, 1.42, 11.6, 0.45, size=19, color=GREY)
    add_card(slide, 0.85, 2.35, 2.6, 1.35, "最终模型", "45维特征\nDNN融合", accent=BLUE)
    add_card(slide, 3.75, 2.35, 2.6, 1.35, "Accuracy", f"{metrics['dnn']['accuracy']:.4f}", accent=GREEN)
    add_card(slide, 6.65, 2.35, 2.6, 1.35, "Macro F1", f"{metrics['dnn']['macro_f1']:.4f}", accent=GREEN)
    add_card(slide, 9.55, 2.35, 2.6, 1.35, "对比对象", "贝叶斯 / K-Means\nID3 / C4.5", accent=BLUE)
    add_image(slide, RESULTS / "supervised_model_comparison.png", 1.05, 4.05, w=5.45, h=2.55)
    add_image(slide, RESULTS / "experiment5_dnn" / "confusion_matrix.png", 7.15, 3.92, w=4.95, h=2.75)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "汇报结构", "重点讲实验五深度学习模型，同时保留五个实验横向对比")
    add_bullets(slide, [
        "实验任务与数据处理：NBA技术统计 -> 五位置分类",
        "前四个模型：贝叶斯、K-Means、ID3、C4.5 的方法和局限",
        "实验五：45维核心特征 + Residual DNN集成 + 年代专家 + ExtraTrees概率融合",
        "模型结果：分类报告、混淆矩阵、训练曲线与特征筛选",
        "对比与结论：深度学习模型优势和消融实验支撑",
    ], 0.95, 1.55, 11.6, 4.7, size=22, bullet=False)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "实验任务与数据流程", "目标：根据球员赛季技术统计预测 C / PF / PG / SF / SG")
    add_process_box(slide, "NBA_Season_Stats.csv", 0.75, 1.55, 2.25, 0.72)
    add_arrow(slide, 3.0, 1.91, 3.45, 1.91)
    add_process_box(slide, "删除身份字段\nPlayer / Tm", 3.45, 1.45, 2.15, 0.92, fill=LIGHT_GREEN)
    add_arrow(slide, 5.60, 1.91, 6.05, 1.91)
    add_process_box(slide, "训练/测试划分\n14981 / 3746", 6.05, 1.45, 2.25, 0.92)
    add_arrow(slide, 8.30, 1.91, 8.75, 1.91)
    add_process_box(slide, "五类位置预测\nC PF PG SF SG", 8.75, 1.45, 2.65, 0.92, fill=LIGHT_GREEN)
    data_cards = [
        ("分类目标", "五分类\nC/PF/PG/SF/SG", BLUE),
        ("传统特征", "26维\n原始数值统计", BLUE),
        ("深度学习特征", "45维\n核心技术画像", GREEN),
        ("合规约束", "不使用\nPlayer / Tm", RED),
    ]
    for i, (t, b, c) in enumerate(data_cards):
        add_card(slide, 0.85 + i * 3.05, 3.20, 2.65, 1.35, t, b, accent=c)
    add_bullets(slide, [
        "位置标签来自原始 Pos 字段；所有模型使用同一份预处理结果。",
        "深度学习模型只基于技术统计和派生指标，不使用任何身份类泄露信息。",
    ], 1.0, 5.25, 11.2, 1.0, size=18)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "五个实验总体对比", "深度学习模型在监督分类指标上明显领先")
    add_image(slide, RESULTS / "supervised_model_comparison.png", 0.75, 1.28, w=6.15, h=3.75)
    table_df = pd.DataFrame([
        ["Bayes", "0.4477", "0.4027"],
        ["ID3", "0.4426", "0.4432"],
        ["C4.5", "0.4389", "0.4392"],
        ["DNN融合", "0.7261", "0.7258"],
    ], columns=["模型", "Accuracy", "Macro F1"])
    add_table(slide, table_df, 7.30, 1.45, 4.8, 2.25, font_size=12)
    add_bullets(slide, [
        "DNN 相比贝叶斯：Macro F1 提升约 0.3230",
        "DNN 相比 ID3：Macro F1 提升约 0.2826",
        "DNN 相比 C4.5：Macro F1 提升约 0.2866",
    ], 7.35, 4.10, 4.75, 1.7, size=16)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "前四个模型与局限", "基础算法可以作为对照，但难以充分表达 NBA 位置的连续边界")
    trad_df = pd.DataFrame([
        ["贝叶斯", "监督分类", "假设特征条件独立", "统计特征强相关，独立性假设偏弱"],
        ["K-Means", "无监督聚类", "按距离形成5个簇", "缺少标签监督，位置簇混合明显"],
        ["ID3", "监督分类", "信息增益划分离散特征", "离散化损失数值细节，单树不稳"],
        ["C4.5", "监督分类", "信息增益率改进ID3", "基础单树版本仍难拟合复杂边界"],
    ], columns=["模型", "类型", "核心思想", "主要局限"])
    add_table(slide, trad_df, 0.55, 1.35, 12.25, 3.35, font_size=9)
    add_image(slide, RESULTS / "kmeans_clustering_metrics.png", 0.85, 5.03, w=3.75, h=1.95)
    add_image(slide, RESULTS / "experiment2_kmeans" / "pca_clusters.png", 5.0, 4.82, w=3.7, h=2.15)
    add_image(slide, RESULTS / "experiment2_kmeans" / "cluster_position_heatmap.png", 9.05, 4.86, w=3.45, h=2.05)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "实验五模型总览", "45维核心特征 + 全局DNN + 年代专家 + ExtraTrees概率融合")
    add_process_box(slide, "45维核心技术特征", 0.65, 2.05, 2.15, 0.80, fill=LIGHT_GREEN)
    add_arrow(slide, 2.80, 2.45, 3.25, 2.45)
    add_process_box(slide, "全局DNN集成\n3 Residual + 3 Contrastive", 3.25, 1.45, 2.45, 1.05)
    add_process_box(slide, "年代专家DNN\n1980s/1990s/2000s/2010s+", 3.25, 2.85, 2.45, 1.05)
    add_process_box(slide, "ExtraTrees\n辅助概率", 3.25, 4.25, 2.45, 0.90, fill=LIGHT_GREEN)
    add_arrow(slide, 5.70, 2.00, 6.35, 2.75)
    add_arrow(slide, 5.70, 3.38, 6.35, 2.95)
    add_arrow(slide, 5.70, 4.70, 6.35, 3.16)
    add_process_box(slide, "概率融合\n0.55 * DNN + 0.45 * Tree", 6.35, 2.50, 2.55, 0.95, fill=LIGHT_GREEN)
    add_arrow(slide, 8.90, 2.98, 9.40, 2.98)
    add_process_box(slide, "最终预测\nC / PF / PG / SF / SG", 9.40, 2.50, 2.65, 0.95)
    add_bullets(slide, [
        "核心思想：用 DNN 学习非线性技术画像，用年代专家吸收时代差异。",
        "ExtraTrees 不单独替代 DNN，而是作为错误模式不同的概率补充。",
        "最终采用软概率融合，而不是硬投票。",
    ], 0.85, 5.65, 11.4, 1.05, size=16)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "45维核心特征体系", "从76维候选特征压缩到45维，保持效果并提高可解释性")
    add_image(slide, RESULTS / "experiment5_feature_selection_search" / "compact_feature_search_macro_f1.png", 0.60, 1.25, w=5.65, h=3.25)
    add_image(slide, RESULTS / "experiment5_feature_selection_search" / "feature_count_vs_macro_f1.png", 6.70, 1.25, w=5.65, h=3.25)
    add_bullets(slide, [
        "45维版本比76维减少约40.8%的输入特征。",
        "Accuracy 保持 0.7261，Macro F1 为 0.7258。",
        "特征集中保留效率、出场强度、投篮结构、组织、防守和相邻位置分离指标。",
    ], 0.85, 5.05, 11.6, 1.2, size=17)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "Residual DNN 与监督对比学习", "用残差网络学习非线性统计组合，用对比损失拉开模糊位置边界")
    add_process_box(slide, "输入45维", 0.70, 2.20, 1.25, 0.60)
    add_arrow(slide, 1.95, 2.50, 2.35, 2.50)
    add_process_box(slide, "Linear\nBatchNorm\nGELU\nDropout", 2.35, 1.85, 1.65, 1.30)
    add_arrow(slide, 4.00, 2.50, 4.40, 2.50)
    add_process_box(slide, "Residual Block ×2\nx + F(x)", 4.40, 1.90, 1.85, 1.20, fill=LIGHT_GREEN)
    add_arrow(slide, 6.25, 2.50, 6.70, 2.50)
    add_process_box(slide, "分类头\n5类概率", 6.70, 2.00, 1.50, 0.95)
    add_arrow(slide, 8.20, 2.50, 8.65, 2.50)
    add_process_box(slide, "预测位置", 8.65, 2.15, 1.35, 0.70)
    add_process_box(slide, "Projection Head\n监督对比学习", 6.70, 3.55, 2.15, 0.90, fill=LIGHT_GREEN)
    add_textbox(slide, "Loss = CrossEntropy + 0.1 × SupervisedContrastiveLoss", 0.85, 4.65, 11.5, 0.45, size=19, bold=True, color=NAVY)
    add_bullets(slide, [
        "同位置样本在隐空间更接近，不同位置样本更分离。",
        "主要服务于 C/PF、PF/SF、PG/SG 等相邻位置区分。",
        "消融结果：去掉对比学习损失后 Macro F1 下降约 0.0011。",
    ], 1.0, 5.35, 11.3, 1.0, size=16)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "全局集成与年代专家", "NBA位置职责随年代变化，分年代建模能学习更局部的边界")
    add_card(slide, 0.75, 1.35, 2.55, 1.20, "全局集成", "6个DNN\n平均概率", accent=BLUE)
    add_card(slide, 3.60, 1.35, 2.55, 1.20, "年代专家", "4个年代段\n每段4个DNN", accent=GREEN)
    add_card(slide, 6.45, 1.35, 2.55, 1.20, "DNN概率", "0.6全局\n+0.4年代", accent=BLUE)
    add_card(slide, 9.30, 1.35, 2.55, 1.20, "最终融合", "DNN +\nExtraTrees", accent=GREEN)
    add_bullets(slide, [
        "年代划分：1980s、1990s、2000s、2010s+。",
        "现代篮球中后卫、锋线、空间型内线的统计边界发生变化。",
        "消融结果：去掉年代专家后 Macro F1 从 0.7260 降到 0.7084。",
    ], 0.95, 3.10, 5.6, 2.0, size=17)
    small_df = pd.DataFrame([
        ["Global only", "0.6997", "0.7001"],
        ["Era only", "0.7125", "0.7122"],
        ["DNN ensemble", "0.7168", "0.7167"],
        ["Final fusion", "0.7261", "0.7258"],
    ], columns=["组件", "Accuracy", "Macro F1"])
    add_table(slide, small_df, 7.05, 3.0, 4.85, 2.35, font_size=11)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "训练与验证过程", "验证曲线和参数对比用于说明训练稳定性")
    add_image(slide, RESULTS / "experiment5_dnn" / "training_curve.png", 0.70, 1.25, w=5.65, h=3.50)
    add_image(slide, RESULTS / "experiment5_dnn" / "validation_comparison.png", 6.80, 1.25, w=5.45, h=3.50)
    add_bullets(slide, [
        f"最佳验证配置：{metrics['dnn']['best_config']['hidden_layers']}，dropout={metrics['dnn']['best_config']['dropout']}",
        f"最佳验证 Macro F1：{metrics['dnn']['best_val_macro_f1']:.4f}",
        "最终模型使用全训练集重训，并在测试集上统一评估。",
    ], 0.95, 5.20, 11.3, 1.0, size=16)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "实验五分类效果", "PG和C识别较好，PF/SF/SG是主要混淆区域")
    add_image(slide, RESULTS / "experiment5_dnn" / "class_metrics.png", 0.65, 1.25, w=5.75, h=3.55)
    cls_df = pd.DataFrame([
        ["C", "0.7653", "0.7450", "0.7550"],
        ["PF", "0.6574", "0.6591", "0.6582"],
        ["PG", "0.8505", "0.8708", "0.8605"],
        ["SF", "0.6583", "0.6583", "0.6583"],
        ["SG", "0.6969", "0.6969", "0.6969"],
    ], columns=["位置", "Precision", "Recall", "F1"])
    add_table(slide, cls_df, 7.00, 1.35, 5.05, 2.65, font_size=10)
    add_card(slide, 7.05, 4.40, 1.65, 0.95, "Accuracy", "0.7261", accent=GREEN)
    add_card(slide, 8.95, 4.40, 1.65, 0.95, "Macro F1", "0.7258", accent=GREEN)
    add_card(slide, 10.85, 4.40, 1.65, 0.95, "Weighted F1", "0.7259", accent=GREEN)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "混淆矩阵分析", "模型主要错误集中在相邻或摇摆位置之间")
    add_image(slide, RESULTS / "experiment5_dnn" / "confusion_matrix.png", 0.70, 1.10, w=6.20, h=4.55)
    add_bullets(slide, [
        "PG 识别最稳定：654 / 751 个样本预测正确。",
        "C 与 PF 有明显混淆：C->PF 162，PF->C 148。",
        "SF 与 SG 混淆明显：SF->SG 118，SG->SF 118。",
        "这符合篮球位置连续性：锋线、摇摆人和空间型内线边界本身更模糊。",
    ], 7.25, 1.55, 5.20, 3.80, size=17)
    add_image(slide, RESULTS / "experiment5_dnn" / "prediction_distribution.png", 7.35, 5.15, w=4.60, h=1.65)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "消融实验：模块贡献", "去掉模块后指标下降，证明深度学习方案不是单点提升")
    add_image(slide, RESULTS / "experiment5_ablation" / "ablation_delta_macro_f1.png", 0.65, 1.18, w=6.05, h=4.0)
    ab_df = pd.DataFrame([
        ["单个Residual DNN", "-0.0340"],
        ["去掉工程特征", "-0.0235"],
        ["去掉年代专家", "-0.0176"],
        ["去掉ExtraTrees融合", "-0.0045"],
        ["去掉Contrastive成员", "-0.0045"],
        ["去掉对比损失", "-0.0011"],
    ], columns=["消融项", "Macro F1变化"])
    add_table(slide, ab_df, 7.10, 1.35, 5.05, 3.0, font_size=11)
    add_bullets(slide, [
        "贡献最大：工程特征、DNN集成、年代专家。",
        "小幅增强：ExtraTrees概率融合、Contrastive成员、监督对比损失。",
        "特征筛选主要价值是降维和解释性，而不是大幅提分。",
    ], 7.20, 4.80, 5.0, 1.30, size=15)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "特征筛选专项结果", "45维版本作为正式模型：输入更少，效果基本保持")
    fs_df = pd.DataFrame([
        ["76维全量", "76", "0.7253", "0.7257"],
        ["59维旧核心", "59", "0.7261", "0.7260"],
        ["45维正式版", "45", "0.7261", "0.7258"],
        ["35维Top重要性", "35", "0.7157", "0.7158"],
        ["32维画像核心", "32", "0.7058", "0.7057"],
    ], columns=["特征集", "特征数", "Accuracy", "Macro F1"])
    add_table(slide, fs_df, 0.75, 1.35, 5.35, 2.55, font_size=10)
    add_image(slide, RESULTS / "experiment5_feature_selection_search" / "feature_count_vs_macro_f1.png", 6.65, 1.18, w=5.65, h=3.55)
    add_bullets(slide, [
        "正式模型从76维压缩到45维，减少约40.8%。",
        "Accuracy 保持0.7261，Macro F1保持在0.7258。",
        "最终特征集中保留相邻位置分离指标，便于解释模型判断依据。",
    ], 0.95, 4.60, 11.3, 1.20, size=17)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "为什么深度学习模型明显更好", "从模型假设、特征表达和概率融合三个层面解释")
    add_card(slide, 0.75, 1.45, 3.70, 1.35, "1. 非线性表达", "学习得分、篮板、助攻、投篮结构之间的组合关系", accent=BLUE)
    add_card(slide, 4.82, 1.45, 3.70, 1.35, "2. 篮球语义特征", "每36分钟、比例指标、位置画像比原始总量更有效", accent=GREEN)
    add_card(slide, 8.90, 1.45, 3.70, 1.35, "3. 年代分层", "吸收不同时代位置职责和三分出手变化", accent=BLUE)
    add_card(slide, 0.75, 3.35, 3.70, 1.35, "4. 多模型集成", "降低单模型随机性，稳定边界概率", accent=GREEN)
    add_card(slide, 4.82, 3.35, 3.70, 1.35, "5. 辅助树融合", "利用与DNN不同的错误模式补充概率", accent=BLUE)
    add_card(slide, 8.90, 3.35, 3.70, 1.35, "最终效果", "Accuracy 0.7261\nMacro F1 0.7258", accent=GREEN)
    add_bullets(slide, [
        "传统模型更多依赖单一假设或单棵树划分；深度学习模型能整合多个弱边界信号。",
        "NBA位置本身具有连续性，尤其PF/SF/SG边界，因此概率融合比硬规则更合适。",
    ], 0.95, 5.45, 11.3, 0.95, size=16)
    add_footer(slide, page); page += 1

    slide = setup_slide(prs)
    add_title(slide, "结论", "实验五深度学习融合模型是本项目中表现最好的方法")
    add_textbox(slide, "最终模型", 0.95, 1.35, 2.5, 0.35, size=18, bold=True, color=NAVY)
    add_bullets(slide, [
        "45维核心技术特征",
        "Residual DNN + Contrastive Residual DNN 全局集成",
        "年代专家 DNN 建模不同时期位置风格",
        "ExtraTrees 辅助概率融合",
    ], 1.0, 1.80, 5.3, 2.1, size=18)
    add_textbox(slide, "最终指标", 7.0, 1.35, 2.5, 0.35, size=18, bold=True, color=NAVY)
    add_card(slide, 7.05, 1.85, 1.75, 1.05, "Accuracy", "0.7261", accent=GREEN)
    add_card(slide, 9.05, 1.85, 1.75, 1.05, "Macro F1", "0.7258", accent=GREEN)
    add_card(slide, 11.05, 1.85, 1.75, 1.05, "特征数", "45", accent=BLUE)
    add_textbox(slide, "汇报结论", 0.95, 4.35, 2.5, 0.35, size=18, bold=True, color=NAVY)
    add_bullets(slide, [
        "深度学习模型显著优于贝叶斯、ID3、C4.5等基础监督模型。",
        "K-Means适合探索数据结构，但难以直接恢复真实位置标签。",
        "最终模型在合规约束下不使用身份信息，仍能取得最优分类效果。",
    ], 1.0, 4.80, 11.2, 1.35, size=18)
    add_footer(slide, page); page += 1

    prs.save(OUT_PATH)
    print(f"Saved PPT: {OUT_PATH}")


if __name__ == "__main__":
    main()
