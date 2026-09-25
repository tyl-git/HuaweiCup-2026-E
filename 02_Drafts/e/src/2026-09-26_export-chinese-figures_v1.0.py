"""Export Chinese publication figures for HuaweiCup-2026 E, preserving source data and English figures."""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FONT = Path(r"C:/Windows/Fonts/simsun.ttc")
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
MOD_ZH = {"text": "文本", "audio": "音频", "vision": "视觉"}
MODEL_ZH = {"temporal": "时序模型", "balanced": "类别均衡时序模型", "text_safe": "输入级文本缺失训练"}
POL_ZH = {"Negative": "负面", "Neutral": "中性", "Positive": "正面"}
COLORS = {"text": "#2878B5", "audio": "#D17A22", "vision": "#4A9A62"}

def setup_font():
    if FONT.exists():
        font_manager.fontManager.addfont(str(FONT))
        name = font_manager.FontProperties(fname=str(FONT)).get_name()
        plt.rcParams["font.family"] = name
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 10,
                         "legend.fontsize": 9, "figure.dpi": 140,
                         "savefig.dpi": 220})

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", facecolor="white")
    plt.close(fig)

def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E4E8EA", linewidth=.8)
    ax.set_axisbelow(True)

def q1(root):
    out = root / "03_Results/e/\u95ee\u9898\u4e00_\u7279\u5f81\u63d0\u53d6\u4e0e\u65f6\u5e8f\u5bf9\u9f50"
    s = read_json(root / "03_Results/e/question-one/paper-assets-v1.0/q1_summary_v1.0.json")["historical_v1_0"]
    labels = ["原始词", "文本有效", "有时间/音频有效", "三模态同时有效"]
    vals = [s["source_words"], s["text_mask_words"], s["audio_mask_words"], s["all_modalities_mask_words"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    bars = ax.bar(labels, vals, color=["#8A9BA8", "#2878B5", "#D17A22", "#4A9A62"], width=.62)
    ax.set_title("问题一：三模态词级对齐的自动有效覆盖")
    ax.set_ylabel("词数（不是对齐准确率）")
    ax.set_ylim(0, max(vals)*1.18)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+25, f"{v:,}", ha="center", va="bottom")
    style_ax(ax); save(fig, out, "q1_zh_multimodal_coverage")
   
    steps = ["原始视频", "语音波形", "CTC词级时间", "BERT词向量", "eGeMAPS音频", "OpenFace视觉", "交叠汇聚", "词级NPZ+掩码"]
    fig, ax = plt.subplots(figsize=(13, 4.0), layout="constrained")
    ax.axis("off")
    xs = np.linspace(.06, .94, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.text(x, .52, textwrap.fill(label, 9), ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=.55", "facecolor": "#EAF3F7" if i < 4 else "#EEF6EE",
                      "edgecolor": "#5D879A" if i < 4 else "#6C9B73"})
        if i < len(steps)-1: ax.annotate("", xy=(xs[i+1]-.055, .52), xytext=(x+.055, .52),
                    arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#667"})
    ax.text(.5, .92, "问题一：特征提取与词级时序对齐流程", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(.5, .12, "文本、音频和视觉分别提取后，按共同时间轴对每个原词做交叠加权汇聚，并保存有效掩码。", ha="center", va="center", fontsize=10)
    save(fig, out, "q1_zh_feature_alignment_pipeline")

def q2(root):
    out = root / "03_Results/e/\u95ee\u9898\u4e8c_\u5c40\u90e8\u7f3a\u5931\u60c5\u611f\u8bc6\u522b"
    q2dir = root / "03_Results/e/question-two"
    rows = read_csv(q2dir / "q2-improvement-evidence-v1.1/seed_metrics.csv")
    models = ["temporal", "balanced", "text_safe"]
    def model_key(row):
        m=row["model"]
        return "balanced" if m=="balanced" else ("text_safe" if "text_safe" in m else "temporal")
    fig, axes = plt.subplots(1,2,figsize=(10.5,4.8),layout="constrained")
    for ax, field, title, ylabel, lower in [(axes[0],"macro_f1","正式测试集分类性能（3个种子）","Macro-F1（越高越好）",False),(axes[1],"mae","正式测试集回归性能（3个种子）","MAE（越低越好）",True)]:
        means=[]; sd=[]
        for m in models:
            a=np.array([float(r[field]) for r in rows if r["split"]=="test" and model_key(r)==m])
            means.append(a.mean()); sd.append(a.std(ddof=1))
        x=np.arange(3); bars=ax.bar(x,means,yerr=sd,capsize=4,color=["#78909C","#2878B5","#D17A22"],width=.62)
        ax.set_xticks(x,["原始时序","类别均衡时序","输入级文本缺失"],rotation=12,ha="right")
        ax.set_title(title); ax.set_ylabel(ylabel); style_ax(ax)
        for b,v in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,v+(.006 if not lower else .001),f"{v:.3f}",ha="center",va="bottom")
    save(fig,out,"q2_zh_official_test_model_comparison")

    summ=read_json(q2dir/"q2-temporal-balanced-sqrt-ensemble-final-v1.1/summary.json")
    base=summ["validation"]["classification"]["macro_f1"]
    cond=[k for k in summ["validation_missing"] if "missing" in k]
    mods=["text","audio","vision"]; fracs=["20pct","40pct"]; poss=["start","middle","end"]
    mat=np.full((3,6),np.nan)
    labels=[]
    for j,(f,p) in enumerate((x for x in [(f,p) for f in fracs for p in poss])):
        labels.append(("20%" if f=="20pct" else "40%")+"\n"+("起始" if p=="start" else "中段" if p=="middle" else "末段"))
        for i,m in enumerate(mods):
            key=f"{m}_missing_{f}_{p}"
            if key in summ["validation_missing"]: mat[i,j]=summ["validation_missing"][key]["classification"]["macro_f1"]-base
    lim=np.nanmax(np.abs(mat))
    fig,ax=plt.subplots(figsize=(10.5,3.9),layout="constrained")
    im=ax.imshow(mat,cmap="BrBG",vmin=-lim,vmax=lim,aspect="auto")
    ax.set_xticks(np.arange(6),labels); ax.set_yticks(np.arange(3),[MOD_ZH[m] for m in mods]); ax.set_title("问题二：验证集局部连续缺失下 Macro-F1 相对变化")
    for i in range(3):
        for j in range(6): ax.text(j,i,f"{mat[i,j]:+.3f}",ha="center",va="center",fontsize=9)
    fig.colorbar(im,ax=ax,label="相对完整输入的变化",shrink=.82); save(fig,out,"q2_zh_contiguous_missing_heatmap")

    cm=np.array(summ["official_test"]["classification"]["confusion_matrix_true_rows_predicted_columns"])
    fig,ax=plt.subplots(figsize=(5.8,4.8),layout="constrained"); im=ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(3),["负面","中性","正面"]); ax.set_yticks(range(3),["负面","中性","正面"]); ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别"); ax.set_title("问题二：正式测试集混淆矩阵")
    for i in range(3):
        for j in range(3): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>100 else "#18324A")
    fig.colorbar(im,ax=ax,label="样本数",shrink=.82); save(fig,out,"q2_zh_official_test_confusion")

    # Seed stability figure uses only the archived seed_metrics.csv; it never invents a learning curve.
    seed_rows = [r for r in rows if r["split"] == "valid" and model_key(r) == "balanced"]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    x = np.arange(len(seed_rows))
    vals = [float(r["macro_f1"]) for r in seed_rows]
    bars = ax.bar(x, vals, color="#2878B5", width=.58)
    ax.set_xticks(x, [f"种子 {r['seed']}" for r in seed_rows], rotation=12, ha="right")
    ax.set_xlabel("固定随机种子"); ax.set_ylabel("验证集 Macro-F1"); ax.set_title("问题二：类别均衡时序模型的种子稳定性"); style_ax(ax)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+.004, f"{v:.3f}", ha="center")
    save(fig, out, "q2_zh_balanced_seed_stability")

    # Text-gap protocol comparison from frozen report.
    safe = read_json(q2dir / "q2-text-safe-eval-balanced-sqrt-v1.0/report.json")
    conds = ["20pct_start","20pct_middle","20pct_end","40pct_start","40pct_middle","40pct_end"]
    post=[]; inp=[]
    for c in conds:
        post.append(float(np.mean([r["macro_f1"] for r in safe["rows"] if r["condition"]==c and r["encoding"]=="post_bert"])))
        inp.append(float(np.mean([r["macro_f1"] for r in safe["rows"] if r["condition"]==c and r["encoding"]=="input_level"])))
    fig,ax=plt.subplots(figsize=(9.2,4.8),layout="constrained"); x=np.arange(6); ax.plot(x,post,marker="o",lw=2,label="BERT后位置清零"); ax.plot(x,inp,marker="s",lw=2,label="BERT前[MASK]"); ax.set_xticks(x,["20%起","20%中","20%末","40%起","40%中","40%末"]); ax.set_ylim(.43,.63); ax.set_ylabel("验证集 Macro-F1（3种子均值）"); ax.set_title("问题二：相同缺失区间在不同文本处理阶段的对比"); ax.legend(frameon=False); style_ax(ax); save(fig,out,"q2_zh_text_masking_stage_comparison")

def q3(root):
    out=root/"03_Results/e/\u95ee\u9898\u4e09_\u53ef\u89e3\u91ca\u9884\u6d4b"; q3dir=root/"03_Results/e/question-three/q3-balanced-ensemble-explanations-v1.1"
    rows=read_csv(q3dir/"attachment4_explanations.csv")
    vals={m:[] for m in MODALITIES}
    for r in rows:
        e=json.loads(r["modality_effects_json"])
        for m in MODALITIES: vals[m].append(float(e[m]["relative_influence"]))
    fig,ax=plt.subplots(figsize=(10.5,4.8),layout="constrained"); x=np.arange(len(rows)); bottom=np.zeros(len(rows))
    for m in MODALITIES:
        a=np.asarray(vals[m]); ax.bar(x,a,bottom=bottom,color=COLORS[m],label=MOD_ZH[m],width=.78); bottom+=a
    ax.set_xticks(x,[r["file_id"] for r in rows],fontsize=8); ax.set_xlabel("附件四样本编号"); ax.set_ylabel("归一化模态影响"); ax.set_ylim(0,1); ax.set_title("问题三：逐模态遮挡的相对影响（20条附件四样本）"); ax.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02)); save(fig,out,"q3_zh_modality_occlusion_influence")

    fig,ax=plt.subplots(figsize=(6.8,4.5),layout="constrained"); s=read_json(root/"03_Results/e/question-three/q3-balanced-ensemble-figures-v1.1/summary.json"); a=[s["candidate_mean_signed_delta"],s["random_mean_signed_delta"]]; bars=ax.bar(["门控×时间候选位置","固定随机位置"],a,color=["#2878B5","#9AA6AC"],width=.58); ax.set_ylabel("目标类别概率下降（带符号）"); ax.set_title("问题三：候选证据与随机位置的遮挡对照"); style_ax(ax); [ax.text(b.get_x()+b.get_width()/2,v+.001,f"{v:.3f}",ha="center") for b,v in zip(bars,a)]; ax.text(.5,-.22,"门控和时间权重用于候选排序，遮挡变化用于验证模型敏感性；不作因果解释。",transform=ax.transAxes,ha="center",fontsize=9); save(fig,out,"q3_zh_candidate_vs_random_occlusion")

    pos=read_csv(q3dir/"attachment4_position_effects.csv"); item=next(r for r in rows if r["file_id"]=="04"); local=[r for r in pos if r["file_id"]=="04" and r["modality"]==item["primary_modality"]]; local.sort(key=lambda r:int(r["position_0based"]))
    labels=[r["text_fragment"] or f"位置{int(r['position_0based'])+1}" for r in local]; d=np.array([float(r["class_probability_delta"]) for r in local]); fig=plt.figure(figsize=(12,6.4),layout="constrained"); gs=fig.add_gridspec(2,2,height_ratios=[1,1.35]); ax=fig.add_subplot(gs[0,0]); ax.axis("off"); info=f"附件四样本04\n预测极性：{POL_ZH.get(item['polarity'],item['polarity'])}\n预测强度：{float(item['intensity']):.3f}\n目标类别概率：{float(item['predicted_class_probability']):.3f}\n主要敏感模态：{MOD_ZH.get(item['primary_modality'],item['primary_modality'])}\n\n原始文本：{item['raw_text']}"; ax.text(0,1,info,va="top",wrap=True,bbox={"boxstyle":"round,pad=.6","facecolor":"#F4F7F8","edgecolor":"#AAB7BD"})
    eff=json.loads(item["modality_effects_json"]); ax2=fig.add_subplot(gs[0,1]); dd=[float(eff[m]["class_probability_delta"]) for m in MODALITIES]; ax2.bar([MOD_ZH[m] for m in MODALITIES],dd,color=[COLORS[m] for m in MODALITIES]); ax2.axhline(0,color="black",lw=.8); ax2.set_ylabel("目标类别概率变化"); ax2.set_title("逐模态遮挡效应"); style_ax(ax2)
    ax3=fig.add_subplot(gs[1,:]); xx=np.arange(len(labels)); ax3.bar(xx,d,color=["#C94C4C" if v<0 else "#4A9A62" for v in d]); ax3.axhline(0,color="black",lw=.8); ax3.set_xticks(xx,labels,rotation=45,ha="right",fontsize=8); ax3.set_xlabel("主要模态中的位置/文本片段（自动候选）"); ax3.set_ylabel("目标类别概率变化"); ax3.set_title("逐位置遮挡效应"); style_ax(ax3); save(fig,out,"q3_zh_example04_explanation_card")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); args=ap.parse_args(); root=Path(args.root); setup_font(); q1(root); q2(root); q3(root); print("中文图表导出完成")
if __name__=="__main__": main()
