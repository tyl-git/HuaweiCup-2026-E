"""Q3 explanations for the fixed three-seed temporal balanced-sqrt ensemble."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
OUT = ROOT / "03_Results" / "e" / "question-three" / "q3-balanced-ensemble-explanations-v1.1"
MODEL_ROOT = Q2 / "q2-temporal-balanced-sqrt-v1.0"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
CLASSES = ("Negative", "Neutral", "Positive")

def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write_json(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
def load(name, file):
    spec=importlib.util.spec_from_file_location(name, SRC/file); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def save_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def checkpoint_jobs(torch, temporal):
    jobs=[]
    for seed in SEEDS:
        folder=MODEL_ROOT/"temporal"/f"seed_{seed}"; ck=folder/"best.pt"
        meta=read_json(folder/"metrics.json"); saved=torch.load(ck,map_location="cpu",weights_only=True)
        if (saved.get("model")!="temporal" or saved.get("seed")!=seed or saved.get("epoch")!=meta["best_epoch"]
            or saved.get("normalization_sha256")!=sha(STATS)
            or json.dumps(saved["config"],sort_keys=True)!=json.dumps(meta["config"],sort_keys=True)):
            raise RuntimeError(f"Checkpoint metadata mismatch: {seed}")
        cfg=temporal.TemporalConfig(**saved["config"])
        model=temporal.build_model(saved["train_priors"],cfg,torch); model.load_state_dict(saved["model_state"],strict=True); model.eval()
        jobs.append({"seed":seed,"model":model,"checkpoint":str(ck),"checkpoint_sha256":sha(ck),"best_epoch":saved["epoch"],"config":saved["config"]})
    return jobs

def tensors(arrays, content, observed, torch, device):
    f={m:torch.as_tensor(np.asarray(arrays[m]).copy(),dtype=torch.float32,device=device) for m in MODALITIES}
    c=torch.as_tensor(np.asarray(content).copy(),dtype=torch.bool,device=device)
    o={m:torch.as_tensor(np.asarray(observed[m]).copy(),dtype=torch.bool,device=device) for m in MODALITIES}
    return f,c,o
def predict(job, f,c,o,torch):
    with torch.inference_mode(): logits,intensity,e=job["model"](f,c,o,return_evidence=True)
    return logits.cpu().numpy(), intensity.cpu().numpy(), {k:v.cpu().numpy() for k,v in e.items()}
def ens_predict(jobs,f,c,o,torch):
    rows=[predict(j,f,c,o,torch) for j in jobs]
    logits=np.mean([r[0] for r in rows],axis=0); intensity=np.mean([r[1] for r in rows],axis=0)
    prob=np.exp(logits-logits.max(axis=-1,keepdims=True)); prob/=prob.sum(axis=-1,keepdims=True)
    return logits,intensity,prob,rows
def tokenize(batch, tokenizer):
    out=[]
    for i,raw in enumerate(batch.raw_text):
        e=tokenizer(raw,padding="max_length",truncation=True,max_length=50,return_offsets_mapping=True)
        for k in ("input_ids","attention_mask","token_type_ids"):
            if not np.array_equal(e[k],getattr(batch,k)[i]): raise RuntimeError(f"Token mismatch: {batch.ids[i]}")
        words={}; p2w={}
        for p,wid in enumerate(e.word_ids()):
            if wid is None or not batch.content_mask[i,p]: continue
            a,b=e["offset_mapping"][p]; words.setdefault(wid,{"positions":[],"start":a,"end":b}); words[wid]["positions"].append(p); words[wid]["start"]=min(words[wid]["start"],a); words[wid]["end"]=max(words[wid]["end"],b)
        for wid,w in words.items(): w["text"]=raw[w["start"]:w["end"]]; [p2w.__setitem__(p,wid) for p in w["positions"]]
        out.append((words,p2w))
    return out
def one_explanation(jobs,f,c,o,torch,word_map):
    logits,y,p,rows=ens_predict(jobs,f,c,o,torch); cls=int(p[0].argmax()); base=float(p[0,cls]); intensity=float(y[0]); effects={}; positions=[]
    for mi,m in enumerate(MODALITIES):
        valid=np.flatnonzero(o[m][0].cpu().numpy()); deltas=[]
        for pos in valid:
            changed={x:v.clone() for x,v in o.items()}; changed[m][:,int(pos)]=False
            _,z,q,_=ens_predict(jobs,f,c,changed,torch)
            dp=base-float(q[0,cls]); dy=intensity-float(z[0]); deltas.append(dp)
            w=word_map[0].get(word_map[1].get(int(pos))) if word_map else None
            candidate=float(np.mean([r[2]["modality_gate"][0, int(pos), mi] * r[2]["time_weight"][0, int(pos)] for r in rows]))
            positions.append({"modality":m,"position_0based":int(pos),"class_probability_delta":dp,"intensity_delta":dy,"candidate_gate_time":candidate,"text_char_start":w["start"] if w else "","text_char_end":w["end"] if w else "","text_fragment":w["text"] if w else ""})
        if len(valid):
            changed={x:v.clone() for x,v in o.items()}; changed[m][:,valid]=False
            _,z,q,_=ens_predict(jobs,f,c,changed,torch)
            total=(base-float(q[0,cls]),intensity-float(z[0]))
        else:
            total=(0.,0.)
        effects[m]={"class_probability_delta":total[0],"intensity_delta":total[1],"candidate_gate_time":float(np.mean([r[2]["modality_gate"][0, valid, mi].dot(r[2]["time_weight"][0, valid]) for r in rows])) if len(valid) else 0.0,"observed_positions":int(len(valid))}
    s=sum(abs(v["class_probability_delta"]) for v in effects.values())
    for v in effects.values(): v["relative_influence"]=abs(v["class_probability_delta"])/s if s else 0.
    top_candidate=max((x["candidate_gate_time"] for x in positions),default=0.0)
    for x in positions: x["candidate_top1"] = bool(top_candidate > 0.0 and x["candidate_gate_time"] == top_candidate)
    positions.sort(key=lambda x:-abs(x["class_probability_delta"]))
    return {"predicted_class_id":cls,"predicted_class":CLASSES[cls],"predicted_intensity":intensity,"predicted_class_probability":base,"class_probabilities":p[0].tolist(),"primary_modality":max(effects,key=lambda x:effects[x]["relative_influence"]) if s else "none","modality_effects":effects,"top_evidence":positions[:5]},positions

def main():
    ap=argparse.ArgumentParser(description=__doc__); g=ap.add_mutually_exclusive_group(required=True); g.add_argument("--check",action="store_true"); g.add_argument("--run",action="store_true"); ap.add_argument("--device",choices=("cuda","cpu"),default="cuda"); a=ap.parse_args()
    import torch
    temporal=load("q3_balanced_temporal","2026-09-24_train-q2-temporal_v1.0.py"); loader=load("q3_balanced_loader","2026-09-24_q2_data_v1.0.py"); cache=load("q3_balanced_cache","2026-09-24_prepare-q2-text_v1.0.py"); robust=load("q3_balanced_robust","2026-09-24_train-q2-robust_v1.0.py"); norm=load("q3_balanced_norm","2026-09-24_q2-normalization_v1.0.py")
    official=loader.load_official(); special=loader.load_special_directory(loader.DEFAULT_SPECIAL[4],4)
    if len(official["valid"])!=728 or len(special)!=20: raise RuntimeError("Unexpected counts")
    scaler=norm.MultimodalStandardizer.load(STATS); jobs=checkpoint_jobs(torch,temporal)
    print(f"Preflight OK: 3 balanced-sqrt checkpoints, 728 valid, 20 attachment-4")
    if a.check: print("No inference or output written"); return
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}); from transformers import BertModel,BertTokenizerFast
    for job in jobs: job["model"].to(a.device)
    enc=BertModel.from_pretrained(str(cache.MODEL),local_files_only=True,attn_implementation="eager").to(a.device).float().eval(); enc.requires_grad_(False); tok=BertTokenizerFast.from_pretrained(str(cache.MODEL),local_files_only=True)
    text=np.empty((20,50,768),dtype=np.float32)
    with torch.inference_mode():
        for s in range(0,20,10):
            inp={k:torch.as_tensor(np.asarray(getattr(special,k)[s:s+10]).copy(),dtype=torch.long,device=a.device) for k in ("input_ids","attention_mask","token_type_ids")}; text[s:s+10]=enc(**inp).last_hidden_state.cpu().numpy()
    maps=tokenize(special,tok); arrays,observed=robust.normalize_once(special,{"text":text,"audio":special.audio,"vision":special.vision},scaler); f,c,o=tensors(arrays,special.content_mask,observed,torch,a.device)
    pred_rows=[]; exp_rows=[]; pos_rows=[]
    for i in range(20):
        onef={m:f[m][i:i+1] for m in MODALITIES}; oneo={m:o[m][i:i+1] for m in MODALITIES}; result,positions=one_explanation(jobs,onef,c[i:i+1],oneo,torch,maps[i]); fid=special.source_paths[i].stem; pred_rows.append({"file_id":fid,"sample_id":special.ids[i],"polarity":result["predicted_class"],"intensity":result["predicted_intensity"]}); exp_rows.append({"file_id":fid,"sample_id":special.ids[i],"polarity":result["predicted_class"],"intensity":result["predicted_intensity"],"predicted_class_probability":result["predicted_class_probability"],"primary_modality":result["primary_modality"],"modality_effects_json":json.dumps(result["modality_effects"]),"top_evidence_json":json.dumps(result["top_evidence"]),"raw_text":special.raw_text[i],"time_mapping_status":"aligned_position_only_seconds_unverified"}); pos_rows.extend({"file_id":fid,"sample_id":special.ids[i],**p} for p in positions); print(f"Attachment 4: {i+1}/20",flush=True)
    tmp=OUT.with_name(OUT.name+".partial"); tmp.mkdir(parents=True); save_csv(tmp/"attachment4_predictions.csv",pred_rows,("file_id","sample_id","polarity","intensity")); save_csv(tmp/"attachment4_explanations.csv",exp_rows,("file_id","sample_id","polarity","intensity","predicted_class_probability","primary_modality","modality_effects_json","top_evidence_json","raw_text","time_mapping_status")); save_csv(tmp/"attachment4_position_effects.csv",pos_rows,("file_id","sample_id","modality","position_0based","class_probability_delta","intensity_delta","candidate_gate_time","candidate_top1","text_char_start","text_char_end","text_fragment"))
    summary={"schema":"q3_balanced_ensemble_explanations_v1.1","seeds":list(SEEDS),"checkpoints":[{k:v for k,v in j.items() if k!="model"} for j in jobs],"normalization_sha256":sha(STATS),"aligned_only":True,"test_evaluated":False,"specialist_label_use":False,"explanation_rule":"arithmetic mean ensemble prediction followed by leave-one-position and leave-one-modality occlusion; gate/time are candidates only","text_occlusion_caveat":"feature-level occlusion after frozen BERT, not raw-text causal deletion","time_mapping_status":"automatic candidate only; exact seconds unverified","attachment4_source_sha256":dict(special.source_sha256),"output_sha256":{p.name:sha(p) for p in tmp.glob("*.csv")}}
    write_json(tmp/"summary.json",summary); tmp.rename(OUT); print(f"Q3 ensemble explanations saved: {OUT}")
if __name__=="__main__": main()
