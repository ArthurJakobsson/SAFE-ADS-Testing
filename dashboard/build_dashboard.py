"""Build a single self-contained HTML dashboard to debug the SAFE pipeline.

Scans the repo for whatever exists and embeds everything (images base64-inlined) into one
standalone `dashboard.html` that opens over file:// with no server or network.

Sections:
  - Status bar: served model / env / counts.
  - SAFE pipeline: per crash case -> input Sketch + Summary, extracted meta, DSL,
    and the synthesized nuPlan-style BEV.
  - Demo BEVs: the --demo synthetic BEVs (one per road type).

Usage:
    python dashboard/build_dashboard.py            # auto-discovers latest results
    python dashboard/build_dashboard.py --open     # also print the file:// URL
"""
import argparse
import base64
import glob
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK = os.path.join(REPO, "Framework")
BEV_OUT = os.path.join(FRAMEWORK, "ADS_Testing", "BEV_Synthesis", "output")
# CIREN -> UniScene pilot dataset (run_ciren_dataset.sh pilot). Lives off-repo; scanned if present.
PILOT_ROOT = os.environ.get("PILOT_ROOT", "/mnt/disk2/CIREN_dataset/CIREN_uniscene_v2")


def _b64(path, mime):
    try:
        with open(path, "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return None


def _img(path):
    if not path or not os.path.exists(path):
        return None
    p = path.lower()
    mime = "image/jpeg" if p.endswith((".jpg", ".jpeg")) else "image/gif" if p.endswith(".gif") else "image/png"
    return _b64(path, mime)


def _newest(pattern):
    hits = glob.glob(pattern, recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def collect():
    data = {"status": {}, "cases": [], "demos": [], "legend": [], "pilot": []}

    # --- env / status ---
    data["status"] = {
        "model": os.environ.get("SAFE_MODEL", "(set SAFE_MODEL)"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "(not set)"),
        "repo": REPO,
    }

    # --- latest meta + DSL pickles ---
    meta_pkl = _newest(os.path.join(FRAMEWORK, "Experiment_results", "Meta_Message_results_*", "meta_data_results.pkl"))
    dsl_pkl = _newest(os.path.join(FRAMEWORK, "Experiment_results", "DSL_results_*", "DSL_extraction_results.pkl"))
    meta_by_case, road_by_case = {}, {}
    if meta_pkl:
        rows = _load_pickle(meta_pkl) or []
        for r in rows:
            meta_by_case[str(r[-1])] = {"road_type": r[0], "num_cars": r[1], "direction": r[2]}
            road_by_case[str(r[-1])] = r[0]
    dsl_by_case = {}
    if dsl_pkl:
        for d in (_load_pickle(dsl_pkl) or []):
            dsl_by_case[str(d.get("Scenario"))] = d
    data["status"]["meta_pkl"] = meta_pkl
    data["status"]["dsl_pkl"] = dsl_pkl

    # --- BEV outputs: scan per-case sidecars directly so both demo and real BEVs always show
    # regardless of which bev_from_dsl run wrote manifest.json last ---
    bev_by_case = {}
    bev_manifest = os.path.join(BEV_OUT, "manifest.json")
    if os.path.exists(bev_manifest):
        try:
            with open(bev_manifest) as f:
                data["legend"] = json.load(f).get("legend", [])
        except Exception:
            pass
    for jp in glob.glob(os.path.join(BEV_OUT, "*.json")):
        if os.path.basename(jp) == "manifest.json":
            continue
        try:
            with open(jp) as f:
                info = json.load(f)
        except Exception:
            continue
        stem = os.path.splitext(os.path.basename(jp))[0]
        cid = str(info.get("case_id", stem))
        # prefer the animated GIF when present, else the representative PNG
        gif = info.get("gif")
        media = os.path.join(BEV_OUT, gif) if gif else os.path.join(BEV_OUT, info.get("png", stem + ".png"))
        if gif and not os.path.exists(media):
            media = os.path.join(BEV_OUT, info.get("png", stem + ".png"))
        bev_by_case[cid] = {"png": _img(media), "info": info, "animated": bool(gif)}

    # --- crash cases (input dataset) ---
    ds_dir = os.path.join(FRAMEWORK, "Crash_dataset")
    case_ids = sorted([d for d in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, d))]) \
        if os.path.isdir(ds_dir) else []
    for cid in case_ids:
        cdir = os.path.join(ds_dir, cid)
        summ = os.path.join(cdir, "Summary.txt")
        summary = open(summ, encoding="utf-8", errors="ignore").read() if os.path.exists(summ) else None
        bev = bev_by_case.get(cid)
        # UniScene-v2 styled views: scene-centred + one ego-centric view per vehicle (animated GIFs)
        uni_views = []
        scene_gif = os.path.join(BEV_OUT, f"{cid}_uniscene.gif")
        if os.path.exists(scene_gif):
            uni_views.append({"label": "scene", "src": _img(scene_gif)})
        for ego in sorted(glob.glob(os.path.join(BEV_OUT, f"{cid}_ego_*.gif"))):
            label = os.path.basename(ego)[len(cid) + len("_ego_"):-len(".gif")]
            uni_views.append({"label": label, "src": _img(ego)})
        data["cases"].append({
            "case_id": cid,
            "sketch": _img(os.path.join(cdir, "Sketch.jpg")),
            "summary": summary,
            "meta": meta_by_case.get(cid),
            "dsl": dsl_by_case.get(cid),
            "bev_png": bev["png"] if bev else None,
            "bev_info": bev["info"] if bev else None,
            "uni_views": uni_views,
        })

    # --- demo BEVs (case ids starting with demo_) ---
    for cid, bev in sorted(bev_by_case.items()):
        if cid.startswith("demo_"):
            data["demos"].append({"case_id": cid, "bev_png": bev["png"], "bev_info": bev["info"]})

    # --- CIREN->UniScene pilot: per-case BEV + per-object height tag ---
    for sp in sorted(glob.glob(os.path.join(PILOT_ROOT, "samples", "*", "*", "pkl_records.pkl"))):
        rec = _load_pickle(sp)
        if not rec:
            continue
        sd = os.path.dirname(sp)
        names = rec.get("nuscenes_names") or []
        models = rec.get("models") or []
        heights = rec.get("heights") or []
        n = max(len(names), len(models), len(heights))
        objs = [{"label": f"V{i + 1}",
                 "name": (models[i] if i < len(models) and models[i] else
                          (names[i] if i < len(names) else "?")),
                 "height": round(float(heights[i]), 2) if i < len(heights) else None}
                for i in range(n)]
        scene_gif = _newest(os.path.join(sd, "scene", "*_uniscene.gif"))
        data["pilot"].append({
            "case_id": rec.get("case_id", os.path.basename(sd)),
            "road_type": rec.get("road_type"),
            "bev": _img(scene_gif),
            "objects": objs,
        })

    return data


HEAD = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>SAFE — crash → DSL → BEV dashboard</title>
<style>
:root{--bg:#16181d;--panel:#1e2128;--panel2:#252933;--line:#333a45;--fg:#e6e9ef;--mut:#9aa4b2;--acc:#5cc8ff;--ok:#4ade80;--warn:#fbbf24;--err:#f87171;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
header{position:sticky;top:0;z-index:10;background:linear-gradient(180deg,#1b1e25,#16181d);border-bottom:1px solid var(--line);padding:14px 20px}
h1{margin:0;font-size:18px;letter-spacing:.3px}
h1 small{color:var(--mut);font-weight:400;font-size:12px;margin-left:8px}
.status{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:8px;color:var(--mut);font-size:12px}
.status b{color:var(--fg)} .status code{color:var(--acc)}
nav{display:flex;gap:6px;margin-top:10px}
nav button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px}
nav button.on{background:var(--acc);color:#06222e;border-color:var(--acc);font-weight:600}
main{padding:18px 20px;max-width:1500px;margin:0 auto}
section{display:none} section.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card h3{margin:0;padding:10px 12px;background:var(--panel2);border-bottom:1px solid var(--line);font-size:14px;display:flex;justify-content:space-between;align-items:center}
.tag{font-size:11px;padding:2px 8px;border-radius:20px;background:#2b3340;color:var(--mut)}
.tag.ok{background:#10331f;color:var(--ok)} .tag.warn{background:#33270f;color:var(--warn)} .tag.err{background:#331717;color:var(--err)}
.row{display:flex;gap:10px;padding:12px}
.col{flex:1;min-width:0}
.imgwrap{background:#0d0f13;border:1px solid var(--line);border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;min-height:150px}
.imgwrap img{width:100%;display:block;image-rendering:pixelated}
.lbl{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.4px;margin:0 0 4px}
.summary{white-space:pre-wrap;font-size:12.5px;max-height:150px;overflow:auto;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px;color:#cfd6e0}
pre.dsl{margin:0;font-size:11.5px;max-height:240px;overflow:auto;background:#0d0f13;border:1px solid var(--line);border-radius:8px;padding:8px;color:#bfe6ff}
.meta{display:flex;gap:6px;flex-wrap:wrap;padding:0 12px 12px}
.chip{font-size:11px;background:#222a36;border:1px solid var(--line);border-radius:6px;padding:3px 8px;color:#cdd6e2}
.empty{color:var(--mut);font-style:italic;padding:8px}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:14px}
.legend span{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;color:var(--mut)}
.legend i{width:12px;height:12px;border-radius:3px;display:inline-block;border:1px solid #0006}
.note{color:var(--mut);font-size:12px;margin:0 0 14px}
</style></head><body>"""

BODY_JS = r"""
<script>
const $=(t,c,a={})=>{const e=document.createElement(t);if(c)e.className=c;for(const k in a)e[k]=a[k];return e};
function imgCol(lbl,src){const c=$('div','col');c.append(Object.assign($('p','lbl'),{textContent:lbl}));
 const w=$('div','imgwrap'); if(src){w.append($('img','',{src}))} else {w.append(Object.assign($('div','empty'),{textContent:'—'}))} c.append(w); return c;}
function textCol(lbl,txt,cls){const c=$('div','col');c.append(Object.assign($('p','lbl'),{textContent:lbl}));
 if(txt){const d=$('div',cls);d.textContent=txt;c.append(d)} else c.append(Object.assign($('div','empty'),{textContent:'not available'})); return c;}

function caseCard(c){
 const card=$('div','card');
 const h=$('h3'); h.append(Object.assign($('span'),{textContent:'Case '+c.case_id}));
 const t=$('span','tag '+(c.dsl?'ok':(c.bev_png?'warn':''))); t.textContent=c.dsl?'DSL ✓':(c.bev_png?'BEV only':'input only'); h.append(t);
 card.append(h);
 const r1=$('div','row'); r1.append(imgCol('input sketch',c.sketch)); r1.append(imgCol('synth BEV (nuPlan-style)',c.bev_png)); card.append(r1);
 if(c.uni_views&&c.uni_views.length){
   const col=$('div','col'); col.append(Object.assign($('p','lbl'),{textContent:'UniScene-v2 BEV — pick ego'}));
   const w=$('div','imgwrap'); const im=$('img','',{src:c.uni_views[0].src}); w.append(im); col.append(w);
   const bar=$('div','meta');
   const setOn=b=>{bar.querySelectorAll('.chip').forEach(x=>{x.style.background='';x.style.color='';});
     b.style.background='var(--acc)';b.style.color='#06222e';};
   c.uni_views.forEach((v,idx)=>{const b=$('span','chip');
     b.textContent=(v.label==='scene'?'scene':v.label+' ego'); b.style.cursor='pointer';
     b.onclick=()=>{im.src=v.src; setOn(b);}; if(idx===0)setOn(b); bar.append(b);});
   const r=$('div','row'); r.append(col); card.append(r); card.append(bar);}
 if(c.meta){const m=$('div','meta');
   m.append(Object.assign($('span','chip'),{textContent:'road: '+c.meta.road_type}));
   m.append(Object.assign($('span','chip'),{textContent:'cars: '+c.meta.num_cars}));
   m.append(Object.assign($('span','chip'),{textContent:'dir: '+c.meta.direction})); card.append(m);}
 if(c.dsl&&c.dsl.Conflict){const cf=c.dsl.Conflict;const m2=$('div','meta');
   const it=cf.impact_type||cf.impact||'';const af=cf.at_fault_vehicle||cf.at_fault||'';
   if(it)m2.append(Object.assign($('span','chip'),{textContent:'impact: '+it}));
   if(af)m2.append(Object.assign($('span','chip'),{textContent:'at-fault: '+af}));
   card.append(m2);}
 const r2=$('div','row'); r2.append(textCol('crash summary',c.summary,'summary')); card.append(r2);
 if(c.dsl){const r3=$('div','row'); const col=$('div','col'); col.append(Object.assign($('p','lbl'),{textContent:'extracted DSL'}));
   const pre=$('pre','dsl'); pre.textContent=JSON.stringify(c.dsl,null,2); col.append(pre); r3.append(col); card.append(r3);}
 return card;
}
function demoCard(d){const card=$('div','card');const h=$('h3');h.append(Object.assign($('span'),{textContent:d.case_id}));
 const t=$('span','tag ok');t.textContent=(d.bev_info&&d.bev_info.road_type)||'';h.append(t);card.append(h);
 const r=$('div','row');r.append(imgCol('synthesized BEV',d.bev_png));card.append(r);
 if(d.bev_info){const m=$('div','meta');(d.bev_info.agents||[]).forEach(a=>m.append(Object.assign($('span','chip'),{textContent:a.label+(a.is_ego?' (ego)':'')+' @'+a.yaw_deg+'°'})));
   m.append(Object.assign($('span','chip'),{textContent:'lanes: '+d.bev_info.n_lanes}));card.append(m);}
 return card;}
function pilotCard(p){const card=$('div','card');const h=$('h3');
 h.append(Object.assign($('span'),{textContent:p.case_id}));
 if(p.road_type){const t=$('span','tag ok');t.textContent=p.road_type;h.append(t);}card.append(h);
 const r=$('div','row');r.append(imgCol('UniScene BEV',p.bev));card.append(r);
 const m=$('div','meta');(p.objects||[]).forEach(o=>m.append(Object.assign($('span','chip'),
   {textContent:o.label+' · '+o.name+' · '+(o.height!=null?o.height+' m':'?')})));card.append(m);
 return card;}
function legend(items){if(!items||!items.length)return null;const l=$('div','legend');
 items.forEach(([n,c])=>{const s=$('span');const i=$('i');i.style.background=c;s.append(i);s.append(document.createTextNode(n));l.append(s)});return l;}

(function(){
 const st=DATA.status;
 document.getElementById('st').innerHTML=
   '<span>model <code>'+st.model+'</code></span>'+
   '<span>endpoint <code>'+st.base_url+'</code></span>'+
   '<span><b>'+DATA.cases.length+'</b> crash cases</span>'+
   '<span><b>'+DATA.cases.filter(c=>c.dsl).length+'</b> with DSL</span>'+
   '<span><b>'+DATA.cases.filter(c=>c.bev_png).length+'</b> with BEV</span>'+
   '<span><b>'+DATA.demos.length+'</b> demo BEVs</span>'+
   (DATA.pilot&&DATA.pilot.length?'<span><b>'+DATA.pilot.length+'</b> pilot cases (height-tagged)</span>':'');

 const A=document.getElementById('secA');
 const lg=legend(DATA.legend); if(lg)A.append(lg);
 A.append(Object.assign($('p','note'),{textContent:'Each card: real crash sketch (input) → synthesized nuPlan-style BEV (18-channel raster) → meta → summary → extracted DSL. Run the pipeline + bev_from_dsl, then rebuild to populate DSL/BEV.'}));
 const ga=$('div','grid'); DATA.cases.forEach(c=>ga.append(caseCard(c))); A.append(ga);
 if(!DATA.cases.length)A.append(Object.assign($('div','empty'),{textContent:'No crash cases found.'}));

 const D=document.getElementById('secD');
 const lg2=legend(DATA.legend); if(lg2)D.append(lg2);
 D.append(Object.assign($('p','note'),{textContent:'Synthetic BEVs (no LLM) — one per road type, to validate the rasterizer.'}));
 const gd=$('div','grid'); DATA.demos.forEach(d=>gd.append(demoCard(d))); D.append(gd);
 if(!DATA.demos.length)D.append(Object.assign($('div','empty'),{textContent:'No demo BEVs. Run: bev_from_dsl.py --demo'}));

 const P=document.getElementById('secP');
 if(P){
  P.append(Object.assign($('p','note'),{textContent:'CIREN→UniScene pilot — per-object roof height (m) tagged on each moving object.'}));
  const gp=$('div','grid'); DATA.pilot.forEach(p=>gp.append(pilotCard(p))); P.append(gp);
  if(!DATA.pilot.length)P.append(Object.assign($('div','empty'),{textContent:'No pilot data.'}));
 }

 const secs={A:'secA',D:'secD'}; if(P)secs.P='secP';
 document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
   document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
   for(const k in secs)document.getElementById(secs[k]).classList.remove('on');
   document.getElementById(secs[b.dataset.s]).classList.add('on');});
})();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "dashboard", "dashboard.html"))
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    data = collect()
    buttons = ['<button class=on data-s=A>SAFE pipeline</button>',
               '<button data-s=D>Demo BEVs</button>']
    sections = ['<section class=on id=secA></section>', '<section id=secD></section>']
    if data["pilot"]:
        buttons.append('<button data-s=P>CIREN pilot · heights</button>')
        sections.append('<section id=secP></section>')
    nav = '<nav>' + ''.join(buttons) + '</nav>'
    header = ('<header><h1>SAFE <small>crash → DSL → nuPlan-style BEV</small></h1>'
              '<div class=status id=st></div>' + nav + '</header>')
    main_html = '<main>' + ''.join(sections) + '</main>'
    html = HEAD + header + main_html + \
        "<script>const DATA=" + json.dumps(data) + ";</script>" + BODY_JS

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[dashboard] wrote {args.out} ({size_mb:.2f} MB)")
    print(f"[dashboard] cases={len(data['cases'])} demos={len(data['demos'])}")
    if args.open:
        print(f"[dashboard] open -> file://{args.out}")


if __name__ == "__main__":
    main()
