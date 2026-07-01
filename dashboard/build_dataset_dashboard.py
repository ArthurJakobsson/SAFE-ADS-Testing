"""Interactive viewer for the CIREN->UniScene dataset.

For each case we recompute the SAME dense rollout the BEV animation uses (from the cached DSL), so
the 3D and the 2D BEV have the *same number of timesteps*. Per timestep we render the UniScene
scene + ego BEV and extract the exact per-object `gt_boxes` (l x w x h in metres). The viewer:
  - case selector,
  - timestep buttons that step the 3D AND the BEV images together,
  - a Three.js 3D scene of the exact boxes (drag = rotate, scroll = zoom) on the drivable footprint,
  - a per-object table (model, class, roof height, L x W x H).

Writes everything under `dashboard/dataset_assets/` (the downloadable folder): the Three.js libs in
`lib/`, and per case the per-frame BEV PNGs + `data.json`. The HTML inlines the small geometry and
references libs/images by RELATIVE path -> works over file:// with no server/network.

    python dashboard/build_dataset_dashboard.py [--limit N]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Framework", "ADS_Testing", "BEV_Synthesis"))
import bev_from_dsl as B          # noqa: E402
import uniscene_bev as U          # noqa: E402
import uniscene_export as UE      # noqa: E402

OUT_ROOT = os.environ.get("PILOT_ROOT", "/mnt/disk2/CIREN_dataset/CIREN_uniscene_v2")
ASSETS = os.path.join(REPO, "dashboard", "dataset_assets")
XY_RES = 0.25
ROAD_DS = 8
BEV_PX = 360


def road_cells(bev18, ds=ROAD_DS):
    drivable = np.any(bev18[0:8].astype(bool), axis=0)
    H, W = drivable.shape
    H2, W2 = (H // ds) * ds, (W // ds) * ds
    d = drivable[:H2, :W2].reshape(H2 // ds, ds, W2 // ds, ds).any(axis=(1, 3))
    ii, jj = np.nonzero(d)
    x = np.round((jj * ds + ds / 2 - W / 2) * XY_RES, 1)
    y = np.round((ii * ds + ds / 2 - H / 2) * XY_RES, 1)
    return [[float(a), float(b)] for a, b in zip(x, y)], ds * XY_RES


def collect(limit=0):
    cases = []
    samples = sorted(glob.glob(os.path.join(OUT_ROOT, "samples", "*", "*", "pkl_records.pkl")))
    if limit:
        samples = samples[:limit]
    for sp in samples:
        rec = pickle.load(open(sp, "rb"))
        sd = os.path.dirname(sp)
        cid = rec.get("case_id")
        road_type = rec.get("road_type")
        dslp = os.path.join(sd, "safe", "dsl.json")
        if not os.path.exists(dslp):
            continue
        dsl = json.load(open(dslp))

        # recompute the dense rollout (same source as the BEV animation)
        scene, _seq_out, _imp_out, seq_fine, imp_fine = UE.simulate_scene(
            dsl, road_type, frames=5, fps=2, step=4)
        N = len(seq_fine)
        names = scene.notes.get("nuscenes_names") or []
        models = scene.notes.get("models") or []
        heights = scene.notes.get("heights") or []

        adir = os.path.join(ASSETS, cid)
        os.makedirs(adir, exist_ok=True)
        base, affine = B.rasterize_static(scene, U.CANVAS, U.PATCH)
        cells, res = road_cells(base)                         # WORLD drivable (fixed road)
        dfl = UE.DEFAULT_VEH_HEIGHT
        frames = []
        for f in range(N):
            bevf = base.copy()
            ego = B.draw_agents(bevf, seq_fine[f], affine)
            Image.fromarray(U.render_uniscene(bevf, BEV_PX, ego, rotate90=True)).save(
                os.path.join(adir, f"scene_{f:02d}.png"))
            eb, ep = U._rasterize_ego(scene, seq_fine[f], 0)
            Image.fromarray(U.render_uniscene(eb, BEV_PX, ep, rotate90=False)).save(
                os.path.join(adir, f"ego_{f:02d}.png"))
            # WORLD-frame boxes: the road is fixed and every car (incl. ego) drives its trajectory,
            # converging at the collision. dx = length (along heading), dy = width.
            fr = []
            for i, st in enumerate(seq_fine[f]):
                cx, cy, yaw, dx, dy, _e = st
                h = heights[i] if i < len(heights) else dfl
                fr.append([round(float(cx), 2), round(float(cy), 2), round(h / 2, 2),
                           round(float(dx), 2), round(float(dy), 2), round(h, 2),
                           round(float(yaw), 3), 1 if i == 0 else 0])
            frames.append(fr)

        rep = imp_fine if imp_fine is not None else N // 2
        objs = []
        for i, st in enumerate(seq_fine[0]):
            dx, dy = float(st[3]), float(st[4])
            h = heights[i] if i < len(heights) else dfl
            objs.append({"label": f"V{i + 1}", "model": models[i] if i < len(models) else "",
                         "name": names[i] if i < len(names) else "?",
                         "h": round(h, 2), "dims": f"{dx:.1f}×{dy:.1f}×{h:.1f}"})

        case = {"id": cid, "road_type": road_type, "n_veh": len(scene.agents), "n_frames": N,
                "rep": int(rep), "res": round(res, 2), "road": cells, "frames": frames, "objects": objs}
        json.dump(case, open(os.path.join(adir, "data.json"), "w"))
        cases.append(case)
        print(f"[dataset] {cid:20s} veh={len(scene.agents)} frames={N} road_cells={len(cells)} impact={imp_fine}")
    return cases


def write_readme():
    open(os.path.join(ASSETS, "README.md"), "w").write(
        "# SAFE -> UniScene dataset viewer assets\n\n"
        "- `lib/` : Three.js + OrbitControls.\n"
        "- `<case>/scene_NN.png`, `ego_NN.png` : per-timestep UniScene BEV (scene + ego V1).\n"
        "- `<case>/data.json` : per-frame 3D boxes [x,y,z,l,w,h,yaw,is_ego], drivable cells, objects.\n\n"
        "Open `../dashboard_dataset.html` (this folder must sit beside it). No server/network.\n")


HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>SAFE — UniScene dataset (3D)</title>
<style>
:root{--bg:#16181d;--panel:#1e2128;--panel2:#252933;--line:#333a45;--fg:#e6e9ef;--mut:#9aa4b2;--acc:#5cc8ff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
header{background:linear-gradient(180deg,#1b1e25,#16181d);border-bottom:1px solid var(--line);padding:12px 18px}
h1{margin:0;font-size:17px}h1 small{color:var(--mut);font-weight:400;font-size:12px;margin-left:8px}
.bar{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center;margin-top:8px}
select,button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;font-size:13px}
button.on{background:var(--acc);color:#06222e;border-color:var(--acc);font-weight:600}
.lbl{color:var(--mut);font-size:12px;margin-left:6px}
main{display:flex;gap:14px;padding:14px 18px;flex-wrap:wrap}
#view{flex:2;min-width:440px;height:64vh;background:#0d0f13;border:1px solid var(--line);border-radius:12px;overflow:hidden;position:relative}
#hint{position:absolute;left:10px;bottom:8px;color:var(--mut);font-size:11px}
.side{flex:1;min-width:300px;display:flex;flex-direction:column;gap:10px}
.imw{background:#0d0f13;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.imw img{width:100%;display:block;image-rendering:pixelated}
.cap{font-size:11px;color:var(--mut);padding:4px 8px;text-transform:uppercase;letter-spacing:.4px}
table{width:100%;border-collapse:collapse;font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}th{color:var(--mut)}td.h{color:#ffd24a;font-weight:600}
.legend{color:var(--mut);font-size:12px}.legend i{width:11px;height:11px;border-radius:3px;display:inline-block;border:1px solid #0006;vertical-align:-1px;margin:0 4px}
</style></head><body>
<header><h1>SAFE <small>CIREN → UniScene dataset · interactive 3D world scene (fixed road, cars drive → collide)</small></h1>
<div class=bar>
  <span class=lbl>case</span><select id=caseSel></select>
  <span class=lbl>timestep</span><span id=frames></span>
  <button id=play>▶ play</button><button id=reset>reset view</button>
  <span class=legend><i style="background:#39e07a"></i>ego &nbsp;<i style="background:#ff9e00"></i>vehicle (exact L×W×H) &nbsp;<i style="background:#4a4f5a"></i>road</span>
</div></header>
<main>
  <div id=view><div id=hint>drag = rotate · scroll = zoom · right-drag = pan</div></div>
  <div class=side>
    <div class=imw><div class=cap id=cap></div><img id=scene></div>
    <div class=imw><div class=cap>ego BEV (V1) — same timestep</div><img id=ego></div>
    <table id=tbl></table>
  </div>
</main>
<script src="dataset_assets/lib/three.min.js"></script>
<script src="dataset_assets/lib/OrbitControls.js"></script>
<script>const DATA=__DATA__;</script>
<script>
let renderer,scene,camera,controls,road,ground,boxes=[],ci=0,fi=0,timer=null;
const view=document.getElementById('view');
function init(){
 const w=view.clientWidth,h=view.clientHeight;
 renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(w,h);renderer.setPixelRatio(devicePixelRatio);
 view.appendChild(renderer.domElement);
 scene=new THREE.Scene();scene.background=new THREE.Color(0x0d0f13);
 camera=new THREE.PerspectiveCamera(50,w/h,0.5,4000);camera.up.set(0,0,1);
 controls=new THREE.OrbitControls(camera,renderer.domElement);
 scene.add(new THREE.AmbientLight(0xffffff,0.8));
 const dl=new THREE.DirectionalLight(0xffffff,0.5);dl.position.set(20,-30,80);scene.add(dl);
 ground=new THREE.Mesh(new THREE.PlaneGeometry(140,140),new THREE.MeshBasicMaterial({color:0x12141a}));scene.add(ground);
 const grid=new THREE.GridHelper(120,48,0x2a3038,0x1f242b);grid.rotation.x=Math.PI/2;scene.add(grid);
 window.addEventListener('resize',onResize);
 (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();
}
function onResize(){const w=view.clientWidth,h=view.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}
function resetView(){camera.position.set(32,-64,48);controls.target.set(0,0,2);controls.update();}
function buildRoad(c){
 if(road)scene.remove(road);if(!c.road.length){road=null;return;}
 const geo=new THREE.BoxGeometry(c.res,c.res,0.15),mat=new THREE.MeshLambertMaterial({color:0x4a4f5a});
 road=new THREE.InstancedMesh(geo,mat,c.road.length);const m=new THREE.Matrix4();
 c.road.forEach((p,i)=>{m.makeTranslation(p[0],p[1],0.075);road.setMatrixAt(i,m);});
 road.instanceMatrix.needsUpdate=true;scene.add(road);
}
function buildFrame(c,f){
 boxes.forEach(b=>scene.remove(b));boxes=[];
 c.frames[f].forEach(b=>{const[x,y,z,l,w,h,yaw,ego]=b;
  const geo=new THREE.BoxGeometry(l,w,h);
  const mesh=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({color:ego?0x39e07a:0xff9e00}));
  mesh.position.set(x,y,z);mesh.rotation.z=yaw;
  mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:0x0a0a0a})));
  scene.add(mesh);boxes.push(mesh);});
}
function setFrame(f){fi=f;const c=DATA[ci];buildFrame(c,f);
 const n=String(f).padStart(2,'0');
 document.getElementById('scene').src='dataset_assets/'+c.id+'/scene_'+n+'.png';
 document.getElementById('ego').src='dataset_assets/'+c.id+'/ego_'+n+'.png';
 [...document.getElementById('frames').children].forEach((b,k)=>b.classList.toggle('on',k===f));}
function frameButtons(c){const wrap=document.getElementById('frames');wrap.innerHTML='';
 for(let f=0;f<c.n_frames;f++){const b=document.createElement('button');b.textContent=f;b.onclick=()=>{stop();setFrame(f);};wrap.appendChild(b);}}
function stop(){if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ play';}}
function play(){const pb=document.getElementById('play');
 if(timer){stop();return;}pb.textContent='⏸ pause';
 timer=setInterval(()=>{setFrame((fi+1)%DATA[ci].n_frames);},250);}
function showCase(i){ci=i;stop();const c=DATA[i];
 document.getElementById('cap').textContent='scene BEV · '+(c.road_type||'')+' · '+c.n_veh+' veh · '+c.n_frames+' frames';
 document.getElementById('tbl').innerHTML='<tr><th>obj</th><th>model</th><th>class</th><th>height</th><th>L×W×H</th></tr>'+
   c.objects.map(o=>'<tr><td>'+o.label+'</td><td>'+o.model+'</td><td>'+o.name+'</td><td class=h>'+o.h+' m</td><td>'+o.dims+'</td></tr>').join('');
 buildRoad(c);frameButtons(c);setFrame(0);resetView();}
init();
const sel=document.getElementById('caseSel');
DATA.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.id+'  ('+(c.road_type||'')+')';sel.appendChild(o);});
sel.onchange=()=>showCase(+sel.value);
document.getElementById('reset').onclick=resetView;document.getElementById('play').onclick=play;
showCase(0);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "dashboard", "dashboard_dataset.html"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    cases = collect(args.limit)
    write_readme()
    open(args.out, "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(cases)))
    print(f"[dataset] wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB) + assets -> {ASSETS}")


if __name__ == "__main__":
    main()
