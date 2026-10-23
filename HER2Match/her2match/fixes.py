"""Controlled HER2match staining fixes; HER2match evaluation only."""
from pathlib import Path
import os,sys,json,csv,random,hashlib,shutil,time,fcntl,traceback,gc,importlib.util
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','4')
PACKAGE=Path(__file__).resolve().parent; ROOT=PACKAGE.parent
DATA_ROOT=Path(os.environ.get('HER2_DATA_ROOT',ROOT/'data/her2match')).expanduser().resolve()
OUT=Path(os.environ.get('HER2_FIX_OUT',ROOT/'outputs/stain-fixes')).resolve()
ORIGINAL=Path(os.environ.get('HER2_BASELINE_OUT',ROOT/'outputs/baseline')).resolve()
sys.path.insert(0,str(PACKAGE))
spec=importlib.util.spec_from_file_location('fix_base',PACKAGE/'baseline.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
import numpy as np
import torch
from PIL import Image
from skimage.color import hed_from_rgb
from stain_error_analysis import read_rgb,tissue_mask,_channels,analyze_pair
NEW=('direct_cfm_stain_sampling','direct_cfm_stain_loss');OLD=('noise_cfm','direct_cfm','direct_cfm_lpips');ALL=OLD+NEW
CLASSES=('0','1+','2+','3+')
b.OUT=OUT;b.NAMES=ALL
b.PROTOCOL=dict(b.PROTOCOL,memory_cap_GiB=14,selection='252 HER2match test pairs;63 per slide score',test_dataset='HER2match')
BASE_DATASET=b.dataset;BASE_LOSS=b.losses
ACTIVE=None
CAL=json.loads((PACKAGE/'manifests/stain_calibration.json').read_text());THRESHOLD=CAL['dab_threshold']

def read_records(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write_records(p,rs):
 value=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs)
 if p.exists():assert p.read_text()==value,f'Changed manifest: {p}'
 else:p.write_text(value)
def run_dir(name):return (ORIGINAL if name in OLD else OUT)/'runs'/name

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'manifests').mkdir(exist_ok=True)
 allrs=read_records(ORIGINAL/'manifests/HER2match.jsonl')
 write_records(OUT/'manifests/HER2match.jsonl',allrs)
 prior=read_records(PACKAGE/'manifests/previous100.jsonl');chosen=[]
 for score in CLASSES:
  pool=[r for r in allrs if r['split']=='test' and r['her2_score']==score]
  groups=sorted({r['group'] for r in pool})
  for j,g in enumerate(groups):
   keep=[r for r in prior if r['group']==g and r['her2_score']==score]
   ids={r['id'] for r in keep}
   candidates=sorted([r for r in pool if r['group']==g and r['id'] not in ids],key=lambda r:hashlib.sha256(('42|extension252|'+r['id']).encode()).hexdigest())
   chosen.extend(keep+candidates[:63//len(groups)+(j<63%len(groups))-len(keep)])
 assert len(chosen)==252 and len({r['id'] for r in chosen})==252
 assert {r['id'] for r in prior}<={r['id'] for r in chosen}
 write_records(OUT/'manifests/test252.jsonl',chosen)
 for filename in ('initial_model.pt',):
  p=OUT/filename
  if not p.exists():p.symlink_to(ORIGINAL/filename)
 shutil.copy2(ORIGINAL/'manifests/validation_subset.json',OUT/'manifests/validation_subset.json')
 b.dump(OUT/'selection.json',dict(count=252,per_slide_score=63,seed=42,includes_previous100=True,selection='Keep previous100; deterministic hash-ranked extension within slide. Score0:32+31 across two slides.',label_scope='Slide-level labels; not patch clinical diagnoses',test_reuse='Exploratory follow-up on already inspected test cohort, not independent confirmation.',patient_ids='Unavailable; official slide-disjoint split',subset_sha256=b.sha(OUT/'manifests/test252.jsonl')))
 b.dump(OUT/'fix_protocol.json',dict(updates=20000,seed=42,effective_batch=8,microbatch=2,accumulation=4,initial_weights_sha256=b.sha(OUT/'initial_model.pt'),crop=512,solver='Heun16',checkpoint_selection='Same original16 center512 validation images, highest EMA SSIM; expanded48 validation staining diagnostics at final selected checkpoint.',sampling='3x3 grid of native512 crops at offsets0,256,512; tissue>=20%; low<=1%,medium1-10%,high>10% DAB coverage; 3/2/3 slots per effective batch; uniform slide then tile then crop within bin; fail explicitly if an eligible bin is empty.',loss='FM + 0.1*balanced stained/unstained absolute DAB error normalized by fixed0.1 DAB scale. Mean per image over nonempty tissue regions. Differentiable HED with auxiliary RGB clamp and straight-through nonnegative DAB projection; no velocity clamp.',loss_weight_policy='Fixed a priori lambda0.1; no test tuning or weight search',dab_threshold=THRESHOLD,validation_calibration_sha256=b.sha(PACKAGE/'manifests/stain_calibration.json'),memory_cap_GiB=14))
 return allrs

def scan_one(r):
 im=read_rgb(DATA_ROOT/r['ihc']);mask=tissue_mask(im);dab=_channels(im)[1]
 # Integral images make all nine native crop measurements exact.
 def integral(a):return np.pad(a.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
 m=integral(mask);d=integral(mask&(dab>THRESHOLD))
 def count(a,y,x):return int(a[y+512,x+512]-a[y,x+512]-a[y+512,x]+a[y,x])
 rows=[]
 for y in (0,256,512):
  for x in (0,256,512):
   n=count(m,y,x);coverage=count(d,y,x)/max(n,1)
   rows.append(dict(id=r['id'],group=r['group'],top=y,left=x,tissue_fraction=n/512**2,dab_coverage=coverage,bin=0 if coverage<=.01 else (1 if coverage<=.1 else 2)))
 return rows

def scan():
 from concurrent.futures import ProcessPoolExecutor
 rs=[r for r in prepare() if r['split']=='train'];p=OUT/'manifests/training_crops.csv'
 if p.exists():return
 tmp=p.with_suffix('.partial.csv');done=set();rows=[]
 if tmp.exists():
  with tmp.open() as f:rows=list(csv.DictReader(f));done={r['id'] for r in rows}
 with ProcessPoolExecutor(max_workers=6) as pool:
  for j,part in enumerate(pool.map(scan_one,[r for r in rs if r['id'] not in done],chunksize=8)):
   rows.extend(part)
   if (j+1)%250==0:
    b.csv_save(tmp,rows);b.status(phase='indexing_training_crops',tiles=len(rows)//9,total=len(rs))
 b.csv_save(p,rows)
 if tmp.exists():tmp.unlink()
 b.dump(OUT/'manifests/crop_index_identity.json',dict(source_sha256=b.sha(OUT/'manifests/HER2match.jsonl'),index_sha256=b.sha(p),tiles=len(rs),crops=len(rows),threshold=THRESHOLD))

class StainDataset(torch.utils.data.Dataset):
 def __init__(self):
  self.root=DATA_ROOT;self.records=[r for r in read_records(OUT/'manifests/HER2match.jsonl') if r['split']=='train'];self.lookup={r['id']:r for r in self.records}
  with (OUT/'manifests/training_crops.csv').open() as f:self.crops=list(csv.DictReader(f))
  self.pools={i:{} for i in range(3)}
  for j,r in enumerate(self.crops):
   if float(r['tissue_fraction'])<.2:continue
   self.pools[int(r['bin'])].setdefault(r['group'],{}).setdefault(r['id'],[]).append(j)
  assert all(self.pools.values()),'Missing eligible staining stratum'
 def __len__(self):return len(self.records)
 def __getitem__(self,key):
  index,seed=key;cr=self.crops[index];r=self.lookup[cr['id']];top,left=int(cr['top']),int(cr['left']);rng=random.Random(seed)
  fx,fy,k=rng.random()<.5,rng.random()<.5,rng.randrange(4);result={'id':r['id'],'group':r['group']}
  for field in ('he','ihc'):
   a=read_rgb(self.root/r[field])[top:top+512,left:left+512]
   a=np.rot90(a[::(-1 if fy else 1),::(-1 if fx else 1)],k)
   result[field]=torch.from_numpy(a.copy()).permute(2,0,1).float()/127.5-1
  return result

class FixBatches(b.TrainingBatches):
 def __iter__(self):
  if ACTIVE!=NEW[0]:yield from super().__iter__();return
  ds=StainDataset();pattern=(0,1,2,0,2,1,0,2);batch=self.start_batch
  while True:
   keys=[]
   for j in range(self.batch_size):
    index=batch*self.batch_size+j;stratum=pattern[index%8];rng=random.Random(b.stable_seed(42,f'balanced-crop:{index}'))
    groups=ds.pools[stratum];g=rng.choice(sorted(groups));tile=rng.choice(sorted(groups[g]));crop=rng.choice(groups[g][tile]);keys.append((crop,b.stable_seed(42,f'crop-augment:{index}')))
   yield keys;batch+=1

def dataset(which,split,crop=None,augment=False):
 assert which=='HER2match','HER2match only'
 if split=='train' and ACTIVE==NEW[0]:return StainDataset()
 return BASE_DATASET(which,split,crop,augment)
b.dataset=dataset;b.TrainingBatches=FixBatches

def dab_tensor(x):
 rgb=(x.float()+1)/2
 # Clamp only auxiliary RGB input; do not amplify gradients outside RGB range.
 # DAB nonnegative projection below uses a straight-through gradient so
 # missing brown stain still receives a gradient at negative raw DAB.
 rgb=rgb.clamp(1e-6,1)
 od=torch.log(rgb)/math_log_eps
 matrix=torch.as_tensor(hed_from_rgb[:,2],device=x.device,dtype=torch.float32)
 raw=torch.einsum('bchw,c->bhw',od,matrix)
 return raw+(raw.clamp_min(0)-raw).detach()
math_log_eps=float(np.log(1e-6))

def balanced_dab(endpoint,target):
 pd=dab_tensor(endpoint);gd=dab_tensor(target).detach();rgb=((target+1)/2).detach();mean=rgb.mean(1);maximum=rgb.amax(1);minimum=rgb.amin(1)
 saturation=(maximum-minimum)/maximum.clamp_min(1e-6)
 mask=((saturation>.08)&(mean<.97))|(mean<.9)
 # Exact binary opening + closing with scipy default cross footprint.
 cross=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],device=target.device,dtype=torch.float32)[None,None]
 def erode(m):return torch.nn.functional.conv2d(m[:,None].float(),cross,padding=1)[:,0]>=5
 def dilate(m):return torch.nn.functional.conv2d(m[:,None].float(),cross,padding=1)[:,0]>0
 mask=erode(dilate(dilate(erode(mask))))
 err=(pd-gd).abs()/.1;means=[];valid=[]
 for region in (mask&(gd>THRESHOLD),mask&(gd<=THRESHOLD)):
  counts=region.sum((1,2));means.append((err*region).sum((1,2))/counts.clamp_min(1));valid.append(counts>0)
 values=torch.stack(means,1);present=torch.stack(valid,1)
 return (values.sum(1)/present.sum(1).clamp_min(1)).mean()

def losses(model,perceptual,name,c,y,micro_index,check=False):
 if name!=NEW[1]:
  total,parts=BASE_LOSS(model,None,'direct_cfm',c,y,micro_index,check)
  return total,dict(parts,stain_raw=0.,stain_weighted=0.)
 rng=torch.Generator(device='cuda').manual_seed(b.stable_seed(42,f'path:{micro_index}'))
 torch.randn(c.shape,generator=rng,device=c.device) # preserve original time draws
 t=torch.rand((len(c),1,1,1),generator=rng,device=c.device);xt=(1-t)*c+t*y
 with torch.autocast('cuda',dtype=torch.bfloat16):v=model(xt,t.flatten(),c)
 fm=(v.float()-(y-c)).square().mean();raw=balanced_dab(xt+(1-t)*v.float(),y);total=fm+.1*raw
 if check:
  grad=torch.autograd.grad(raw,v,retain_graph=True)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
 return total,dict(fm_loss=fm.item(),lpips_raw=0.,lpips_weighted=0.,stain_raw=raw.item(),stain_weighted=.1*raw.item(),total_loss=total.item())
b.losses=losses

def config(name):
 if name in OLD:return json.loads((run_dir(name)/'config.json').read_text())
 return dict(b.PROTOCOL,experiment=name,initial_sha256=b.sha(OUT/'initial_model.pt'),splits={'HER2match':b.sha(OUT/'manifests/HER2match.jsonl')},fix_protocol=json.loads((OUT/'fix_protocol.json').read_text()),implementation_sha256=b.sha(Path(__file__)),base_sha256=b.sha(PACKAGE/'baseline.py'),crop_index_sha256=b.sha(OUT/'manifests/training_crops.csv') if name==NEW[0] else None)
b.config=config

def setup_gpu():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(0).total_memory);torch.backends.cudnn.benchmark=False
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def smoke():
 global ACTIVE
 setup_gpu();reports=[]
 for name in NEW:
  ACTIVE=name;ds=dataset('HER2match','train',512,True)
  if name==NEW[0]:
   keys=next(iter(FixBatches(len(ds),2,42)));samples=[ds[k] for k in keys]
   # Independent paired spatial verification, using an identity-coordinate pair.
   key=keys[0];r=ds.crops[key[0]];rng=random.Random(key[1]);fx,fy,k=rng.random()<.5,rng.random()<.5,rng.randrange(4)
   for field in ('he','ihc'):
    raw=read_rgb(ds.root/ds.lookup[r['id']][field]);raw=raw[int(r['top']):int(r['top'])+512,int(r['left']):int(r['left'])+512];raw=np.rot90(raw[::(-1 if fy else 1),::(-1 if fx else 1)],k)
    assert torch.equal(samples[0][field],torch.from_numpy(raw.copy()).permute(2,0,1).float()/127.5-1)
   allkeys=[key for batch in list(__import__('itertools').islice(FixBatches(len(ds),2,42),4)) for key in batch]
   assert [int(ds.crops[k[0]]['bin']) for k in allkeys]==[0,1,2,0,2,1,0,2]
   assert list(__import__('itertools').islice(FixBatches(len(ds),2,42),3,5))==list(__import__('itertools').islice(FixBatches(len(ds),2,42,start_batch=3),2))
  else:
   b.verify_crops(ds);probe=StainDataset()
   high=next(i for i,r in enumerate(probe.crops) if int(r['bin'])==2 and float(r['tissue_fraction'])>=.2)
   samples=[ds[(0,123)],probe[(high,124)]]
  c=torch.stack([s['he'] for s in samples]).cuda();y=torch.stack([s['ihc'] for s in samples]).cuda()
  ref=np.stack([_channels(((s['ihc'].permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8))[1] for s in samples])
  assert np.max(np.abs(dab_tensor(y).detach().cpu().numpy()-ref))<1e-5
  assert balanced_dab(y,y).item()==0
  model=b.model_new();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);times=[];torch.cuda.reset_peak_memory_stats()
  for step in range(4):
   torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
   for micro in range(4):
    loss,parts=losses(model,None,name,c,y,step*4+micro,check=True);assert torch.isfinite(loss);(loss/4).backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);assert norm>0;opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
  assert torch.equal(b.source_image(name,c,['a','b']),c)
  assert torch.isfinite(b.predict(model,name,c,['a','b'])).all()
  seconds=float(np.mean(times[1:]));reports.append(dict(experiment=name,updates=4,seconds_per_update=seconds,training_hours_excluding_io_validation=seconds*20000/3600,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finite_gradients=True,synchronized_crops=True,correct_source=True,dab_matches_skimage=True,stain_gradient_verified=name==NEW[1],last_losses=parts))
  b.dump(OUT/'smoke.json',reports);print('SMOKE',json.dumps(reports[-1]),flush=True)
  del model,opt,c,y,loss;gc.collect();torch.cuda.empty_cache()
 ACTIVE=None

def render():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,4,figsize=(20,4))
 for name in ALL:
  p=run_dir(name)/'history.csv'
  if not p.exists():continue
  with p.open() as f:rows=list(csv.DictReader(f))
  for ax,key in zip(axes,['fm_loss','total_loss','stain_raw','val_ssim']):
   rr=[r for r in rows if r.get(key) not in (None,'')]
   if not rr:continue
   x=np.array([int(r['step']) for r in rr]);y=np.array([float(r[key]) for r in rr])
   if key!='val_ssim' and len(y)>=100:x=x[99:];y=np.convolve(y,np.ones(100)/100,'valid')
   ax.plot(x,y,label=name);ax.set_title(key);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
 axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(OUT/'training_validation.png',dpi=150);plt.close(fig)
b.render=render

def evaluate(name):
 # Reuse trusted evaluator; re-score every selected pair and rebuild FID/KID.
 run=OUT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
 if name in OLD:
  for f in ('best.pt','config.json'):
   if not (run/f).exists():(run/f).symlink_to(run_dir(name)/f)
 rs=read_records(OUT/'manifests/test252.jsonl');dest=run/'HER2match';dest.mkdir(exist_ok=True)
 identity=dict(checkpoint_sha256=b.sha(run/'best.pt'),subset_sha256=b.sha(OUT/'manifests/test252.jsonl'),solver='heun16',native_shape=[1024,1024],seed=42)
 p=dest/'subset_identity.json'
 if p.exists():assert json.loads(p.read_text())==identity
 else:b.dump(p,identity)
 for r in rs:
  p=dest/'predictions'/f"{r['id']}.png"
  if p.exists():continue
  if name in OLD:
   for base in [ORIGINAL/'runs']:
    src=base/name/'HER2match/predictions'/f"{r['id']}.png"
    ident=base/name/'HER2match/evaluation_identity.json'
    if src.exists() and ident.exists() and json.loads(ident.read_text())['checkpoint_sha256']==identity['checkpoint_sha256']:
     p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(src);break
 old_dataset=b.dataset
 b.dataset=lambda which,split,crop=None,augment=False:b.PairedDataset(DATA_ROOT,OUT/'manifests/test252.jsonl',split,crop,augment)
 try:b.evaluate(name,'HER2match')
 finally:b.dataset=old_dataset
 with (dest/'per_image_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
 assert [r['id'] for r in metrics]==[r['id'] for r in rs]
 for r,m in zip(rs,metrics):m['her2_score']=r['her2_score']
 b.csv_save(dest/'per_image_metrics_labeled.csv',metrics)
 stainpath=dest/'staining_metrics.csv'
 if not stainpath.exists():
  rows=[]
  for i,r in enumerate(rs):
   rows.append(dict(id=r['id'],group=r['group'],her2_score=r['her2_score'],**analyze_pair(read_rgb(DATA_ROOT/r['ihc']),read_rgb(dest/'predictions'/f"{r['id']}.png"),CAL)))
   if (i+1)%50==0:b.status(phase='staining_metrics',experiment=name,pairs=i+1,total=252)
  b.csv_save(stainpath,rows)
 report()

def report():
 rows=[];classes=[]
 for name in ALL:
  folder=OUT/'runs'/name/'HER2match'
  if not (folder/'test_metrics.json').exists() or not (folder/'staining_metrics.csv').exists():continue
  metric=json.loads((folder/'test_metrics.json').read_text())
  with (folder/'staining_metrics.csv').open() as f:stains=list(csv.DictReader(f))
  with (folder/'per_image_metrics_labeled.csv').open() as f:pixels=list(csv.DictReader(f))
  def mean(rs,k):
   vals=[float(r[k]) for r in rs if r.get(k) not in (None,'')]
   return float(np.mean(vals)) if vals else None
  mappings={'dab_mae':'dab_mae','missing_stain':'understain_area_fraction','false_stain':'overstain_area_fraction','stain_iou':'dab_mask_iou','edge_ratio':'edge_strength_ratio'}
  rows.append(dict(model=name,count=metric['count'],**{k:metric[k] for k in ('ssim','psnr_db','lpips','fid','kid')},**{k:mean(stains,v) for k,v in mappings.items()}))
  for score in CLASSES:
   sr=[r for r in stains if r['her2_score']==score];pr=[r for r in pixels if r['her2_score']==score]
   classes.append(dict(model=name,slide_score=score,count=len(pr),**{k:mean(pr,k) for k in ('ssim','psnr_db','lpips')},**{k:mean(sr,v) for k,v in mappings.items()}))
 if rows:b.csv_save(OUT/'comparison.csv',rows)
 if classes:b.csv_save(OUT/'per_class_comparison.csv',classes)

@torch.inference_mode()
def validation_diagnostics(name):
 out=OUT/'runs'/name/'expanded_validation.csv'
 if out.exists():return
 val=[r for r in read_records(OUT/'manifests/HER2match.jsonl') if r['split']=='val'];selected=[]
 # Fixed eight hash-ranked images per validation slide, independent of predictions.
 for g in sorted({r['group'] for r in val}):selected.extend(sorted([r for r in val if r['group']==g],key=lambda r:hashlib.sha256(('42|valdiagnostic|'+r['id']).encode()).hexdigest())[:8])
 write_records(OUT/'manifests/expanded_validation.jsonl',selected)
 model=b.model_new();ck=torch.load(OUT/'runs'/name/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['ema']);model.eval();rows=[]
 ds=b.PairedDataset(DATA_ROOT,OUT/'manifests/expanded_validation.jsonl','val',512)
 for i,r in enumerate(selected):
  item=ds[i];p=b.predict(model,name,item['he'][None].cuda(),[r['id']])[0].clamp(-1,1).cpu();pred=((p+1)*127.5).round().byte().permute(1,2,0).numpy();gt=((item['ihc']+1)*127.5).round().byte().permute(1,2,0).numpy()
  rows.append(dict(id=r['id'],checkpoint_step=ck['step'],**b.pixel_metrics(pred,gt),**analyze_pair(gt,pred,CAL)))
 b.csv_save(out,rows);del model,ck;gc.collect();torch.cuda.empty_cache()

def main():
 global ACTIVE
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','scan','smoke','run','report','evaluate-old']);args=parser.parse_args()
 if args.action=='prepare':prepare();return
 if args.action=='scan':scan();return
 if args.action=='report':report();return
 lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  setup_gpu()
  if args.action=='evaluate-old':
   for name in OLD:evaluate(name)
   return
  if args.action=='smoke':smoke();return
  assert (OUT/'manifests/training_crops.csv').exists()
  smoke();indices=json.loads((OUT/'manifests/validation_subset.json').read_text())['indices']
  # First provide the larger-cohort baseline table; then sequential new training/eval.
  for name in OLD:evaluate(name)
  for name in NEW:
   ACTIVE=name;b.status(phase='training_start',experiment=name)
   b.train(name,indices);ACTIVE=None;validation_diagnostics(name);evaluate(name)
  render();report();b.status(phase='complete',experiments=list(ALL),dataset='HER2match',test_pairs=252)
 except BaseException as e:b.status(phase='failed',error=repr(e));traceback.print_exc();raise
if __name__=='__main__':main()
