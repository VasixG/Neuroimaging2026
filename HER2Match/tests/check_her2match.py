"""CPU checks of the packaged protocol and recorded results; no training/downloads."""
from pathlib import Path
import sys,json,csv,tempfile,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'her2match'))
import baseline,fixes
from ihc_flow.flow import integrate
from ihc_flow.data import TrainingBatches
from ihc_flow.runtime import write_json,save_image
from stain_error_analysis import _channels

def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
rs=records(ROOT/'her2match/manifests/HER2match.jsonl');test=records(ROOT/'her2match/manifests/test252.jsonl')
groups={s:{r['group'] for r in rs if r['split']==s} for s in ['train','val','test']}
assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
assert len(test)==len({r['id'] for r in test})==252
assert all(sum(r['her2_score']==s for r in test)==63 for s in fixes.CLASSES)
assert {r['id'] for r in test}<={r['id'] for r in rs if r['split']=='test'}
# CPU checks of the differentiable stain transform and its endpoint gradient.
rng=np.random.default_rng(42);rgb=rng.integers(1,255,size=(32,32,3),dtype=np.uint8)
y=torch.from_numpy(rgb).permute(2,0,1)[None].float()/127.5-1
assert np.max(np.abs(fixes.dab_tensor(y).numpy()[0]-_channels(rgb)[1]))<1e-5
assert fixes.balanced_dab(y,y).item()==0
brown=torch.tensor([.45,.25,.12])[None,:,None,None].expand(1,3,32,32)*2-1
blue=(torch.tensor([.5,.5,.8])[None,:,None,None].expand(1,3,32,32)*2-1).clone().requires_grad_()
loss=fixes.balanced_dab(blue,brown);loss.backward()
assert torch.isfinite(blue.grad).all() and blue.grad.abs().sum()>0
class Zero(torch.nn.Module):
 def forward(self,x,t,c):return torch.zeros_like(x)
assert torch.equal(integrate(Zero(),y,y,16,'heun'),y)
assert torch.equal(baseline.source_image('direct_cfm',y,['probe']),y)
assert not torch.equal(baseline.source_image('noise_cfm',y,['probe']),y)
import itertools
seq=list(itertools.islice(TrainingBatches(8,2,42),8))
assert seq[5:]==list(itertools.islice(TrainingBatches(8,2,42,start_batch=5),3))
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);write_json(tmp/'test.json',{'ok':True});save_image(y[0],tmp/'test.png')
 assert json.loads((tmp/'test.json').read_text())['ok']
 # Exercise path-portable preparation without requiring a GPU or real dataset.
 original=tmp/'original';(original/'manifests').mkdir(parents=True)
 for f in ['HER2match.jsonl','test252.jsonl','validation_subset.json']:
  (original/'manifests'/f).write_bytes((ROOT/'her2match/manifests'/f).read_bytes())
 torch.save({},original/'initial_model.pt')
 fixes.ORIGINAL=original;fixes.OUT=tmp/'fixes';fixes.b.OUT=fixes.OUT
 fixes.prepare()
 assert (fixes.OUT/'manifests/test252.jsonl').read_bytes()==(ROOT/'her2match/manifests/test252.jsonl').read_bytes()
# Reconcile submitted aggregate metrics with the actual saved per-image scores.
for summary in read(ROOT/'results/her2match/comparison.csv'):
 folder=ROOT/'results/her2match/runs'/summary['model']
 pixels=read(folder/'per_image_metrics_labeled.csv');stains=read(folder/'staining_metrics.csv')
 assert [r['id'] for r in pixels]==[r['id'] for r in test]
 assert [r['id'] for r in stains]==[r['id'] for r in test]
 for k in ['ssim','psnr_db','lpips']:
  assert abs(np.mean([float(r[k]) for r in pixels])-float(summary[k]))<1e-10
 for k,v in {'dab_mae':'dab_mae','missing_stain':'understain_area_fraction','false_stain':'overstain_area_fraction'}.items():
  assert abs(np.mean([float(r[v]) for r in stains if r[v]])-float(summary[k]))<1e-10
 recorded=json.loads((folder/'recorded_metrics.json').read_text());assert recorded['optimizer_updates']==20000
print('PASS: split/cohort identity, paired recorded scores, source initialization, sampler resume, DAB forward/gradients, portable preparation, and image/JSON helpers.')
