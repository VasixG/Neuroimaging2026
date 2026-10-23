"""Original three HER2match experiments; driven by her2match_experiments.ipynb."""
# %% Environment and common protocol
import os
# Set CUDA_VISIBLE_DEVICES before importing this module; caller chooses the GPU.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
DATA_ROOT = Path(os.environ.get('HER2_DATA_ROOT', ROOT / 'data/her2match')).expanduser().resolve()
OUT = Path(os.environ.get('HER2_BASELINE_OUT', ROOT / 'outputs/baseline')).resolve()
os.environ.setdefault('TORCH_HOME', str(ROOT / 'outputs/cache/torch'))
import sys, json, csv, time, math, random, hashlib, shutil, copy, gc, subprocess, fcntl
import numpy as np
import torch
import lpips
from PIL import Image
from torch.utils.data import DataLoader, Subset
from dataclasses import asdict
from datetime import datetime, timezone
sys.path.insert(0, str(PACKAGE))
from ihc_flow.model import ConditionalUNet
from ihc_flow.config import ModelConfig
from ihc_flow.data import PairedDataset, TrainingBatches
from ihc_flow.flow import integrate
from ihc_flow.runtime import initial_noise, stable_seed, save_image
from ihc_flow.metrics import pixel_metrics, frechet_distance, kernel_distance

NAMES = ('noise_cfm', 'direct_cfm', 'direct_cfm_lpips')
ARCH = dict(channels=(32,64,128,256), blocks_per_level=2, dropout=0.)
PROTOCOL = dict(seed=42, architecture=ARCH, crop=512, microbatch=2, accumulation=4,
    updates=20000, learning_rate=1e-4, weight_decay=.01, warmup=500, ema=.999,
    grad_clip=1., checkpoint_every=500, validation_samples=16, solver='heun', ode_steps=16,
    precision='bfloat16 autocast, FP32 states/losses/LPIPS', lpips_weight=.1,
    lpips_net='alex', lpips_version='0.1', lpips_crop=256, test_batch_size=2,
    test_spatial='native 1024x1024, no cropping or resizing; PNG quantization before metrics',
    feature_extractor='pytorch-fid 0.3.0 TensorFlow-compatible InceptionV3 pool3 2048',
    feature_preprocessing='RGB uint8 /255; internal bilinear 299x299 align_corners=False then [-1,1]',
    kid_subset_size=100, kid_subsets=100, kid_seed=42,
    memory_cap_GiB=14, grouping='Released HER2match slide splits; patient identities unavailable',
    selection='252 HER2match pairs;63 per slide-level score')

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def status(**kw):
    dump(OUT/'status.json',dict(pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    print(json.dumps(kw),flush=True)

def atomic_save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def csv_save(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def setup():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifests').mkdir(exist_ok=True)
    for filename in ('HER2match.jsonl','test252.jsonl'):
        src=PACKAGE/'manifests'/filename; dest=OUT/'manifests'/filename
        if dest.exists(): assert sha(dest)==sha(src), 'Manifest differs from submitted protocol'
        else: shutil.copy2(src,dest)
    manifest=OUT/'manifests/HER2match.jsonl'
    records=[json.loads(x) for x in manifest.read_text().splitlines()]
    groups={s:{r['group'] for r in records if r['split']==s} for s in ('train','val','test')}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    for r in records:
        assert Path(r['he']).name==Path(r['ihc']).name
        assert (DATA_ROOT/r['he']).is_file() and (DATA_ROOT/r['ihc']).is_file()
    audit=dict(grouping=PROTOCOL['grouping'],counts={s:sum(r['split']==s for r in records) for s in groups},groups={s:sorted(g) for s,g in groups.items()},overlap=False)
    dump(OUT/'manifests/audit.json',audit)
    val=[r for r in records if r['split']=='val']
    indices=np.linspace(0,len(val)-1,16,dtype=int).tolist()
    dump(OUT/'manifests/validation_subset.json',dict(indices=indices,ids=[val[i]['id'] for i in indices],spatial='center 512x512'))
    if not (OUT/'initial_model.pt').exists():
        torch.manual_seed(42);m=ConditionalUNet(ModelConfig(**ARCH));atomic_save(m.state_dict(),OUT/'initial_model.pt');del m
    dump(OUT/'protocol.json',PROTOCOL)
    return indices

def config(name):
    return dict(**PROTOCOL,experiment=name,initial_sha256=sha(OUT/'initial_model.pt'),
      splits={p.name:sha(p) for p in sorted((OUT/'manifests').glob('*.jsonl'))},
      implementation_sha256=sha(Path(__file__)),
      bundle_sha256={p.name:sha(p) for p in sorted((PACKAGE/'ihc_flow').glob('*.py'))})

def dataset(which,split,crop=None,augment=False):
    assert which=='HER2match'
    manifest=OUT/('manifests/test252.jsonl' if split=='test' else 'manifests/HER2match.jsonl')
    return PairedDataset(DATA_ROOT,manifest,split,crop,augment)

def model_new():
    model=ConditionalUNet(ModelConfig(**ARCH)).cuda()
    model.load_state_dict(torch.load(OUT/'initial_model.pt',map_location='cpu',weights_only=True))
    return model

def perceptual_new():return lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval().requires_grad_(False)

# %% Objectives and deterministic paired sampling

def source_image(name,c,ids):
    return initial_noise(c,ids,42) if name=='noise_cfm' else c.clone()

def losses(model,perceptual,name,c,y,micro_index,check=False):
    rng=torch.Generator(device='cuda').manual_seed(stable_seed(42,f'path:{micro_index}'))
    z=torch.randn(c.shape,generator=rng,device=c.device)
    t=torch.rand((len(c),1,1,1),generator=rng,device=c.device)
    source=z if name=='noise_cfm' else c
    xt=(1-t)*source+t*y
    with torch.autocast('cuda',dtype=torch.bfloat16):v=model(xt,t.flatten(),c)
    fm=(v.float()-(y-source)).square().mean()
    raw=torch.zeros((),device=c.device)
    if perceptual is not None:
        endpoint=xt+(1-t)*v.float()
        rng_crop=random.Random(stable_seed(42,f'lpips:{micro_index}'))
        ps,ys=[],[]
        for i in range(len(c)):
            top=rng_crop.randrange(c.shape[-2]-256+1);left=rng_crop.randrange(c.shape[-1]-256+1)
            ps.append(endpoint[i,:,top:top+256,left:left+256]);ys.append(y[i,:,top:top+256,left:left+256])
        raw=perceptual(torch.stack(ps).clamp(-1,1),torch.stack(ys)).mean()
        if check:
            grad=torch.autograd.grad(raw,v,retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum()>0,'LPIPS must backpropagate into generator'
    total=fm+.1*raw
    return total,dict(fm_loss=fm.detach().item(),lpips_raw=raw.detach().item(),lpips_weighted=.1*raw.detach().item(),total_loss=total.detach().item())

@torch.inference_mode()
def predict(model,name,c,ids):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return integrate(model,source_image(name,c,ids),c,16,'heun')

@torch.inference_mode()
def validation(model,name,indices):
    ds=Subset(dataset('HER2match','val',512),indices);rows=[];model.eval()
    for batch in DataLoader(ds,batch_size=2,num_workers=0):
        c=batch['he'].cuda();y=batch['ihc'];pred=predict(model,name,c,batch['id']).clamp(-1,1).cpu()
        for p,t in zip(pred,y):
            a=((p+1)*127.5).round().byte().permute(1,2,0).numpy()
            b=((t+1)*127.5).round().byte().permute(1,2,0).numpy()
            rows.append(pixel_metrics(a,b))
    return dict(val_ssim=float(np.mean([r['ssim'] for r in rows])),val_psnr=float(np.mean([r['psnr_db'] for r in rows])))

def verify_crops(ds):
    # Independently reconstruct the shared transformation from the raw real pair.
    sample=ds[(0,987)];record=ds.records[0];rng=random.Random(987)
    with Image.open(ds.root/record['he']) as im:w,h=im.size
    top=rng.randrange(h-512+1);left=rng.randrange(w-512+1)
    fx,fy,k=rng.random()<.5,rng.random()<.5,rng.randrange(4)
    for field,key in [('he','he'),('ihc','ihc')]:
        with Image.open(ds.root/record[field]) as im:a=np.array(im.convert('RGB'))
        a=a[top:top+512,left:left+512]
        a=np.rot90(a[::(-1 if fy else 1),::(-1 if fx else 1)],k)
        expected=torch.from_numpy(a.copy()).permute(2,0,1).float()/127.5-1
        assert torch.equal(expected,sample[key]),'Unsynchronized spatial transform'

# %% Real-data preflight: all three losses, gradients, sources and memory

def smoke(indices):
    ds=dataset('HER2match','train',512,True);verify_crops(ds)
    batches=list(zip(range(12),TrainingBatches(len(ds),2,42)))
    cached=[]
    for j,keys in batches:
        samples=[ds[key] for key in keys]
        cached.append((torch.stack([x['he'] for x in samples]).cuda(),torch.stack([x['ihc'] for x in samples]).cuda(),[x['id'] for x in samples]))
    reports=[]
    for name in NAMES:
        model=model_new();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        perceptual=perceptual_new() if name.endswith('lpips') else None
        torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(3):
            torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
            for j in range(4):
                c,y,ids=cached[step*4+j]
                loss,logs=losses(model,perceptual,name,c,y,step*4+j,check=True)
                assert torch.isfinite(loss)
                (loss/4).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
            assert perceptual is None or all(p.grad is None for p in perceptual.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize();times.append(time.monotonic()-start)
        c,y,ids=cached[0];source=source_image(name,c,ids)
        assert torch.equal(source,c) if name!='noise_cfm' else not torch.equal(source,c)
        class Zero(torch.nn.Module):
            def forward(self,x,t,c):return torch.zeros_like(x)
        assert torch.equal(integrate(Zero(),source,c,16,'heun'),source)
        assert torch.equal((1-torch.zeros_like(c))*source+torch.zeros_like(c)*y,source)
        assert torch.equal((1-torch.ones_like(c))*source+torch.ones_like(c)*y,y)
        assert torch.isfinite(predict(model,name,c,ids)).all()
        seconds=float(np.mean(times[1:]))
        reports.append(dict(experiment=name,steps=3,finite_gradients=True,paired_crops_verified=True,correct_initialization=True,
          lpips_gradient_verified=True if perceptual else None,seconds_per_update=seconds,
          training_hours_lower_estimate=seconds*20000/3600,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
        dump(OUT/'smoke.json',reports);print('SMOKE',json.dumps(reports[-1]),flush=True)
        del model,opt,perceptual,loss;gc.collect();torch.cuda.empty_cache()
    del cached;gc.collect();torch.cuda.empty_cache()
    # Native-resolution Heun benchmark uses only validation images, never test images.
    model=model_new();ds=dataset('HER2match','val');samples=[ds[i] for i in indices[:2]]
    c=torch.stack([x['he'] for x in samples]).cuda();ids=[x['id'] for x in samples]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    p=predict(model,'noise_cfm',c,ids);torch.cuda.synchronize()
    dump(OUT/'native_inference_benchmark.json',dict(batch=2,shape=list(p.shape),seconds_per_image=(time.monotonic()-start)/2,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
    del model,c,p;gc.collect();torch.cuda.empty_cache()

# %% Training: identical initialization and sampling, EMA, guarded resume

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])

def train(name,indices):
    cfg=config(name);run=OUT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists():
        assert fingerprint(json.loads((run/'config.json').read_text()))==fingerprint(cfg),'Existing run config differs'
    dump(run/'config.json',cfg)
    model=model_new();ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    # BF16 does not need scaling; retain scaler state explicitly for resumability.
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    perceptual=perceptual_new() if name.endswith('lpips') else None
    step=0;best=-float('inf');history=[]
    if (run/'latest.pt').exists():
        ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=False) # trusted locally produced checkpoint
        assert ck['config_identity']==fingerprint(cfg),'Refusing mismatched experiment/config/split resume'
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        step=ck['step'];best=ck['best_ssim'];restore_rng(ck['rng']);history=ck['history'];del ck
    if history:csv_save(run/'history.csv',history)
    elif (run/'history.csv').exists():(run/'history.csv').unlink()
    ds=dataset('HER2match','train',512,True)
    loader=iter(DataLoader(ds,batch_sampler=TrainingBatches(len(ds),2,42,start_batch=step*4),num_workers=4,pin_memory=True,prefetch_factor=2))
    model.train();start=time.monotonic();start_step=step
    while step<20000:
        lr=1e-4*((step+1)/500 if step<500 else .5*(1+math.cos(math.pi*(step-500)/(20000-500))))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);row=dict(step=step+1,learning_rate=lr)
        for micro in range(4):
            batch=next(loader);c=batch['he'].cuda(non_blocking=True);y=batch['ihc'].cuda(non_blocking=True)
            assert len(c)==2
            loss,parts=losses(model,perceptual,name,c,y,step*4+micro)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {name} loss at {step}')
            (loss/4).backward()
            for k,v in parts.items():row[k]=row.get(k,0)+v/4
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.001)
            for eb,b in zip(ema.buffers(),model.buffers()):eb.copy_(b)
        step+=1;row['grad_norm']=norm.item();row['seconds_per_update']=(time.monotonic()-start)/(step-start_step)
        if step%500==0:
            row.update(validation(ema,name,indices))
            if row['val_ssim']>best:
                best=row['val_ssim'];atomic_save(dict(ema=ema.state_dict(),step=step,val_ssim=best,config=cfg,config_identity=fingerprint(cfg)),run/'best.pt')
        history.append(row)
        # Append each completed optimizer update so partial progress remains visible.
        fields=['step','learning_rate','fm_loss','lpips_raw','lpips_weighted','total_loss','stain_raw','stain_weighted','grad_norm','seconds_per_update','val_ssim','val_psnr']
        new=not (run/'history.csv').exists()
        with (run/'history.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields)
            if new:w.writeheader()
            w.writerow(row)
        if step%50==0:status(phase='training',experiment=name,step=step,total=20000,seconds_per_update=row['seconds_per_update'])
        if step%500==0:
            atomic_save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),step=step,best_ssim=best,config=cfg,config_identity=fingerprint(cfg),history=history),run/'latest.pt')
            render()
    dump(run/'training_complete.json',dict(step=step,best_val_ssim=best,config_identity=fingerprint(cfg)))
    del model,ema,opt,perceptual,loader,loss,c,y;gc.collect();torch.cuda.empty_cache()

# %% Final evaluation: every native pair, saved PNGs, consistent metrics

def panel(which,ds,folder):
    chosen=np.linspace(0,len(ds)-1,4,dtype=int).tolist()
    dump(folder/'panel_examples.json',[ds.records[i]['id'] for i in chosen])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(12,16))
    for row,i in enumerate(chosen):
        r=ds.records[i]
        paths=[ds.root/r['he'],folder/'predictions'/f"{r['id']}.png",ds.root/r['ihc']]
        for col,path in enumerate(paths):
            with Image.open(path) as im:axes[row,col].imshow(im.copy())
            axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(['H&E','Generated HER2','Real HER2'][col])
    fig.tight_layout();fig.savefig(folder/'predictions.png',dpi=120);plt.close(fig)

@torch.inference_mode()
def evaluate(name,which):
    from pytorch_fid.inception import InceptionV3
    run=OUT/'runs'/name;folder=run/which;folder.mkdir(exist_ok=True)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    assert ck['config_identity']==fingerprint(config(name))
    identity=dict(checkpoint_sha256=sha(run/'best.pt'),config_identity=ck['config_identity'],dataset=which)
    if (folder/'evaluation_identity.json').exists():assert json.loads((folder/'evaluation_identity.json').read_text())==identity
    else:dump(folder/'evaluation_identity.json',identity)
    if (folder/'test_metrics.json').exists() and (folder/'predictions.png').exists():return
    model=model_new();model.load_state_dict(ck['ema']);model.eval();del ck
    ds=dataset(which,'test');loader=DataLoader(ds,batch_size=2,num_workers=4,pin_memory=True)
    for i,batch in enumerate(loader):
        paths=[folder/'predictions'/f'{identity}.png' for identity in batch['id']]
        if not all(p.exists() for p in paths):
            pred=predict(model,name,batch['he'].cuda(),batch['id'])
            for p,path in zip(pred,paths):path.parent.mkdir(parents=True,exist_ok=True);save_image(p,path)
        if i%100==0:status(phase='generating',experiment=name,dataset=which,pairs=min((i+1)*2,len(ds)),total=len(ds))
    del model,loader;gc.collect();torch.cuda.empty_cache()
    perceptual=perceptual_new();features=InceptionV3([3]).cuda().eval().requires_grad_(False)
    rows=[];reals=[];fakes=[]
    for i,r in enumerate(ds.records):
        with Image.open(folder/'predictions'/f"{r['id']}.png") as im:a=np.array(im.convert('RGB'))
        with Image.open(ds.root/r['ihc']) as im:b=np.array(im.convert('RGB'))
        assert a.shape==b.shape==(1024,1024,3)
        p=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).cuda().float()/255
        y=torch.from_numpy(b).permute(2,0,1).unsqueeze(0).cuda().float()/255
        row=dict(id=r['id'],group=r['group'],**pixel_metrics(a,b),lpips=perceptual(2*p-1,2*y-1).item())
        rows.append(row);fakes.append(features(p)[0].flatten(1).cpu().numpy());reals.append(features(y)[0].flatten(1).cpu().numpy())
        if (i+1)%100==0:
            csv_save(folder/'per_image_metrics.csv',rows);status(phase='metrics',experiment=name,dataset=which,pairs=i+1,total=len(ds))
    csv_save(folder/'per_image_metrics.csv',rows)
    real=np.concatenate(reals);fake=np.concatenate(fakes)
    np.savez(folder/'inception_features.npz',real=real,fake=fake)
    metrics=dict(experiment=name,count=len(rows),**identity,
      **{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in ('ssim','psnr_db','lpips')},
      perfect_match_count=sum(r['perfect_match'] for r in rows),protocol=PROTOCOL,
      limitations='FID/KID measure distributions and do not establish input-output correspondence. PSNR of exact matches is infinite, serialized as null in per-image rows; mean excludes these.')
    if len(rows)>=2:metrics.update(fid=frechet_distance(real,fake),**kernel_distance(real,fake,100,100,42))
    else:metrics.update(fid=None,kid=None,undefined_reason='Fewer than two test pairs')
    dump(folder/'test_metrics.json',metrics);panel(which,ds,folder)
    del perceptual,features,p,y;gc.collect();torch.cuda.empty_cache();render()

# %% Combined plots and the HER2match comparison table

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,4));rows=[]
    for name in NAMES:
        run=OUT/'runs'/name;history=run/'history.csv'
        if history.exists():
            with history.open() as f:h=list(csv.DictReader(f))
            if h:
                steps=np.array([int(r['step']) for r in h]);fm=np.array([float(r['fm_loss']) for r in h]);total=np.array([float(r['total_loss']) for r in h])
                width=min(100,len(h));smooth=lambda a:np.convolve(a,np.ones(width)/width,mode='valid')
                axes[0].plot(steps[width-1:],smooth(fm),label=name);axes[1].plot(steps[width-1:],smooth(total),label=name)
                val=[r for r in h if r.get('val_ssim')]
                axes[2].plot([int(r['step']) for r in val],[float(r['val_ssim']) for r in val],label=name)
                for filename,keys in [('losses.png',['fm_loss','lpips_raw','lpips_weighted','total_loss']),('validation_metrics.png',['val_ssim','val_psnr'])]:
                    f,ax=plt.subplots(1,len(keys),figsize=(5*len(keys),4));ax=np.atleast_1d(ax)
                    for a,key in zip(ax,keys):
                        selected=[r for r in h if r.get(key)]
                        a.plot([int(r['step']) for r in selected],[float(r[key]) for r in selected]);a.set_title(key);a.set_xlabel('Optimizer update')
                    f.tight_layout();f.savefig(run/filename,dpi=130);plt.close(f)
        for which in ('HER2match',):
            p=run/which/'test_metrics.json'
            if p.exists():
                m=json.loads(p.read_text());rows.append({k:m[k] for k in ('experiment','dataset','count','ssim','psnr_db','lpips','fid','kid')})
    for ax,title in zip(axes,['Training FM MSE (100-update mean)','Total loss (100-update mean)','Validation SSIM (EMA, Heun 16)']):
        ax.set_title(title);ax.set_xlabel('Optimizer update');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'plots.png',dpi=160);plt.close(fig)
    if rows:
        csv_save(OUT/'comparison.csv',rows)
        fig,ax=plt.subplots(figsize=(14,1.2+.5*len(rows)));ax.axis('off')
        keys=list(rows[0]);cells=[[f'{r[k]:.4f}' if isinstance(r[k],float) else r[k] for k in keys] for r in rows]
        table=ax.table(cellText=cells,colLabels=keys,loc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.5)
        fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)


def export_notebook():
    pass # The root notebook is the review and execution entry point.

# %% Sequential execution, exclusive lock and durable status

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        indices=setup();status(phase='preflight')
        smoke(indices)
        for name in NAMES:train(name,indices)
        for name in NAMES:
            for which in ('HER2match',):evaluate(name,which)
        render();status(phase='complete',experiments=list(NAMES),test_datasets=['HER2match'])
    except BaseException as error:
        status(phase='failed',error=repr(error));raise

if __name__=='__main__':main()
