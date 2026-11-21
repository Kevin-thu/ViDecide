"""Post-training of a Wan2.2-TI2V-5B SFT model on collected rollouts (CFT and its baselines).

The training rows come from a metadata file written by scripts/posttrain/prepare_*_cft.py and mix three kinds of
rows: ``branch`` (a successful continuation resampled from a failed rollout, with
its context and divergence step), ``root`` (a successful whole rollout) and ``anchor`` (an expert video).

``rft_mode`` selects the method (the data of each method is prepared separately):
    cft           Corrective Fine-tuning: the divergence segment reuses the noise of the failed rollout and draws
                  t=1 with probability ``fork_sigma1_prob`` (0.5)
    decision_rft  decision-level RFT on the same continuations with fresh noise
    sample_rft    sample-level RFT on successful whole rollouts
    sft           continued SFT on expert videos

Usage:
    torchrun --nproc_per_node=3 scripts/posttrain/train_posttrain.py --config-path configs/cft/sokoban.yaml \
        --logdir outputs/cft/sokoban --disable-wandb metadata_path=<manifest> sft_checkpoint=<SFT full state>
(bash scripts/train.sh posttrain ... does the same).
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sft'))
import train_sft as P
from videcide.training.modes import canonical_rft_mode
import lmdb
from omegaconf import OmegaConf

BaseModel=P.SFTDiffusion
BaseTrainer=P.SFTTrainer
LMDB_ENVS={}


def sanitize_eos_target(latent, row):
    n = int(row['actual_latent_frames'])
    if not 0 < n <= len(latent):
        raise ValueError('Native context exceeds stored trajectory')
    result = latent[:n]
    if row.get('eos_target_pad_from') is not None:
        end = int(row['eos_target_pad_from'])
        if end != int(row['validity_mask_end']) or not 1 <= end <= n:
            raise ValueError('EOS padding must start at the valid supervision end')
        if end < n:
            result = result.clone()
            result[end:] = result[end - 1].clone()
    return result


def collected_rft_mask(row, bucket):
    n, start = int(row['actual_latent_frames']), int(row['validity_mask_start'])
    end = int(row.get('validity_mask_end', n))
    if not 0 <= start < end <= n <= bucket:
        raise ValueError('Invalid context or supervised suffix bounds')
    mask = torch.zeros(bucket)
    mask[start:end] = 1
    if row['rft_kind'] != 'branch':
        return mask
    t, block = int(row.get('divergence_latent', start)), int(row.get('fork_block', 1))
    if block < 1 or not start <= t < end or t + block > n:
        raise ValueError('Need a full context/noise fork block and nonempty valid fork supervision')
    valid_fork = min(t + block, end) - t
    total = end - start
    if total > valid_fork:
        mask[start:end] = .5 * total / (total - valid_fork)
        mask[t:t + valid_fork] = .5 * total / valid_fork
    return mask


class CollectedDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,max_pair=int(1e8),metadata_path=None,metadata_filter=None,frame_buckets=None,**_):
        rows=json.loads(Path(metadata_path).read_text())
        flt=OmegaConf.to_container(metadata_filter) if OmegaConf.is_config(metadata_filter) else (metadata_filter or {})
        self.metadata_rows=[r for r in rows if all(r.get(k)==v for k,v in flt.items())]
        self.index=[r['row_idx'] for r in self.metadata_rows]
        self.rows_by_id={r['row_idx']:r for r in rows}
        self.frame_buckets=frame_buckets;self.max_pair=max_pair
        self.spatial_shape=(48,24,24)
        key=str(Path(data_path).resolve())
        if key not in LMDB_ENVS:LMDB_ENVS[key]=lmdb.open(key,readonly=True,lock=False,readahead=False)
        self.env=LMDB_ENVS[key]
    def __len__(self):return min(len(self.index),self.max_pair)
    def nframes_for_index(self,i):return self.metadata_rows[i]['actual_latent_frames']
    def metadata_for_index(self,i):return self.metadata_rows[i]
    def __getitem__(self,i):
        r=self.metadata_rows[i];n=r['actual_latent_frames']
        if r['rft_kind']=='anchor' and not r.get('latent_path'):
            with self.env.begin() as txn:
                raw=txn.get(f"latents_{r['base_row_idx']}_data".encode())
            latent=torch.from_numpy(np.frombuffer(raw,dtype=np.float16).reshape(n,*self.spatial_shape).copy()).float()
        else:
            latent=torch.load(r['latent_path'],map_location='cpu',weights_only=False)['latents'].float()
            stored=int(r.get('stored_latent_frames',n))
            assert len(latent)==stored and 0<n<=stored
            latent=sanitize_eos_target(latent,r)
        bucket=next(b for b in self.frame_buckets if b>=n)
        x=torch.zeros(bucket,*self.spatial_shape);x[:n]=latent
        mask=collected_rft_mask(r,bucket)
        return {'prompts':r['prompt'],'clean_latent':x,'valid_frames':n,'row_idx':r['row_idx'],
                'clean_x_latents':None,'validity_mask':mask,'training_noise_group_id':r['training_noise_group_id']}


class NativeRFTModel(BaseModel):
    def _get_timestep(self,*args,**kwargs):
        index=super()._get_timestep(*args,**kwargs)
        current=getattr(self,'rft_current',None)
        if current and (current['cft'] or current.get('plain_sigma1')) and current['force_sigma1']:
            f=current['fork'];B=current.get('fork_block',1)
            index[:,f:f+B]=0
        if current and current.get('band') is not None:
            lo,hi=current['band']
            index[:,current['fork']]=torch.randint(lo,hi+1,(index.shape[0],),device=index.device,dtype=index.dtype)
        return index
    def generator_loss(self,*args,**kwargs):
        current=getattr(self,'rft_current',None)
        original=P.diffusion_module._randn_like_with_sample_seeds
        previous_weight=self.fixed_training_loss_weight
        if current and current['rft_kind']!='anchor' and current.get('weighting','uniform')=='uniform':
            self.fixed_training_loss_weight=1.
        def with_failed_noise(clean,seeds):
            noise=original(clean,seeds)
            f=current['fork'];assert 1<=f<noise.shape[1]
            fn=current['failed_noise'].to(noise)
            if fn.dim()==noise.dim()-1:
                noise[:,f:f+fn.shape[0]]=fn
            else:
                noise[:,f]=fn
            return noise
        if current and current['cft']:P.diffusion_module._randn_like_with_sample_seeds=with_failed_noise
        try:return super().generator_loss(*args,**kwargs)
        finally:
            P.diffusion_module._randn_like_with_sample_seeds=original
            self.fixed_training_loss_weight=previous_weight


class NativeRFTTrainer(BaseTrainer):
    def __init__(self,config):
        canonical_rft_mode(config.rft_mode)
        super().__init__(config)
    def _build_lr_scheduler(self,schedule_start_step=0,schedule_steps=None,warmup_steps=None,base_lr=None):
        scheduler=super()._build_lr_scheduler(schedule_start_step,schedule_steps,warmup_steps,base_lr)
        if scheduler is None or str(getattr(self.config,'lr_schedule','')).lower()!='constant_decay':return scheduler
        start,steps,warm=scheduler._cf_schedule_start_step,scheduler._cf_schedule_steps,scheduler._cf_warmup_steps
        decay=int(self.config.lr_decay_steps);assert 0<decay<=steps-warm
        def lr_lambda(current_step):
            local=max(0,int(current_step)-start)
            if warm>0 and local<warm:return local/warm
            return min(1.,max(0.,(steps-local)/decay))
        scheduler.lr_lambdas=[lr_lambda]*len(scheduler.lr_lambdas)
        for group,base in zip(self.generator_optimizer.param_groups,scheduler.base_lrs):group['lr']=base*lr_lambda(scheduler.last_epoch)
        scheduler._cf_schedule_type='constant_decay';scheduler._cf_decay_steps=decay
        return scheduler
    def _resume_from_full_state(self,path):
        phase_entry=str(path)==str(self.config.rft_phase_entry_checkpoint)
        previous=self.config.restore_rank_state
        previous_lr_reset=self.config.resume_reset_lr_scheduler
        if phase_entry:self.config.restore_rank_state=False
        else:self.config.resume_reset_lr_scheduler=False
        try:return super()._resume_from_full_state(path)
        finally:
            self.config.restore_rank_state=previous
            self.config.resume_reset_lr_scheduler=previous_lr_reset
    def _val_loss(self):
        previous=getattr(self.model,'rft_current',None);self.model.rft_current=None
        try:return super()._val_loss()
        finally:self.model.rft_current=previous
    def train_one_step(self,batch):
        idx=batch['row_idx'];idx=idx.tolist() if torch.is_tensor(idx) else idx
        if isinstance(idx,int):idx=[idx]
        assert len(idx)==1,'microbatch1 required for exact fork weighting'
        row=self.dataset.rows_by_id[idx[0]]
        enabled=canonical_rft_mode(self.config.rft_mode)=='cft' and row['rft_kind']=='branch'
        current={'cft':enabled,'fork':int(row.get('divergence_latent',row['validity_mask_start'])),'force_sigma1':False,'rft_kind':row['rft_kind'],
                 'weighting':getattr(self.config,'rft_weighting','uniform'),'band':None}
        if row['rft_kind']=='branch':
            if getattr(self.config,'fork_sigma','force1_half')=='band':
                current['band']=tuple(self.config.fork_index_band)
            else:
                draw=bool(torch.rand((),device=self.device)<float(getattr(self.config,'fork_sigma1_prob',.5)))
                current['force_sigma1']=enabled and draw
                if not enabled and bool(getattr(self.config,'plain_fork_sigma1',False)):
                    current['force_sigma1']=draw;current['plain_sigma1']=True
        current['fork_block']=int(row.get('fork_block',1))
        if enabled:
            tn=torch.load(row['failed_noise_path'],map_location='cpu',weights_only=False)
            current['failed_noise']=tn['failed_fork_block_noise'] if current['fork_block']>1 else tn['failed_fork_noise']
        self.model.rft_current=current
        p_unc=float(getattr(self.config,'rft_uncond_prob',0.) or 0.)
        if row['rft_kind']!='anchor' and p_unc>0 and bool(torch.rand((),device=self.device)<p_unc):
            neg=str(self.config.negative_prompt)
            batch={**batch,'prompts':[neg]*len(batch['prompts']) if isinstance(batch['prompts'],(list,tuple)) else neg}
        try:return super().train_one_step(batch)
        finally:self.model.rft_current=None


if __name__=='__main__':
    P.trainer_module.VariableLengthLatentLMDBDataset=CollectedDataset
    P.SFTDiffusion=NativeRFTModel;P.SFTTrainer=NativeRFTTrainer
    P.main()
