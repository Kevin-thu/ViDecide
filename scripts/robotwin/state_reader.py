"""State reader of the RoboTwin-Plan verifier: frozen DINOv3 ViT-L/16 patch features and a small transformer decoder
that predict, for every frame, the position of each block, whether a gripper holds it and whether it is visible, and
the image position of both grippers (Sec. B of the supplement).

    python scripts/robotwin/state_reader.py infer --videos JOBS.json --reader checkpoints/videcide/robotwin_verifier/state_reader.pt --out DIR
    python scripts/robotwin/state_reader.py train --out DIR [--max-videos N] [--epochs 1]

infer reads a JSON list of jobs ({key, video, row, eos}) and writes DIR/<key>.json with one prediction per frame up to
the EOS frame. train fits the decoder on the simulator states of the RoboTwin-Plan training episodes and validates on
the validation episodes. The backbone is loaded from checkpoints/dinov3-vitl16.
"""
import argparse, json, math, random, sys, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
BACKBONE = 'checkpoints/dinov3-vitl16'
G = 32; RES = 512
COLOURS = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
XYZ0 = np.array([0.0, -0.14, 0.75], np.float32)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frame_targets(row, frames):
    import scene
    meta = json.loads(scene.data_path(row['source_path']).read_text()); cam = meta['camera']; info = meta['info']['info']
    st = json.loads(scene.data_path(row['state_path']).read_text()); cols = info['colours']; nb = len(cols)
    edge = scene.Geometry(row).edge_px; size = frames.shape[1]; r = max(2, int(0.2 * edge)); out = []
    for t in range(min(len(frames), len(st))):
        s = st[t]; hsv = cv2.cvtColor(frames[t], cv2.COLOR_RGB2HSV); sat = (hsv[..., 1] > 120) & (hsv[..., 2] > 70)
        held = {}; arms = []
        for a in range(2):
            e = np.array(s['ee'][a]); p = scene.project((e[0], e[1], e[2] - 0.135), cam)
            hb = [b for b in range(nb) if np.linalg.norm(np.array(s['actors'][f'block{b + 1}'][:2]) - e[:2]) < 0.02
                  and abs(e[2] - s['actors'][f'block{b + 1}'][2] - 0.135) < 0.03]
            for b in hb:
                held[b] = True
            inv = 0 <= p[0] < size and 0 <= p[1] < size
            arms.append((float(np.clip(p[0] / size, 0, 1)), float(np.clip(p[1] / size, 0, 1)), float(inv), float(bool(hb))))
        blocks = {}
        for b, c in enumerate(cols):
            q = np.array(s['actors'][f'block{b + 1}'][:3], np.float32); p = scene.project(tuple(q), cam); x, y = int(p[0]), int(p[1])
            vis = False
            if r <= x < size - r and r <= y < size - r:
                win = (slice(y - r, y + r + 1), slice(x - r, x + r + 1))
                vis = bool((scene.hue_mask(hsv[win], c) & sat[win]).mean() >= 0.5)
            blocks[c] = ((q - XYZ0).tolist(), float(held.get(b, False)), float(vis))
        out.append((blocks, arms))
    return out


class Reader(torch.nn.Module):
    def __init__(self, dim=1024, d=256, layers=3):
        super().__init__()
        self.proj = torch.nn.Linear(dim, d); self.pos = torch.nn.Parameter(torch.randn(G * G, d) * 0.02)
        self.colour = torch.nn.Embedding(len(COLOURS), d); self.arm = torch.nn.Embedding(2, d)
        layer = torch.nn.TransformerDecoderLayer(d, 8, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = torch.nn.TransformerDecoder(layer, layers)
        self.block_head = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 5))
        self.arm_head = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 4))

    def forward(self, tokens, colour_ids):
        B = tokens.shape[0]; mem = self.proj(tokens) + self.pos
        q = torch.cat([self.colour(colour_ids), self.arm.weight]).unsqueeze(0).expand(B, -1, -1)
        h = self.dec(q, mem); K = len(colour_ids)
        return self.block_head(h[:, :K]), self.arm_head(h[:, K:])


@torch.no_grad()
def embed(model, frames):
    x = torch.from_numpy(np.stack([cv2.resize(f, (RES, RES), interpolation=cv2.INTER_CUBIC) for f in frames]))
    x = ((x.permute(0, 3, 1, 2).float().cuda() / 255) - MEAN.cuda()) / STD.cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = model(pixel_values=x).last_hidden_state
    return h[:, -G * G:].float()


class Videos(torch.utils.data.Dataset):
    def __init__(self, rows, stride):
        self.rows, self.stride = rows, stride

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import scene
        r = self.rows[i]
        try:
            fr = scene.read(scene.data_path(r['native_video'])); lab = frame_targets(r, fr)
        except Exception:
            return None
        idx = list(range(random.randrange(self.stride), len(lab), self.stride))
        cols = list(lab[0][0].keys())
        x = np.stack([cv2.resize(fr[t], (RES, RES), interpolation=cv2.INTER_CUBIC) for t in idx])
        xyz = np.array([[lab[t][0][c][0] for c in cols] for t in idx], np.float32)
        hv = np.array([[lab[t][0][c][1:] for c in cols] for t in idx], np.float32)
        arms = np.array([lab[t][1] for t in idx], np.float32)
        return torch.from_numpy(x), torch.tensor([COLOURS.index(c) for c in cols]), torch.from_numpy(xyz), torch.from_numpy(hv), torch.from_numpy(arms)


def tokens_of(model, x):
    device = next(model.parameters()).device
    x = ((x.permute(0, 3, 1, 2).float().to(device) / 255) - MEAN.to(device)) / STD.to(device)
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with torch.no_grad(), autocast:
        return model(pixel_values=x).last_hidden_state[:, -G * G:].float()


def losses(ob, oa, xyz, hv, arms):
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    vis = hv[..., 1] > 0.5
    l_xyz = torch.nn.functional.smooth_l1_loss(ob[..., :3][vis], xyz[vis], beta=0.01) if vis.any() else ob.sum() * 0
    l = 10 * l_xyz + bce(ob[..., 3], hv[..., 0]) + bce(ob[..., 4], hv[..., 1])
    inv = arms[..., 2] > 0.5
    l = l + bce(oa[..., 2], arms[..., 2]) + bce(oa[..., 3], arms[..., 3])
    if inv.any():
        l = l + 10 * torch.nn.functional.smooth_l1_loss(oa[..., :2][inv], arms[..., :2][inv], beta=0.01)
    return l


@torch.no_grad()
def validate(model, net, rows):
    st = dict(n=0, held_ok=0, vis_ok=0, held_pos=0, held_hit=0, cell_n=0, cell_ok=0, arm_n=0); err = []; aerr = []
    import scene
    for r in rows:
        fr = scene.read(scene.data_path(r['native_video'])); lab = frame_targets(r, fr); idx = list(range(0, len(lab), 3))
        cols = list(lab[0][0].keys()); cid = torch.tensor([COLOURS.index(c) for c in cols]).cuda()
        x = torch.from_numpy(np.stack([cv2.resize(fr[t], (RES, RES), interpolation=cv2.INTER_CUBIC) for t in idx]))
        ob, oa = net(tokens_of(model, x), cid); ob, oa = ob.cpu(), oa.cpu()
        geo = scene.Geometry(r); slots = np.array(geo.info['slots']); half = float(np.median(geo.sizes))
        for i, t in enumerate(idx):
            for k, c in enumerate(cols):
                xyz, h, v = lab[t][0][c]; st['n'] += 1
                st['held_ok'] += int((ob[i, k, 3] > 0) == (h > .5)); st['vis_ok'] += int((ob[i, k, 4] > 0) == (v > .5))
                if h > .5:
                    st['held_pos'] += 1; st['held_hit'] += int(ob[i, k, 3] > 0)
                if v > .5:
                    p = ob[i, k, :3].numpy() + XYZ0; g = np.array(xyz) + XYZ0; err.append(np.abs(p - g))
                    if h < .5:
                        st['cell_n'] += 1
                        same_slot = np.argmin(np.linalg.norm(slots - p[:2], axis=1)) == np.argmin(np.linalg.norm(slots - g[:2], axis=1))
                        st['cell_ok'] += int(same_slot and abs(p[2] - g[2]) < half)
            for a in range(2):
                if lab[t][1][a][2] > .5:
                    st['arm_n'] += 1; aerr.append(np.abs(oa[i, a, :2].numpy() - np.array(lab[t][1][a][:2])) * fr.shape[1])
    e = np.concatenate(err) if err else np.zeros(1)
    return {'held_acc': st['held_ok'] / max(1, st['n']), 'held_recall': st['held_hit'] / max(1, st['held_pos']),
            'visible_acc': st['vis_ok'] / max(1, st['n']), 'cell_acc_visible_resting': st['cell_ok'] / max(1, st['cell_n']),
            'xyz_err_cm_median': float(100 * np.median(e)), 'xyz_err_cm_p99': float(100 * np.quantile(e, .99)),
            'arm_err_px_median': float(np.median(np.concatenate(aerr))) if aerr else None}


def train(a):
    import scene
    from transformers import AutoModel
    model = AutoModel.from_pretrained(BACKBONE).cuda().eval()
    net = Reader(model.config.hidden_size).cuda()
    rows, val = scene.manifest_rows(['train']), scene.manifest_rows(['val'])
    random.seed(0); random.shuffle(rows); random.shuffle(val); rows = rows[:a.max_videos] if a.max_videos else rows; val = val[:a.val]
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    total = a.epochs * len(rows); warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        random.seed(ep); random.shuffle(rows)
        dl = torch.utils.data.DataLoader(Videos(rows, a.stride), batch_size=None, num_workers=16, prefetch_factor=4)
        for vi, item in enumerate(dl):
            if item is None:
                continue
            x, cid, xyz, hv, arms = item
            tok = tokens_of(model, x); cid = cid.cuda()
            ob, oa = net(tok, cid); loss = losses(ob, oa, xyz.cuda(), hv.cuda(), arms.cuda())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if vi % 200 == 0:
                print(json.dumps({'epoch': ep, 'video': vi, 'loss': round(float(loss), 4), 'minutes': round((time.time() - t0) / 60, 1)}), flush=True)
            if vi and vi % a.val_every == 0:
                net.eval(); v = validate(model, net, val); net.train()
                print(json.dumps({'epoch': ep, 'video': vi, 'val': v}), flush=True); torch.save(net.state_dict(), a.out / 'reader.pt')
    net.eval(); v = validate(model, net, val); print(json.dumps({'final_val': v}), flush=True)
    torch.save(net.state_dict(), a.out / 'reader.pt'); (a.out / 'val.json').write_text(json.dumps(v))


@torch.no_grad()
def infer(a):
    import scene
    from transformers import AutoModel
    device = getattr(a, 'device', 'cuda')
    if device == 'cpu':
        torch.set_num_threads(4)
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    net = Reader(model.config.hidden_size).to(device)
    net.load_state_dict(torch.load(a.reader, map_location=device, weights_only=False)); net.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    if getattr(a, 'strip_eos_overlay', False):
        scene.cache_contract(a.out, {'preprocess': scene.PREPROCESS_VERSION,
                                     'reader': str(a.reader), 'backbone': BACKBONE})
    for j in json.loads(a.videos.read_text())[a.shard::a.shards]:
        dst = a.out / f"{j['key']}.json"
        if dst.exists():
            continue
        fr = scene.read_visible(j['video'])
        if j.get('eos') is not None:
            fr = fr[:j['eos']]
        if getattr(a, 'strip_eos_overlay', False):
            fr = scene.remove_eos_overlay(fr)
        row = j['row'] if 'row' in j else scene.rows_by_id()[j['key'].split('_', 1)[1]]   # keys are <prefix>_<sample id>
        cols = scene.Geometry(row).info['colours']
        cid = torch.tensor([COLOURS.index(c) for c in cols]).to(device); size = fr.shape[1]; out = []
        for i in range(0, len(fr), 32):
            x = torch.from_numpy(np.stack([cv2.resize(f, (RES, RES), interpolation=cv2.INTER_CUBIC) for f in fr[i:i + 32]]))
            ob, oa = net(tokens_of(model, x), cid); ob, oa = ob.cpu().numpy(), oa.cpu().numpy()
            for t in range(len(ob)):
                out.append({'blocks': {c: {'xyz': (ob[t, k, :3] + XYZ0).round(4).tolist(), 'held': float(1 / (1 + np.exp(-ob[t, k, 3]))),
                                           'visible': float(1 / (1 + np.exp(-ob[t, k, 4])))} for k, c in enumerate(cols)},
                            'arms': [{'x': float(oa[t, q, 0] * size), 'y': float(oa[t, q, 1] * size), 'in_view': float(1 / (1 + np.exp(-oa[t, q, 2]))),
                                      'holding': float(1 / (1 + np.exp(-oa[t, q, 3])))} for q in range(2)]})
        dst.write_text(json.dumps(out)); print(j['key'], len(out), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['train', 'infer']); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--videos', type=Path, help='infer: JSON list of jobs {key, video, row, eos}')
    ap.add_argument('--stride', type=int, default=3, help='train: every n-th frame of a video')
    ap.add_argument('--max-videos', type=int, default=0, help='train: number of training videos (0: all)')
    ap.add_argument('--val', type=int, default=60); ap.add_argument('--val-every', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=1); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--reader'); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--strip-eos-overlay', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    train(a) if a.cmd == 'train' else infer(a)
