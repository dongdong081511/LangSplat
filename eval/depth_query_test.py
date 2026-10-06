"""Depth-query cross-frame 2D retrieval test (N-teacher routing feasibility, EXP-047).

Protocol mirrors eval_image_query.py but queries are DEPTH patches:
  - query = Depth Anything V2 relative-depth crop inside GT bbox of frame A (32x32, normalized)
  - db    = frame B full depth map, sliding-window (32x32, stride 16) Pearson correlation
  - hit   = peak falls inside frame B's same-label GT bbox (first / any)

Feasibility question: does geometry alone localize objects across frames?
If depth retrieval is discriminative on repetition-rich scenes (ramen), a depth
teacher field is worth building; if not, N-teacher routing stays at 2 teachers.
"""
import json, glob, os, sys, argparse
import numpy as np
import cv2
import torch

PS, ST = 32, 16


def norm01(d):
    d = d.astype(np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-6)


def patchify(d):
    H, W = d.shape
    ys = np.arange(0, H - PS + 1, ST)
    xs = np.arange(0, W - PS + 1, ST)
    out = np.stack([d[y:y+PS, x:x+PS] for y in ys for x in xs])
    centers = [(y + PS/2, x + PS/2) for y in ys for x in xs]
    return out, centers


def pearson(P, q):
    P = P - P.mean(1, keepdims=True)
    q = q - q.mean()
    n = np.linalg.norm(P, axis=1) * (np.linalg.norm(q) + 1e-8)
    return (P @ q) / (n + 1e-8)


def load_gt(gt_dir):
    gt = {}
    for js in sorted(glob.glob(os.path.join(gt_dir, 'frame_*.json'))):
        with open(js) as f:
            d = json.load(f)
        h, w = d['info']['height'], d['info']['width']
        frame = d['info']['name'].split('.jpg')[0]
        objs = []
        for o in d['objects']:
            x1, y1, x2, y2 = np.array(o['bbox']).reshape(-1)
            objs.append({'label': o['category'],
                         'bbox': (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))})
        gt[frame] = {'size': (h, w), 'objs': objs}
    return gt


def extract_depths(scene_dir, out_dir, ckpt):
    os.makedirs(out_dir, exist_ok=True)
    repo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'third_party', 'Depth-Anything-V2')
    sys.path.insert(0, repo)
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vitl', features=256,
                            out_channels=[256, 512, 1024, 1024],
                            use_bn=False, use_clstoken=False)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.cuda().eval()
    imgs = sorted(glob.glob(os.path.join(scene_dir, 'images', '*.jpg')))
    for i, ip in enumerate(imgs):
        out = os.path.join(out_dir, f'{i:05d}.npy')
        if os.path.exists(out):
            continue
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            d = model.infer_image(img, input_size=518)
        np.save(out, d.astype(np.float16))
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--depth_ckpt', default='ckpts/depth_anything_v2_vitl.pth')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--tag', default='depth_query')
    args = ap.parse_args()

    SR = os.path.join('dataset', 'lerf_ovs', args.scene)
    DD = os.path.join(SR, 'depth_maps')
    if args.extract:
        n = extract_depths(SR, DD, args.depth_ckpt)
        print(f'[{args.tag}] extracted depth maps -> {DD}', flush=True)

    gt = load_gt(args.gt_dir)
    frames = sorted(gt.keys())
    patches, centers = {}, {}
    for fr in frames:
        idx = int(fr.split('_')[-1]) - 1
        d = norm01(np.load(os.path.join(DD, f'{idx:05d}.npy')))
        P, C = patchify(d)
        patches[fr], centers[fr] = P.reshape(len(P), -1), C
    n_obj = sum(len(v['objs']) for v in gt.values())
    print(f'[{args.tag}] scene={args.scene} frames={len(frames)} objects={n_obj}', flush=True)

    first = any_ = n = 0
    for frA in frames:
        for obj in gt[frA]['objs']:
            x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
            idxA = int(frA.split('_')[-1]) - 1
            crop = cv2.resize(np.load(os.path.join(DD, f'{idxA:05d}.npy')).astype(np.float32)
                              [max(0, y1):y2, max(0, x1):x2], (PS, PS)).reshape(-1)
            best = None
            for frB in frames:
                if frB == frA:
                    continue
                matches = [o for o in gt[frB]['objs'] if o['label'] == obj['label']]
                if not matches:
                    continue
                sims = pearson(patches[frB], crop)
                Hg = (gt[frB]['size'][0] - PS) // ST + 1
                Wg = (gt[frB]['size'][1] - PS) // ST + 1
                sm = cv2.filter2D(sims.reshape(Hg, Wg), -1, np.ones((3, 3)) / 9)
                pi = int(np.argmax(sm))
                cy, cx = centers[frB][pi]
                bboxes = [o['bbox'] for o in matches]
                cand = (float(sm.flat[pi]), frB, (cy, cx), bboxes)
                if best is None or cand[0] > best[0]:
                    best = cand
            if best is None:
                continue
            _, frB, (cy, cx), bboxes = best
            hf = bboxes[0][0] <= cx <= bboxes[0][2] and bboxes[0][1] <= cy <= bboxes[0][3]
            ha = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in bboxes)
            n += 1
            first += int(hf)
            any_ += int(ha)
    print(f'[{args.tag}] depth-query top1: first {first}/{n} = {first/max(n,1):.2%}  '
          f'any {any_}/{n} = {any_/max(n,1):.2%}', flush=True)


if __name__ == '__main__':
    main()
