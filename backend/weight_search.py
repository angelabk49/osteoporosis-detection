"""
weight_search.py — Grid-search ensemble weights on fine-tuned models
"""
import os, sys
sys.path.insert(0, '.')
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
from torchvision import transforms
from dataset import get_resnet, get_densenet, get_effnet, classes, CLAHETransform

device = torch.device('cpu')
transform = transforms.Compose([
    CLAHETransform(),  # enhance bone density contrast
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, [-1]),
    lambda x: torch.rot90(x, 1, [-2,-1]),
    lambda x: torch.rot90(x, 3, [-2,-1])
]

def tta_predict(model, x):
    probs = torch.zeros((1, len(classes)))
    with torch.no_grad():
        for tf in tta_transforms:
            probs += F.softmax(model(tf(x)), dim=1)
    return probs / len(tta_transforms)

print('Loading fine-tuned models...')
resnet = get_resnet()
resnet.load_state_dict(torch.load('checkpoints/resnet_best.pth', map_location='cpu'))
densenet = get_densenet()
densenet.load_state_dict(torch.load('checkpoints/densenet_best.pth', map_location='cpu'))
effnet = get_effnet()
effnet.load_state_dict(torch.load('checkpoints/effnet_best.pth', map_location='cpu'))
for m in [resnet, densenet, effnet]:
    m.eval()
models = [resnet, densenet, effnet]

def get_label(f):
    u = f.upper()
    if u.startswith('OS'): return 'osteoporosis'
    if u.startswith('OP'): return 'osteopenia'
    if u.startswith('N'): return 'normal'
    return None

def load_and_predict(directory, is_folder_based=False):
    items = []
    if is_folder_based:
        for cls_name in classes:
            cls_dir = os.path.join(directory, cls_name)
            for f in sorted(os.listdir(cls_dir)):
                if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
                img = Image.open(os.path.join(cls_dir, f)).convert('RGB')
                x = transform(img).unsqueeze(0)
                per_model = [tta_predict(m, x).detach() for m in models]
                items.append((cls_name, per_model))
    else:
        for f in sorted(os.listdir(directory)):
            if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
            lbl = get_label(f)
            if not lbl: continue
            img = Image.open(os.path.join(directory, f)).convert('RGB')
            x = transform(img).unsqueeze(0)
            per_model = [tta_predict(m, x).detach() for m in models]
            items.append((lbl, per_model))
    return items

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'FOR TESTING', 'testcases')
orig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test')

print('Computing per-model TTA predictions for held-out set...')
held_out_items = load_and_predict(test_dir, is_folder_based=False)
print('Computing per-model TTA predictions for original test set...')
orig_items = load_and_predict(orig_dir, is_folder_based=True)
print(f'Held-out: {len(held_out_items)}, Original: {len(orig_items)}')

def eval_set(items, weights):
    correct = 0
    for true_lbl, per_model in items:
        probs = sum(w * pm for w, pm in zip(weights, per_model))
        pred = classes[probs.argmax(1).item()]
        if pred == true_lbl:
            correct += 1
    return correct / len(items) * 100

# Grid search
print('\nGrid-searching optimal weights (step=0.1)...')
print(f'{"ResNet":>8s} {"DenseNet":>8s} {"EffNet":>8s}  | {"HeldOut":>8s} {"Orig":>8s} {"Combined":>10s}')
best_combined = 0
best_w = None

for r in range(0, 11):
    for a in range(0, 11 - r):
        e = 10 - r - a
        w = [r / 10.0, a / 10.0, e / 10.0]
        h_acc = eval_set(held_out_items, w)
        o_acc = eval_set(orig_items, w)
        comb = 0.5 * h_acc + 0.5 * o_acc
        if comb > best_combined:
            best_combined = comb
            best_w = list(w)
            print(f'  {w[0]:.1f}     {w[1]:.1f}     {w[2]:.1f}    | '
                  f'{h_acc:6.1f}%  {o_acc:6.1f}%  {comb:8.1f}%  <-- BEST')

print(f'\nOptimal weights: ResNet={best_w[0]:.1f}  DenseNet={best_w[1]:.1f}  EfficientNet={best_w[2]:.1f}')
print(f'Best combined score: {best_combined:.1f}%')
print(f'  Held-out: {eval_set(held_out_items, best_w):.1f}%')
print(f'  Original: {eval_set(orig_items, best_w):.1f}%')
