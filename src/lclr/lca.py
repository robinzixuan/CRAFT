import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import wandb

class ProjectionHead(nn.Module):
    def __init__(self, d_in, d=512): 
        super().__init__() 
        self.fn1= nn.Linear(d_in,1024)
        self.gelu = nn.GELU() 
        self.fn2 = nn.Linear(1024,d)

    def forward(self, x): 
        x = self.fn1(x)
        x = self.gelu(x)
        x = self.fn2(x)
        return F.normalize(x, dim=-1)

class SafetyHead(nn.Module):
    def __init__(self, d, num_classes=3): 
        super().__init__()
        self.fc=nn.Linear(d, num_classes)  # 3 classes: safe, unsafe, rethink
    def forward(self, z): 
        return self.fc(z)  # logits for 3 classes

def pool(h, attn): 
    return (h*attn.unsqueeze(-1)).sum(1)/(attn.sum(1,keepdim=True)+1e-6)

def aug_text(t):
    # cheap text aug: drop 5–10% tokens or swap a few; keep semantics
    toks=t.split()
    k=max(1,int(0.05*len(toks)))
    idx=set(random.sample(range(len(toks)),k))
    return " ".join([tok for i,tok in enumerate(toks) if i not in idx])

def info_nce(z1, z2, queue, tau=0.1):
    # z1,z2:[B,d]; queue:[Q,d] negatives
    z1=F.normalize(z1,dim=-1) 
    z2=F.normalize(z2,dim=-1)
    q=F.normalize(queue,dim=-1) if queue.numel()>0 else queue
    pos=(z1*z2).sum(-1)/tau
    if queue.numel()==0: 
        return -pos.mean()
    neg=torch.matmul(z1, q.T)/tau  # [B,Q]
    logits=torch.cat([pos.unsqueeze(-1), neg], -1)
    labels=torch.zeros(z1.size(0), dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels)

def contractive_loss(z, mu_safe, mu_unsafe, mu_rethink, margin=0.2):
    """
    Contractive learning loss that ensures all traces have potential for safety.
    This encourages traces to be closer to the safe prototype than unsafe prototype.
    
    Args:
        z: [B, d] embeddings
        mu_safe: [d] safe prototype
        mu_unsafe: [d] unsafe prototype  
        mu_rethink: [d] rethink prototype
        margin: margin for contractive loss
    
    Returns:
        contractive loss value
    """
    z_norm = F.normalize(z, dim=-1)
    mu_safe_norm = F.normalize(mu_safe, dim=-1)
    mu_unsafe_norm = F.normalize(mu_unsafe, dim=-1)
    mu_rethink_norm = F.normalize(mu_rethink, dim=-1)
    
    # Cosine similarities
    cos_safe = (z_norm * mu_safe_norm).sum(-1)  # [B]
    cos_unsafe = (z_norm * mu_unsafe_norm).sum(-1)  # [B]
    cos_rethink = (z_norm * mu_rethink_norm).sum(-1)  # [B]
    
    # Contractive loss: encourage all traces to be closer to safe than unsafe
    # This ensures every trace has potential for safety
    L_contractive = F.relu(margin - cos_safe + cos_unsafe).mean()
    
    # Additional loss: encourage rethink traces to be between safe and unsafe
    # This gives them potential to move towards safety
    L_rethink_guidance = F.relu(margin - cos_safe + cos_rethink).mean() + \
                        F.relu(margin - cos_rethink + cos_unsafe).mean()
    
    return L_contractive + 0.5 * L_rethink_guidance

@torch.no_grad()
def text_safe_prob(verifier, ver_tok, texts, device):
    if verifier is None: 
        return None
    batch=ver_tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    logits=verifier(**batch).logits
    if logits.size(-1)==1: 
        return torch.sigmoid(logits.squeeze(-1))
    return logits.softmax(-1)[..., -1]

def train_latent_contrastive(
    model_name, data, mu_safe, mu_unsafe, mu_rethink, epochs=3, bs=16,
    alpha=1.0, beta=0.5, gamma=1.0, delta=0.5, epsilon=0.3, margin=0.2, tau=0.1, queue_size=4096,
    verifier_name_or_path=None, wandb_run=None, output_dir=None, save_checkpoints=False, 
    checkpoint_freq=1, resume_from=None
):
    print(f"\n🚀 Starting LCA Training Process")
    print(f"📊 Training Configuration:")
    print(f"   - Model: {model_name}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {bs}")
    print(f"   - Data samples: {len(data)}")
    print(f"   - Loss weights: α={alpha}, β={beta}, γ={gamma}, δ={delta}, ε={epsilon}")
    print(f"   - Margin: {margin}, Temperature: {tau}")
    print(f"   - Wandb logging: {'Enabled' if wandb_run else 'Disabled'}")
    print(f"   - Save checkpoints: {'Enabled' if save_checkpoints else 'Disabled'}")
    print(f"   - Resume from: {resume_from or 'None'}")
    
    # Handle resume from checkpoint
    start_epoch = 0
    if resume_from:
        print(f"\n🔄 RESUMING FROM CHECKPOINT")
        print(f"   - Loading checkpoint from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cuda', weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        print(f"   - Resuming from epoch: {start_epoch}")
        print(f"   - Checkpoint timestamp: {checkpoint['timestamp']}")
    
    print(f"\n🔧 Initializing models...")
    print(f"   - Loading tokenizer from {model_name}...")
    tok=AutoTokenizer.from_pretrained(model_name)
    print(f"   - Loading language model from {model_name}...")
    lm =AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.bfloat16).eval().cuda()
    hidden_size = lm.config.hidden_size  # 动态获取模型的 hidden_size
    print(f"   - Initializing projection head (input_dim={hidden_size}, output_dim=512)...")
    proj=ProjectionHead(d_in=hidden_size).cuda()  # 使用动态值而非硬编码
    print(f"   - Initializing safety head (3-way classification)...")
    head=SafetyHead(512).cuda()
    print(f"   - Setting up AdamW optimizer (lr=1e-4, weight_decay=0.01)...")
    opt=torch.optim.AdamW(list(proj.parameters())+list(head.parameters()), lr=1e-4, weight_decay=0.01)
    
    # Load model states from checkpoint if resuming
    if resume_from:
        print(f"   - Loading model states from checkpoint...")
        proj.load_state_dict(checkpoint['projection_head_state'])
        head.load_state_dict(checkpoint['safety_head_state'])
        opt.load_state_dict(checkpoint['optimizer_state'])
        print(f"   - Model states loaded successfully")

    device=next(proj.parameters()).device
    print(f"   - Device: {device}")
    
    print(f"\n🎯 Setting up prototype vectors...")
    if resume_from:
        # Load prototype vectors from checkpoint
        mu_s = torch.tensor(checkpoint['mu_safe'], device=device, dtype=torch.float32)
        mu_u = torch.tensor(checkpoint['mu_unsafe'], device=device, dtype=torch.float32)
        mu_r = torch.tensor(checkpoint['mu_rethink'], device=device, dtype=torch.float32)
        print(f"   - Loaded prototype vectors from checkpoint")
    else:
        # Use provided prototype vectors
        mu_s=torch.tensor(mu_safe,   device=device, dtype=torch.float32)
        mu_u=torch.tensor(mu_unsafe, device=device, dtype=torch.float32)
        mu_r=torch.tensor(mu_rethink, device=device, dtype=torch.float32)
        print(f"   - Using provided prototype vectors")
    
    print(f"   - Safe prototype shape: {mu_s.shape}, norm: {mu_s.norm().item():.4f}")
    print(f"   - Unsafe prototype shape: {mu_u.shape}, norm: {mu_u.norm().item():.4f}")
    print(f"   - Rethink prototype shape: {mu_r.shape}, norm: {mu_r.norm().item():.4f}")

    # optional frozen verifier
    if verifier_name_or_path:
        print(f"\n🛡️ Setting up safety verifier...")
        print(f"   - Loading verifier from {verifier_name_or_path}...")
        vtok=AutoTokenizer.from_pretrained(verifier_name_or_path)
        vmod=AutoModelForSequenceClassification.from_pretrained(verifier_name_or_path).to(device).eval()
        for p in vmod.parameters(): 
            p.requires_grad=False
        print(f"   - Verifier loaded and frozen")
    else:
        print(f"\n⚠️ No verifier specified - skipping distillation loss")
        vtok=vmod=None

    # negatives queue (MoCo-style)
    print(f"\n📚 Initializing negative queue (MoCo-style, size={queue_size})...")
    queue=torch.empty(0,512, device=device)

    def batch_iter(lst, n): 
        for i in range(0, len(lst), n): 
            yield lst[i:i+n]

    print(f"\n🏃 Starting training loop...")
    total_batches = (len(data) + bs - 1) // bs
    print(f"   - Total batches per epoch: {total_batches}")
    print(f"   - Starting from epoch: {start_epoch + 1}")
    
    # Track best loss for checkpoint saving
    best_loss = float('inf')
    
    for ep in range(start_epoch, epochs):
        print(f"\n📈 EPOCH {ep+1}/{epochs}")
        print(f"   - Shuffling data...")
        random.shuffle(data)
        
        epoch_losses = []
        batch_idx = 0
        
        for batch in batch_iter(data, bs):
            batch_idx += 1
            texts=[b["reasoning_trace"] for b in batch]
            # Convert labels to indices: safe=0, unsafe=1, rethink=2
            label_map = {"safe": 0, "unsafe": 1, "rethink": 2}
            labels=torch.tensor([label_map[b["label"]] for b in batch], device=device).long()
            
            # Count labels in this batch
            safe_count = (labels == 0).sum().item()
            unsafe_count = (labels == 1).sum().item()
            rethink_count = (labels == 2).sum().item()
            
            if batch_idx % 10 == 1 or batch_idx <= 5:  # Print first 5 batches and every 10th batch
                print(f"   📦 Batch {batch_idx}/{total_batches} (size={len(batch)}, safe={safe_count}, unsafe={unsafe_count}, rethink={rethink_count})")

            def encode(txts):
                enc=tok(txts, return_tensors="pt", truncation=True, max_length=3548, padding=True).to(device)
                with torch.no_grad(): 
                    h=lm(**enc).hidden_states[-1]
                z_raw=pool(h, enc["attention_mask"])
                return proj(z_raw)  # [B,512]

            # Encode original and augmented texts
            z1=encode(texts)
            z2=encode([aug_text(t) for t in texts])

            # Compute losses
            cos_s=(F.normalize(z1,dim=-1)*F.normalize(mu_s,dim=-1)).sum(-1)
            cos_u=(F.normalize(z1,dim=-1)*F.normalize(mu_u,dim=-1)).sum(-1)
            cos_r=(F.normalize(z1,dim=-1)*F.normalize(mu_r,dim=-1)).sum(-1)
            L_proto=F.relu(margin - cos_s + cos_u).mean()

            L_inst=info_nce(z1, z2, queue, tau)

            logits=head(z1)
            L_ce=F.cross_entropy(logits, labels)  # 3-way classification

            # Contractive learning loss for safety potential
            L_contractive = contractive_loss(z1, mu_s, mu_u, mu_r, margin)

            if vmod:
                p_text=text_safe_prob(vmod, vtok, texts, device)
                p_lat = F.softmax(logits, dim=-1)[:, 0].clamp(1e-5, 1-1e-5)  # safe class probability
                L_distill=F.kl_div(p_lat.log(), p_text, reduction="batchmean")
            else:
                L_distill=torch.tensor(0.0, device=device)

            # Compute weighted total loss
            loss = alpha*L_proto + beta*L_inst + gamma*L_ce + delta*L_distill + epsilon*L_contractive
            
            # Track losses for this batch
            batch_losses = {
                'total': loss.item(),
                'proto': L_proto.item(),
                'inst': L_inst.item(),
                'ce': L_ce.item(),
                'contractive': L_contractive.item(),
                'distill': L_distill.item()
            }
            epoch_losses.append(batch_losses)
            
            # Print detailed loss info for first few batches and every 10th batch
            if batch_idx % 10 == 1 or batch_idx <= 5:
                print(f"      💡 Losses: Total={loss.item():.4f}, Proto={L_proto.item():.4f}, Inst={L_inst.item():.4f}, CE={L_ce.item():.4f}, Contractive={L_contractive.item():.4f}, Distill={L_distill.item():.4f}")
            
            # Log to wandb if enabled
            if wandb_run:
                wandb.log({
                    'epoch': ep + 1,
                    'batch': batch_idx,
                    'batch_loss': loss.item(),
                    'batch_proto_loss': L_proto.item(),
                    'batch_inst_loss': L_inst.item(),
                    'batch_ce_loss': L_ce.item(),
                    'batch_contractive_loss': L_contractive.item(),
                    'batch_distill_loss': L_distill.item(),
                    'learning_rate': opt.param_groups[0]['lr']
                })
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update prototypes by EMA on current batch means
            with torch.no_grad():
                m=0.98
                old_mu_s_norm = mu_s.norm().item()
                old_mu_u_norm = mu_u.norm().item()
                old_mu_r_norm = mu_r.norm().item()
                
                mu_s.mul_(m).add_((1-m)*z1[labels==0].mean(0) if (labels==0).any() else 0)  # safe=0
                mu_u.mul_(m).add_((1-m)*z1[labels==1].mean(0) if (labels==1).any() else 0)  # unsafe=1
                mu_r.mul_(m).add_((1-m)*z1[labels==2].mean(0) if (labels==2).any() else 0)  # rethink=2

                # Update queue
                q_new = z2.detach()
                if queue.numel()==0: 
                    queue=q_new
                else:
                    need=max(0, queue.size(0)+q_new.size(0)-queue_size)
                    if need>0: queue=queue[need:]
                    queue=torch.cat([queue, q_new], dim=0)
                
                # Print prototype updates for first few batches
                if batch_idx <= 3:
                    print(f"      🔄 Prototype updates: Safe {old_mu_s_norm:.4f}→{mu_s.norm().item():.4f}, Unsafe {old_mu_u_norm:.4f}→{mu_u.norm().item():.4f}, Rethink {old_mu_r_norm:.4f}→{mu_r.norm().item():.4f}")
                    print(f"      📚 Queue size: {queue.size(0)}")

        # Print epoch summary
        avg_losses = {k: np.mean([b[k] for b in epoch_losses]) for k in epoch_losses[0].keys()}
        print(f"\n📊 EPOCH {ep+1} SUMMARY:")
        print(f"   - Average Loss: {avg_losses['total']:.4f}")
        print(f"   - Proto: {avg_losses['proto']:.4f}, Inst: {avg_losses['inst']:.4f}, CE: {avg_losses['ce']:.4f}")
        print(f"   - Contractive: {avg_losses['contractive']:.4f}, Distill: {avg_losses['distill']:.4f}")
        print(f"   - Final Prototype Norms: Safe={mu_s.norm().item():.4f}, Unsafe={mu_u.norm().item():.4f}, Rethink={mu_r.norm().item():.4f}")
        print(f"   - Queue size: {queue.size(0)}")
        
        # Log epoch metrics to wandb
        if wandb_run:
            wandb.log({
                'epoch': ep + 1,
                'epoch_loss': avg_losses['total'],
                'epoch_proto_loss': avg_losses['proto'],
                'epoch_inst_loss': avg_losses['inst'],
                'epoch_ce_loss': avg_losses['ce'],
                'epoch_contractive_loss': avg_losses['contractive'],
                'epoch_distill_loss': avg_losses['distill'],
                'prototype_safe_norm': mu_s.norm().item(),
                'prototype_unsafe_norm': mu_u.norm().item(),
                'prototype_rethink_norm': mu_r.norm().item(),
                'queue_size': queue.size(0)
            })
        
        # Save checkpoint if enabled
        if save_checkpoints and (ep + 1) % checkpoint_freq == 0:
            print(f"\n💾 SAVING CHECKPOINT")
            is_best = avg_losses['total'] < best_loss
            if is_best:
                best_loss = avg_losses['total']
                print(f"   - New best loss: {best_loss:.4f}")
            
            # Import the save_checkpoint function from train_lca
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from train_lca import save_checkpoint
            
            save_checkpoint(
                epoch=ep + 1,
                proj_state=proj.state_dict(),
                head_state=head.state_dict(),
                mu_safe=mu_s.detach().cpu().numpy(),
                mu_unsafe=mu_u.detach().cpu().numpy(),
                mu_rethink=mu_r.detach().cpu().numpy(),
                optimizer_state=opt.state_dict(),
                output_dir=output_dir,
                is_best=is_best
            )

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"   - Final prototype norms: Safe={mu_s.norm().item():.4f}, Unsafe={mu_u.norm().item():.4f}, Rethink={mu_r.norm().item():.4f}")
    print(f"   - Final queue size: {queue.size(0)}")
    print(f"   - Returning trained model states...")
    
    return proj.state_dict(), head.state_dict(), mu_s.detach().cpu().numpy(), mu_u.detach().cpu().numpy(), mu_r.detach().cpu().numpy()
