import torch as t
import torch.nn.functional as F

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))

def reg_pick_embeds(embeds_list):
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss

def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()

def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


# ============================================================================
# Base Model Loss Computation Utilities
# ============================================================================

def detect_base_model_type(inter):
    """
    Detect the type of base model from intermediate outputs.
    
    Args:
        inter: dict with intermediate outputs from base model
    
    Returns:
        str: 'cvga', 'l_diffrec', or 'mult_vae' (default)
    """
    # CVGA: has recon_x already computed
    if 'recon_x' in inter:
        return 'cvga'
    
    # L-DiffRec: has diffusion_loss
    if 'diffusion_loss' in inter and inter['diffusion_loss'] is not None:
        return 'l_diffrec'
    
    # Default: Mult-VAE or other VAE-based models
    return 'mult_vae'


def compute_kld(mu_src, mu_llm, logvar_src, logvar_llm, inter, base_model_type=None):
    """
    Compute KLD loss based on base model type.
    
    Args:
        mu_src: [batch_size, latent_dim] mean from collaborative space
        mu_llm: [batch_size, latent_dim] mean from LLM space
        logvar_src: [batch_size, latent_dim] logvar from collaborative space
        logvar_llm: [batch_size, latent_dim] logvar from LLM space
        inter: dict with intermediate outputs from base model
        base_model_type: str, optional. If None, will be auto-detected.
    
    Returns:
        KLD: scalar KLD loss
    """
    if base_model_type is None:
        base_model_type = detect_base_model_type(inter)
    
    if base_model_type == 'cvga':
        # CVGA: use mu_src and logvar_src directly (no mu_llm addition)
        # CVGA's architecture is different: it uses GNN encoding, not standard VAE addition
        KLD = - 0.5 * t.mean(t.sum(1 + logvar_src - mu_src.pow(2) - logvar_src.exp(), dim=1))
        return KLD
    
    elif base_model_type == 'l_diffrec':
        # L-DiffRec: diffusion-based model, does NOT use KLD
        # L-DiffRec is a diffusion-based generative model, not a VAE
        # It does NOT have KLD in its loss function - diffusion process itself provides regularization
        # Return 0.0 to maintain interface compatibility, but semantically L-DiffRec has no KLD
        device = mu_src.device
        return t.tensor(0.0, device=device)
    
    else:
        # Mult-VAE or other VAE-based models: standard VAE KLD
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm
        KLD = - 0.5 * t.mean(t.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return KLD


def compute_reconstruction_loss(recon_x, data):
    """
    Compute reconstruction loss (BCE).
    
    Args:
        recon_x: [batch_size, item_num] reconstructed interactions
        data: [batch_size, item_num] ground truth interactions
    
    Returns:
        BCE: scalar reconstruction loss
    """
    return - t.mean(t.sum(F.log_softmax(recon_x, 1) * data, -1))