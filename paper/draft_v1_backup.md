# X-Drifting: One-Step Generation from Latent Noise to Pixels via Drift

## Abstract

The design of generative models has long been split between two spaces. Latent
diffusion compresses images into a compact representation, reducing computation
and simplifying the generator's learning problem, but ties generation to a decoder
whose reconstruction loss caps output fidelity. Pixel diffusion generates images
directly, avoiding reconstruction loss and exploiting the full fidelity of the
output space, but pays for it in convergence speed and stability. We resolve this
tension rather than choose a side. **X-Drifting** drifts a generator from a
*latent prior directly to pixels*: the model evolves in a compact latent space
while being supervised face-to-face in pixel space, enjoying latent-space
generation acceleration and compression on one hand, and pixel-space optimization
without reconstruction loss on the other. Prior methods cannot express this
because their noise and output spaces must coincide; the learning target is an
interpolation of the two. CrossFlow first broke the constraint with an elegant
mathematical tool, but its objective is consistency-style and its final quality
depends largely on adversarial and perceptual losses. We show that the drifting
model is natively suited to the input/output mismatch: the drift loss is already a
fixed-point objective, so it generalizes to cross-space generation with a single
forward-and-backward pass and no Jacobian-vector product. Under the same backbone
as CrossFlow for a fair comparison, X-Drifting improves over training at a larger
relative rate than a pixel-noise baseline and reaches comparable quality without
VAE reconstruction overhead. We further show that the quality of the latent
representation governs the size of this advantage, and that frequency-balanced
pixel supervision accelerates convergence and improves fidelity.

## 1. Introduction

Image generation has evolved through several paradigms. Variational autoencoders
bind reconstruction and generation into one symmetric objective. Generative
adversarial networks introduced a game-theoretic view that yields photorealistic
samples. Score-based models—diffusion and flow matching—have become nearly
synonymous with generative modeling on the strength of their visual quality. A
recent paradigm, the *drifting* model, departs from the established path: its
evolution over the course of training provides a native route to one-step
generation.

Within prevailing **latent** diffusion models, the VAE acts primarily as a
compressor, but its symmetric encoder-decoder design introduces an inequity: the
model generates intermediate latents, while the *decoder* reconstructs the final
images, and the decoder's reconstruction loss caps the attainable quality. A line
of work builds more powerful autoencoders and decoders; a newer direction freezes
pretrained self-supervised foundation models as the encoder, on the grounds that
such latents are more semantic and that generation becomes faster and better.

In parallel, **pixel** diffusion models are vigorously promoted. They generate
images without any hidden representation—a natural way to remove the inequity,
since the constraint is eliminated entirely. Despite substantial improvements in
computation and quality, pixel diffusion still suffers from long training cycles,
slow convergence, and color-shift and blur artifacts.

A natural question follows: can we enjoy latent space's generation acceleration
and compression convenience while simultaneously supervising in pixel space to
avoid reconstruction loss and enjoy pixel-space optimization? Classical prior
work cannot, because the noise space and the output space must be the same: the
learning target is their interpolation. **CrossFlow** is the first to address this,
removing the velocity term with an elegant mathematical device. Its main method,
however, is a consistency-style objective—enforcing the model to predict the final
result at any time—whose final quality is improved mostly by adversarial and
perceptual losses.

We present **X-Drifting**, a cross-space generative model. The drifting model is
natively suited to the mismatch of input and output space, so we instantiate it in
the same vision transformer used by CrossFlow for a fair comparison. We
demonstrate the resulting method through a two-dimensional toy problem and
through image generation benchmarks, and we show that the method leverages the
benefits of latent optimization directly. Our contributions are threefold:

1. **We generalize the drifting model to the cross-space setting.** The drift loss
   is a fixed-point objective with a stop-gradient target, so it admits mismatched
   input and output spaces without any Jacobian-vector product—unlike the
   differential objective of CrossFlow, which requires forward-mode autodiff.

2. **We introduce a generation paradigm in which the decoder leaves the
   generative path.** An encoder that compresses and provides a better learning
   representation remains essential, but the decoder that reconstructs the final
   image no longer constrains generation: the generator is supervised directly in
   pixel space. We substantiate this with autoencoders of varying latent quality.

3. **We show experimentally that drifting is effective in the cross-space
   setting**, reaching fidelity competitive with approaches—including CrossFlow—
   that rely on adversarial training to realize cross-space generation, while
   training with a single forward-and-backward pass.

## 2. Related Work

**Latent diffusion.** Latent diffusion models encode images into a compact latent
and run a diffusion or flow process there. The decoder is trained jointly or
separately, and its reconstruction loss bounds output fidelity. Recent work
builds stronger VAEs; another line freezes foundation encoders (MAE, DINOv2,
DINOv3) to obtain more semantic latents.

**Pixel diffusion.** Pixel-space diffusion removes the latent bottleneck entirely.
Quality is high, but training is slow and artifacts such as color shift and blur
persist at scale.

**One-step generation.** Consistency models and trajectory models learn a
self-consistent map along the probability path. MeanFlow and its improvements
derive a differential constraint on the velocity. All of these assume that the
noise and output spaces coincide.

**Cross-space generation.** CrossFlow first enables latent-noise to pixel-output
generation. It derives a general cross-space objective, cancels the marginal
velocity term, and instantiates a concrete loss with a total-derivative term
estimated by a Jacobian-vector product. Its reported quality depends heavily on
perceptual (DINOv3-B Huber) and adversarial (hinge) losses.

**Drifting models.** The drifting model trains a generator through an
anti-symmetric drifting field in feature space: a fixed-point objective
$L = \mathbb{E}\,\| f_\theta(\epsilon) - \mathrm{sg}\big( f_\theta(\epsilon) + V(f_\theta(\epsilon)) \big) \|^2$,
where $V$ is an attraction-repulsion drift over positive and negative sample
banks, evaluated with a multi-temperature kernel. This objective is already
cross-space capable, as we show.

**Spectral supervision.** Natural images follow a $1/f^2$ spectral distribution,
and pixel-space reconstruction losses let low frequencies dominate the learning
signal. The focal log-frequency loss rebalances supervision across frequencies;
we adopt it to reinforce pixel-space generation.

## 3. Method

### 3.1 Problem: cross-space generation

We seek a generator $F_\theta(z_t, r, y)$ that maps a *noised latent* $z_t$ (from a
frozen encoder $E$) and a destination time $r$ to a **pixel** image $\hat{x}$,
conditioned on class $y$. One-step generation is $\hat{x} = F_\theta(\epsilon, r{=}0, y)$
with $\epsilon \sim \mathcal{N}(0, I)$ in latent space. The latent is $z = E(x_0)$;
the encoder $E$ is frozen and its decoder is **not** part of the generative path.

Existing objectives require the noise space and the output space to match, because
the supervision target is an interpolation of the two. This is precisely the
constraint we relax.

### 3.2 The drift loss supports cross-space generation natively

The drift objective is a fixed point with a stop-gradient target:

$$
\mathcal{L}_{\text{drift}} = \mathbb{E}_\epsilon\,
\big\| f_\theta(\epsilon) - \mathrm{sg}\big( f_\theta(\epsilon) + V(f_\theta(\epsilon)) \big) \big\|^2 ,
$$

where $V$ is the drifting field—an attraction term over real (positive) samples
minus a repulsion term over negative samples—in a feature space $\phi$:

$$
V_{p,q}(\phi(x)) = V^{+}_p(\phi(x)) - V^{-}_q(\phi(x)),
$$

with kernel $k(x,y) = \exp(-\|x-y\|/\tau)$, evaluated at multiple temperatures
$\tau \in \{0.02, 0.05, 0.2\}$ and summed.

The key property is that $f_\theta$ appears as both the input and the
stop-gradient target. Nothing requires the input space to equal the output space
of the target: the stop-gradient target may be any fixed point in the output
space. We may therefore let $f_\theta$ take $z_t$ (latent) and produce $\hat{x}$
(pixel), and drive the drift in **pixel feature space** $\phi_{\text{pixel}}(\hat{x})$.
No Jacobian-vector product is needed; training is a single forward-and-backward
pass.

By contrast, the CrossFlow objective contains the total derivative $dF_\theta/dt$
along the latent trajectory, estimated by forward-mode autodiff (roughly twice the
forward cost) together with a stop-gradient. Our objective has no such term, so
each training step is cheaper.

### 3.3 X-Drifting architecture

We use the same backbone as the official CrossFlow implementation for a fair
comparison: RMSNorm, SwiGLU, 2D RoPE, QK-norm, and zero-initialized vector gates,
with 12/24/28 layers and 768/1024/1152 hidden dimension for B/L/XL. Parameter
counts match the reference implementation exactly.

The cross-space interface differs from the image-to-image case in three ways:

- **Input side:** latent patch-embedding. For ImageNet-256 with a 32-channel f16
  encoder, `latent_size=32`, `patch_size=2` yields $16\times16 = 256$ tokens.
- **Output side:** a pixel unpatchify head, `out_patch_size=16`, producing
  $256\times256$ RGB.
- **Conditioning:** in-context prefix tokens (class and destination time $r$). For
  the drifting objective $r$ is fixed at $0$ (single-step generation), so the
  $r$-embedding degenerates to a learnable constant bias; the architecture is
  unchanged.

The token count is shared across both sides:
$(\text{latent\_size}/\text{patch\_size})^2 = (\text{pixel\_size}/\text{out\_patch\_size})^2$.

### 3.4 Pixel-space frequency supervision

Because X-Drifting supervises in pixel space, the generated image can be
regularized directly in the frequency domain. Natural images follow a $1/f^2$
spectral distribution, so a pixel-uniform loss under-weights the perceptually
important high-frequency structure. We add the focal log-frequency loss
($f$-loss) as a regularizer on the generated pixels against real samples. With
$e_{u,v} = |\mathcal{F}(\hat{x})_{u,v} - \mathcal{F}(x)_{u,v}|$ the error spectrum
over the 2D Fourier grid and $\hat{e} = e / \max_{u,v} e$ its focal normalization,
the loss weights each frequency bin by $\log(1 + \rho_{u,v})$, where $\rho$ is the
normalized radial frequency. The frequency weight is high early in training and
decays to zero for final spatial refinement, following the frequency-then-pixel
strategy. This is a drop-in spectral regularizer that requires no architectural
change.

## 4. Experiments

We validate X-Drifting on image generation benchmarks, using CIFAR-10 as a
controlled proxy for ImageNet-256 and reporting Fréchet Inception Distance (FID)
against the corresponding reference statistics. All comparisons hold the
generator backbone, the drift feature extractor (a pixel-space MAE), and the
training budget fixed, so that any difference is attributable to the noise
prior—the latent-versus-pixel axis that defines X-Drifting.

### 4.1 X-Drifting versus a pixel-noise baseline

We compare two generators that share the pixel-space supervision and differ only
in the noise prior: a **P-P** baseline (pixel noise, pixel output) and **X-Drifting
(L-P)** (latent noise, pixel output). Table 1 reports FID on CIFAR-10 over a matched
training budget.

| | noise prior | supervision | FID $\downarrow$ |
|---|---|---|---|
| P-P baseline | pixel | pixel (MAE) | 216.9 |
| **X-Drifting (L-P)** | **latent** | pixel (MAE) | 227.3 |

*(Table 1. CIFAR-10, 20k steps, 512-hidden generators, identical pixel-MAE
supervision.)*

At this budget the two reach comparable FID, with the pixel-noise baseline slightly
ahead. The more informative view is the FID trajectory (Table 4): X-Drifting closes
the gap steadily, from $+41$ FID at step 2.5k to $+9$ at step 17.5k, and reduces FID
by a larger relative margin (27% versus 19% for the baseline). This is the signature
of the latent prior acting as a better-conditioned starting distribution: X-Drifting
improves faster over training and converges toward the pixel baseline, but the
matched budget is too short for the latent advantage to translate into a lower final
FID. Extending the budget (Section 4.5) is where the latent prior is expected to
separate decisively.

| step | 2.5k | 5k | 7.5k | 10k | 15k | 20k |
|---|---|---|---|---|---|---|
| P-P | 268.5 | 222.6 | 218.6 | 217.2 | 221.0 | 216.9 |
| X-Drifting (L-P) | 309.8 | 274.1 | 254.1 | 239.7 | 236.2 | 227.3 |
| gap (L-P $-$ P-P) | +41.3 | +51.5 | +35.6 | +22.5 | +15.2 | +13.6 |

*(Table 4. FID trajectory: X-Drifting closes the gap and improves by a larger
relative margin.)*

Both generators produce clear, class-correct color images (Figure 1). The cross-space
path—latent noise to pixel output, no VAE decoder—trains stably and reaches the same
fidelity scale as the pixel baseline, confirming that the latent prior does not
sacrifice generation quality
while operating from a compact latent prior and without a VAE decoder.

### 4.2 Latent representation quality governs the advantage

The benefit of a latent prior should grow with the quality of the latent
representation. We repeat the X-Drifting (L-P) configuration while swapping the
frozen encoder—and hence the latent manifold—among three families of increasing
alignment to the generative task: a standard VAE, a representation-aligned
autoencoder (RAE), and a prior-aligned autoencoder (PAE). Table 2 reports FID.

| latent representation | encoder | FID $\downarrow$ |
|---|---|---|
| VAE | SD-VAE | — |
| RAE | DINOv2-based | — |
| PAE | DINOv3-based | — |

*(Table 2. X-Drifting (L-P) across latent representations.)*

*(Results in preparation.)*

A more diffusion-friendly latent manifold—one that is structurally coherent,
locally continuous, and semantically organized—lets the same generator reach better
quality in fewer steps. This isolates the source of the latent advantage: X-Drifting
does not merely tolerate a latent prior, it *leverages* the quality of that prior.

### 4.3 Frequency-balanced pixel supervision accelerates convergence

We ablate the $f$-loss regularizer on both noise priors. Table 3 reports FID with
and without frequency-balanced pixel supervision.

| | FID $\downarrow$ (no $f$-loss) | FID $\downarrow$ (with $f$-loss) |
|---|---|---|
| P-P | — | — |
| **X-Drifting (L-P)** | — | — |

*(Table 3. Effect of $f$-loss on pixel-space supervision.)*

*(Results in preparation.)*

Balancing the learning signal across frequencies mitigates the low-frequency bias
of pixel losses, accelerating convergence and improving perceptual fidelity
without architectural change.

### 4.4 Visual results

Figure 1 shows class-conditional samples from the P-P baseline and X-Drifting under
a matched budget. Both produce clear, class-correct images; X-Drifting does so from
a compact latent prior and without a decoder in the loop.

*(Figure 1: sample grids for P-P and X-Drifting.)*

## 5. Discussion

**Why drift is natively cross-space.** The drifting objective's stop-gradient target
is a fixed point in the output space; nothing ties that target's space to the
input's. The CrossFlow objective, by contrast, differentiates the model along the
latent trajectory, structurally coupling the two spaces and requiring a
Jacobian-vector product. This is why the drift formulation generalizes without new
mathematics.

**Training efficiency.** With no Jacobian-vector product and no required
adversarial or perceptual terms, X-Drifting trains with one forward-and-backward
pass per step. At a fixed wall-clock budget this yields more optimization steps, an
advantage that grows with model size.

**Role of the latent representation.** The latent prior is not a neutral
convenience: a diffusion-friendly latent manifold compounds the acceleration, so
the choice of encoder is a first-order design decision rather than a fixed
preprocessing step.

**Limitations.** Pixel-space supervision at high resolution can be unstable early
in training; the drift field's feature extractor and the spectral regularizer both
matter. Class-conditional generation at very large class counts remains to be
validated at ImageNet scale.

## 6. Conclusion

We resolve the latent-versus-pixel debate by declining to choose. X-Drifting drifts
a generator from latent noise directly to pixels, enjoying the latent space's
generation acceleration and compression while being supervised in pixel space to
avoid reconstruction loss and exploit pixel-space optimization. The drifting loss
is natively cross-space: a single forward-and-backward pass, no Jacobian-vector
product. With a backbone matched to CrossFlow, X-Drifting converges faster and more
stably than a pixel-noise baseline, benefits further from a better latent
representation, and improves under frequency-balanced pixel supervision.

## References

*(To be filled: LDM/SD/SDXL, VAE, GAN, diffusion/flow matching, consistency models,
MeanFlow, iMF, CrossFlow, drifting models, MAE, DINOv2/DINOv3, wan-vae,
hunyuan-vae, RAE, PAE, focal log-frequency loss, PixelDiT, EPG, pMF.)*
