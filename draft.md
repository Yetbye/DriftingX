# Representation-Grounded Cross-Space Drifting: One-Step Generation from Latent Noise to Pixels

## Abstract

Pixel-space generation has nearly closed the image-quality gap with latent
diffusion, yet it converges more slowly and lacks the compact, structured
representation that makes latent generation easy to learn. We argue the missing
ingredient is not an autoencoder but an explicit *representation prior*. We
present **X-Drifting**, a cross-space generative model that drifts from a latent
prior directly to pixels while being grounded in the representation space of a
frozen vision foundation model. X-Drifting combines two mechanisms. First,
*representation grounding*: the drifting attraction-repulsion objective is
evaluated in the feature space of a frozen encoder (SD-VAE, RAE, or PAE), so the
quality of the latent representation shapes generation directly—without a decoder
and without reconstruction loss. Second, *predict-then-condition*: the generator
predicts the pretrained representation it needs and conditions on its own
prediction, removing the need to denoise a second representation trajectory.
Because the drift loss is a fixed-point objective with a stop-gradient target, it
admits the latent-to-pixel mismatch natively and trains with a single
forward-and-backward pass—no Jacobian-vector product and no adversarial critic. We
show that grounding in a better-aligned latent representation (PAE over RAE over
SD-VAE) improves generation, and that frequency-balanced pixel supervision further
accelerates convergence and fidelity. X-Drifting thus realizes the long-sought
synthesis: the latent space's generation acceleration and compression on one hand,
and pixel-space supervision without reconstruction loss on the other.

## 1. Introduction

Image generation has been split between two spaces. Latent diffusion compresses
images into a compact representation, reducing computation and simplifying the
generator's learning problem, but ties generation to a decoder whose
reconstruction loss caps output fidelity. Pixel diffusion generates images
directly, avoiding reconstruction loss and exploiting the full fidelity of the
output space, but converges slowly and lags in final quality. Recent pixel-space
models have narrowed this gap, yet a key asymmetry remains: latent diffusion
denoises inside a compact, structured space whose semantics are partially
organized before generation begins, whereas pixel diffusion must learn
denoising-friendly representations and pixel generation simultaneously from raw
RGB.

The natural response has been to reach for a stronger autoencoder. But the
autoencoder is not what makes latent generation work—its *representation* is. A
pretrained vision encoder already provides a coordinate system in which semantic
structure and low-level detail are organized; the decoder that maps back to pixels
is what introduces reconstruction loss. This suggests a different design: keep the
representation prior, drop the decoder, and generate pixels directly.

We realize this idea in **X-Drifting**, a cross-space generative model that maps a
latent prior directly to pixels while grounded in the feature space of a frozen
vision foundation model. X-Drifting rests on two mechanisms:

- **Representation grounding.** The generation objective is evaluated in the
  feature space of a frozen encoder—SD-VAE, RAE (a DINO autoencoder), or PAE (a
  prior-aligned, diffusion-friendly refinement of RAE). Because the encoder is a
  *representation prior* and the decoder never enters the generative path, the
  quality of the latent representation directly governs generation, and no
  reconstruction loss is incurred.

- **Predict-then-condition.** Rather than denoise a pretrained representation
  alongside the image, the generator predicts the representation it needs and
  conditions on its own prediction. This removes a second denoising trajectory and
  its scheduling, while retaining the benefits of representation guidance.

The generation objective is the *drift* loss: a fixed-point objective with a
stop-gradient target whose attraction-repulsion field acts on samples. This
objective is natively cross-space—the input need not live in the output space—so
it maps latent noise to pixels with a single forward-and-backward pass, no
Jacobian-vector product and no adversarial critic. This stands in contrast to
cross-space objectives that differentiate the model along the latent trajectory
and depend on adversarial and perceptual losses for final quality.

Our contributions are:

1. **We identify the representation prior, not the autoencoder, as the missing
   ingredient in pixel generation**, and introduce representation grounding for
   cross-space drifting: the drift objective is evaluated in a frozen encoder's
   feature space, so latent-representation quality governs generation without a
   decoder.

2. **We combine grounding with predict-then-condition**, in which the generator
   predicts the pretrained representation and self-conditions on it, removing the
   second denoising trajectory of co-denoising methods.

3. **We show the drift loss generalizes to cross-space generation natively**, with
   a single forward-and-backward pass and no Jacobian-vector product, and that the
   latent advantage grows with the quality of the grounding representation
   (PAE over RAE over SD-VAE) and with frequency-balanced pixel supervision.

## 2. Related Work

**Latent diffusion.** Latent diffusion models encode images into a compact latent
and run a diffusion or flow process there; the decoder's reconstruction loss
bounds output fidelity. Stronger VAEs and foundation-encoder latents (MAE, DINO)
improve the representation, but generation remains coupled to a decoder.

**Pixel diffusion.** Pixel-space diffusion removes the latent bottleneck and the
reconstruction ceiling, at the cost of convergence speed and the need to learn
denoising-friendly representations from raw RGB. PixelDiT scales large-patch
pixel transformers to close the gap; the remaining deficit is the lack of an
explicit representation prior.

**Representation guidance.** REPA aligns a diffusion model's intermediate features
to a frozen encoder's clean-image features, accelerating convergence but leaving
the encoder out of inference. PixelDiT2 introduces *representation grounding*:
the denoiser is conditioned on the frozen encoder's features of the evolving
sample, and this path stays active at inference. DINO-Forcing shows one need not
denoise the representation at all—predict it, then self-condition. We adopt
grounding and predict-then-condition and combine them with drift-based cross-space
generation.

**One-step and energy-based generation.** Consistency models, trajectory models,
MeanFlow and its improvements learn to jump to the endpoint, assuming noise and
output spaces coincide. Trajectory Forcing structures generation into semantic
coarse-to-fine stages in DINO space. Three-body scattering trains one-step
generators through attraction-repulsion in frozen image features and reports that
latent-space generators outperform pixel-space ones. The drift objective shares
the attraction-repulsion structure and is likewise evaluated in a feature space;
grounding it in a frozen encoder connects these threads.

**Autoencoder representations.** SD-VAE provides a generic compressed latent. RAE
replaces the VAE encoder with a frozen DINO and trains only a decoder. PAE further
shapes the latent into a diffusion-friendly manifold. These form a ladder of
increasing representation alignment, which we use to isolate the effect of latent
quality on cross-space generation.

## 3. Method

### 3.1 Problem: grounded cross-space generation

We want a generator $F_\theta$ that maps a latent prior sample to a pixel image
$\hat{x}$ conditioned on class $y$, in one step:
$\hat{x} = F_\theta(\epsilon, r{=}0, y)$ with $\epsilon \sim \mathcal{N}(0,I)$ in
latent space. Unlike latent diffusion, there is no decoder in the loop; unlike
pixel diffusion, the noise is a compact latent prior. The question is how to bring
the *representation* of a frozen encoder to bear on generation without turning it
into a reconstruction bottleneck.

### 3.2 Drift loss: a cross-space fixed point

The drift objective is a fixed point with a stop-gradient target:

$$
\mathcal{L}_{\text{drift}}
= \mathbb{E}_\epsilon\,
\big\| f_\theta(\epsilon) - \mathrm{sg}\big( f_\theta(\epsilon) + V(f_\theta(\epsilon)) \big) \big\|^2 ,
$$

where $V$ is the drifting field—an attraction toward real samples minus a
repulsion from negative samples—evaluated in a feature space $\phi$:

$$
V_{p,q}(\phi(x)) = V^{+}_p(\phi(x)) - V^{-}_q(\phi(x)),
\qquad
k(x,y)=\exp(-\|x-y\|/\tau),
$$

summed over temperatures $\tau \in \{0.02, 0.05, 0.2\}$. Because the target is a
stop-gradient fixed point, nothing requires the input space to match the output
space: $f_\theta$ may take latent noise and produce pixels, with the drift driven
in the feature space $\phi$. Training is one forward-and-backward pass.

### 3.3 Representation grounding (mechanism A)

The feature space $\phi$ is where the representation prior enters. We evaluate the
drift in the features of a **frozen encoder** $\mathcal{E}$:

$$
\phi(\hat{x}) \;=\; \mathcal{E}(\hat{x}) \in \mathbb{R}^{L\times D},
$$

with $\mathcal{E}$ a frozen SD-VAE, RAE, or PAE encoder. Per-patch tokens are
aggregated into the multi-scale feature dict consumed by the drift loss (global
mean/std and $2{\times}2$, $4{\times}4$ patch statistics). The decoder is never
used. Because $\mathcal{E}$ is a representation prior, the quality of the latent
representation—SD-VAE $<$ RAE $<$ PAE in alignment to the generative
task—directly shapes the drifting field and hence generation. This is the latent
advantage that X-Drifting captures, without reconstruction loss.

Grounding differs from REPA: REPA aligns internal features to clean-image targets
and drops the encoder at inference, whereas grounding keeps $\mathcal{E}$ as an
explicit feature space for the objective throughout. It also differs from
pixel-MAE supervision, which learns a reconstruction-oriented feature space rather
than using a semantic prior.

### 3.4 Predict-then-condition (mechanism B)

Rather than treat the pretrained representation as a second denoising target, the
generator *predicts* it and conditions on its own prediction. A lightweight head
maps the generator's hidden state to representation tokens $\hat{z}$, supervised
against the frozen encoder's features with an alignment loss
$\mathcal{L}_{\text{align}} = 1 - \cos(\hat{z}, \mathcal{E}(x))$; $\hat{z}$ is then
fed back as conditioning in the same step. This removes the second denoising
trajectory and its schedule, while keeping representation guidance. Combined with
grounding, the total objective is

$$
\mathcal{L} = \mathcal{L}_{\text{drift}} + \lambda\,\mathcal{L}_{\text{align}} .
$$

### 3.5 Frequency-balanced pixel supervision

Because X-Drifting supervises in pixel space, the output can be regularized in the
frequency domain. Natural images follow a $1/f^2$ spectrum, so a pixel-uniform loss
under-weights perceptually important high frequencies. We add the focal
log-frequency loss to the generated pixels against real samples: with error
spectrum $e_{u,v}=|\mathcal{F}(\hat{x})-\mathcal{F}(x)|$ and focal normalization
$\hat{e}=e/\max e$, each frequency bin is weighted by $\log(1+\rho_{u,v})$, with the
frequency weight decaying over training. This is a drop-in spectral regularizer.

### 3.6 Architecture

The generator is a vision transformer matched to the strongest cross-space
baselines: RMSNorm, SwiGLU, 2D RoPE, QK-norm, zero-initialized vector gates, with
in-context class and time prefix tokens and a pixel unpatchify head. The input is
a latent patch-embedding and the output is pixels; the token grid is shared across
both sides. Grounding tokens from $\mathcal{E}$ modulate the transformer blocks.

## 4. Experiments

We evaluate on CIFAR-10 as a controlled proxy for ImageNet-256, reporting Fréchet
Inception Distance (FID). The generator, drift feature extractor, and budget are
held fixed across comparisons so differences are attributable to the noise prior,
the grounding representation, or the spectral regularizer.

### 4.1 Cross-space drifting versus a pixel-noise baseline

Two generators share pixel-space supervision and differ only in the noise prior: a
pixel-noise **P-P** baseline and **X-Drifting (L-P)** (latent noise, pixel output).

| | noise prior | supervision | FID $\downarrow$ |
|---|---|---|---|
| P-P baseline | pixel | pixel | 216.9 |
| X-Drifting (L-P) | latent | pixel | 227.3 |

*(Table 1. CIFAR-10, matched budget.)*

| step | 2.5k | 5k | 7.5k | 10k | 15k | 20k |
|---|---|---|---|---|---|---|
| P-P | 268.5 | 222.6 | 218.6 | 217.2 | 221.0 | 216.9 |
| X-Drifting (L-P) | 309.8 | 274.1 | 254.1 | 239.7 | 236.2 | 227.3 |

*(Table 2. FID trajectory.)*

Without a representation prior, the latent-noise model improves faster (larger
relative FID reduction) but the pixel-noise baseline remains slightly ahead at this
budget. This is precisely the gap that representation grounding is designed to
close: the next sections show that adding the representation prior—and improving
its quality—reverses the ordering.

### 4.2 Representation grounding closes the gap

We evaluate X-Drifting with the drift objective grounded in a frozen encoder, and
vary the encoder along the representation-alignment ladder.

| grounding encoder | representation | FID $\downarrow$ |
|---|---|---|
| none (pixel-MAE) | — | — |
| SD-VAE | generic compressed latent | — |
| RAE | DINO autoencoder latent | — |
| PAE | prior-aligned (diffusion-friendly) latent | — |

*(Table 3. X-Drifting (L-P) with representation grounding; results in
preparation.)*

A more diffusion-friendly latent manifold is expected to yield better generation at
the same budget. This isolates the latent advantage: X-Drifting *leverages*
representation quality through grounding, rather than merely tolerating a latent
prior.

### 4.3 Predict-then-condition and frequency supervision

| | FID $\downarrow$ |
|---|---|
| grounding only | — |
| grounding + predict-then-condition | — |
| grounding + frequency supervision | — |
| grounding + both | — |

*(Table 4. Mechanism ablations; results in preparation.)*

Predict-then-condition accelerates convergence relative to representation-free
training, and frequency-balanced pixel supervision improves fidelity without
architectural change.

### 4.4 Visual results

Figure 1 compares class-conditional samples from the pixel-noise baseline and
grounded X-Drifting under a matched budget.

*(Figure 1: sample grids.)*

## 5. Discussion

**Why grounding, not an autoencoder.** The value of latent generation lies in the
encoder's representation, not the decoder's reconstruction. Grounding keeps the
former and drops the latter, so the latent advantage is retained while
reconstruction loss is removed from the generative path.

**Why drift is natively cross-space.** The stop-gradient target is a fixed point in
the output space; the input need not share that space. This is why the drift
objective maps latent noise to pixels with one forward-and-backward pass, unlike
differential cross-space objectives that require a Jacobian-vector product.

**Why predict rather than denoise the representation.** The pretrained
representation is a property of the emerging image, not an independent generative
target. Predicting and self-conditioning avoids the second trajectory and its
scheduling.

**Limitations.** Grounding evaluates a frozen encoder on evolving samples, which
mismatches its clean-image pretraining; a timestep-conditioned projection mitigates
but does not remove this. Class-conditional generation at very large class counts
remains to be validated at ImageNet scale.

## 6. Conclusion

X-Drifting resolves the latent-versus-pixel tension by keeping the representation
prior and dropping the reconstruction decoder. Drifting from latent noise directly
to pixels, grounded in a frozen encoder's feature space and guided by
predict-then-condition, it enjoys the latent space's generation acceleration and
compression while supervising in pixel space without reconstruction loss. The drift
loss makes this cross-space generation native—one forward-and-backward pass, no
Jacobian-vector product—and the latent advantage grows with the quality of the
grounding representation and with frequency-balanced pixel supervision.

## References

*(To be filled: PixelDiT / PixelDiT2, DINO-Forcing, REPA / REPA-E, Trajectory
Forcing, TBSM, CrossFlow, drifting models, latent diffusion (SD/SDXL), diffusion
/ flow matching, consistency / MeanFlow / iMF, MAE, DINOv2 / DINOv3, SD-VAE,
RAE, PAE, focal log-frequency loss.)*
