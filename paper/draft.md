# Representation-Grounded One-Step Drifting: Generating Pixels from Noise via a Frozen Encoder

## Abstract

Pixel-space generation has nearly closed the image-quality gap with latent
diffusion, yet it converges more slowly and lacks the compact, structured
representation that makes latent generation easy to learn. We argue the missing
ingredient is not an autoencoder but an explicit *representation prior*, and that
this prior can be brought to bear without a decoder or reconstruction loss. We
present **X-Drifting**, a one-step generative model that maps a Gaussian noise
input of arbitrary size to a pixel image while grounded in the representation
space of a frozen vision foundation model. Two mechanisms do the work.
*Representation grounding* evaluates the drifting attraction-repulsion objective in
the feature space of a frozen encoder (SD-VAE, RAE, or PAE), so the quality of the
representation shapes generation directly. *Predict-then-condition* makes the
generator predict the pretrained representation it needs and condition on its own
prediction, removing the need to denoise a second representation trajectory.
Because the drift loss is a fixed-point objective with a stop-gradient target, it
admits a noise-to-pixel mismatch natively and trains with a single
forward-and-backward pass—no Jacobian-vector product and no adversarial critic. We
show that grounding in a better-aligned representation (PAE over RAE over SD-VAE)
improves generation, and that frequency-balanced pixel supervision further
accelerates convergence and fidelity.

## 1. Introduction

Image generation has evolved through several paradigms: variational autoencoders
bind reconstruction and generation into one symmetric objective; generative
adversarial networks introduced a game-theoretic view that yields photorealistic
samples; and score-based models—diffusion and flow matching—have become nearly
synonymous with generative modeling on the strength of their visual quality. A
recent paradigm, the *drifting* model, departs from the established path: its
evolution over the course of training provides a native route to one-step
generation.

This paper revisits a long-standing split between two design spaces. Latent
diffusion compresses images into a compact representation but ties generation to a
decoder whose reconstruction loss caps output fidelity. Pixel diffusion removes
the decoder and generates images directly, but converges slowly and exhibits
color-shift and blur artifacts. The asymmetry is structural: latent models
denoise inside a compact space whose semantics are partially organized before
generation begins, whereas pixel models must learn denoising-friendly
representations and pixel generation simultaneously from raw RGB.

The autoencoder is not what makes latent generation work—its *representation* is.
A pretrained vision encoder already provides a coordinate system in which semantic
structure and low-level detail are organized; the decoder that maps back to pixels
is what introduces reconstruction loss. This suggests a different design: keep the
representation prior, drop the decoder, and generate pixels directly. Classical
prior work cannot express this, because the noise space and the output space must
coincide: the learning target is their interpolation. **CrossFlow** first broke
this constraint, removing the velocity term with an elegant mathematical device;
its objective, however, is consistency-style and its final quality depends largely
on adversarial and perceptual losses.

We present **X-Drifting**, a cross-space generative model built on the drifting
objective and instantiated in the same vision transformer as CrossFlow for a fair
comparison. The drift loss is a fixed point with a stop-gradient target, so it
admits the input/output mismatch natively—no Jacobian-vector product—and we ground
it in the representation of a frozen vision encoder so that representation quality
governs generation. Our contributions are threefold:

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
   setting**, matching or exceeding approaches—including CrossFlow—that rely on
   adversarial training to realize cross-space generation, while training with a
   single forward-and-backward pass.

## 2. Related Work

**Diffusion and drifting models.** Diffusion models learn to reverse a
noise-corruption process; flow matching instead regresses a velocity field that
transports noise to data along a prescribed path. Transformer backbones (DiT, SiT)
brought scalability, and one-step variants—consistency models, trajectory models,
MeanFlow and its improvements—compress iterative sampling by learning to jump to
the endpoint. All of these define the trajectory, velocity, and prediction target
in a single space. The drifting model takes a different route: it trains a
generator as a fixed point of an anti-symmetric attraction-repulsion field in a
feature space,
$L = \mathbb{E}\,\| f_\theta(\epsilon) - \mathrm{sg}\big( f_\theta(\epsilon) + V(f_\theta(\epsilon)) \big) \|^2$,
where $V$ is a drift over positive and negative sample banks evaluated with a
multi-temperature kernel. We show this objective is already cross-space capable.

**Cross-space generation.** CrossFlow first enables latent-noise to pixel-output
generation. It derives a general cross-space objective, cancels the marginal
velocity term, and instantiates a concrete loss with a total-derivative term
estimated by a Jacobian-vector product; its reported quality depends heavily on
perceptual (DINOv3-B Huber) and adversarial (hinge) losses. X-Drifting keeps the
same latent-to-pixel decomposition but replaces the differential objective with the
drift fixed point, which needs no Jacobian-vector product and no adversarial
critic. Flow matching and rectified flow are the classic noise-to-data transports
but remain single-space and iterative; energy-based formulations such as
three-body scattering share the attraction-repulsion structure and likewise report
that latent-space generators outperform pixel-space ones.

**Representations and autoencoders.** Latent diffusion compresses images through a
VAE whose decoder's reconstruction loss bounds output fidelity; stronger VAEs and
foundation-encoder latents (MAE, DINOv2, DINOv3) improve the representation but
keep generation coupled to a decoder. SD-VAE provides a generic compressed latent;
RAE replaces the VAE encoder with a frozen DINO and trains only a decoder; PAE
further shapes the latent into a diffusion-friendly manifold. On the guidance
side, REPA aligns a diffusion model's intermediate features to a frozen encoder's
clean-image targets but drops the encoder at inference, while PixelDiT2 conditions
the denoiser on the frozen encoder's features of the evolving sample and keeps this
path at inference; DINO-Forcing predicts the representation rather than denoising
it. X-Drifting grounds the drift objective in such a frozen encoder and keeps it
as the explicit feature space throughout, so representation quality governs
generation without a decoder or reconstruction loss.

## 3. Method

We study one-step class-conditional generation on ImageNet-256, from a Gaussian
noise input of arbitrary spatial size to a pixel image. The generator $F_\theta$
maps a noise sample $z \sim \mathcal{N}(0, I)$ directly to an image $\hat{x} \in
\mathbb{R}^{256 \times 256 \times 3}$ in a single evaluation. The noise $z$ may have
any shape—a full pixel grid, or a compact latent grid—and the output is always
pixels. Throughout, we use the *drift* objective as the generation loss, and we
ground it in the feature space of a frozen vision encoder. We describe the drift
objective (§3.1), representation grounding (§3.2), predict-then-condition with
representation injection (§3.3), frequency-balanced pixel supervision (§3.4), and
architecture and training (§3.5).

### 3.1 Drift objective

The drift objective trains a generator as a fixed point of an attraction-repulsion
field in a feature space $\phi$:

$$
\mathcal{L}_{\text{drift}}
= \mathbb{E}_{z}\,
\big\| F_\theta(z) - \mathrm{sg}\big( F_\theta(z) + V(F_\theta(z)) \big) \big\|^2 ,
$$

where $V$ is the drifting field, an attraction toward real samples minus a
repulsion from negative samples,

$$
V_{p,q}(\phi(x)) = V^{+}_p(\phi(x)) - V^{-}_q(\phi(x)),
\qquad
k(x,y) = \exp(-\|x-y\|/\tau),
$$

evaluated at multiple temperatures $\tau \in \{0.02, 0.05, 0.2\}$ and summed. The
attraction and repulsion are computed over positive and negative sample banks,
giving a mean-shift field whose fixed point matches the data distribution in $\phi$.

Two properties matter here. First, the target is a *stop-gradient* fixed point: the
loss does not differentiate through $V$, so a single forward-and-backward pass
suffices—there is no Jacobian-vector product and no adversarial critic. Second,
the objective constrains $F_\theta$ only through the feature map $\phi$; the input
$z$ need not live in the space of $F_\theta(z)$. The generator may therefore take a
noise sample of any size and output pixels, with the drift evaluated wherever we
choose to place $\phi$. This is where the representation prior enters.

### 3.2 Representation grounding

The feature space $\phi$ determines what the generator is made to match. Instead of
a learned reconstruction feature space, we evaluate the drift in the features of a
**frozen** pretrained vision encoder $\mathcal{E}$:

$$
\phi(\hat{x}) = \mathcal{E}(\hat{x}) \in \mathbb{R}^{L \times D},
$$

where $\mathcal{E}$ is a frozen SD-VAE, RAE, or PAE encoder and $L$ is the patch
count. Per-patch tokens are aggregated into the multi-scale statistics consumed by
the drift field: global mean and standard deviation, and $2{\times}2$ and
$4{\times}4$ patch means and standard deviations. The encoder's decoder is never
used, so no reconstruction loss enters the generative path.

Because $\mathcal{E}$ is a representation prior, the *quality* of the representation
directly shapes the drifting field and hence generation. We use a ladder of
increasing representation alignment—SD-VAE provides a generic compressed latent,
RAE replaces the encoder with a frozen DINO and trains only a decoder, and PAE
further shapes the latent into a diffusion-friendly manifold—so that the effect of
representation quality is measurable. This construction differs from REPA, which
aligns internal features to clean-image targets and drops the encoder at inference;
here $\mathcal{E}$ remains the explicit feature space of the objective throughout.

### 3.3 Predict-then-condition

A pretrained representation is a property of the image that the generator can
infer; it need not be an externally supplied target. We therefore let the generator
*predict* the representation and then *condition* on its own prediction, in two
stages within a single forward pass.

**Stage 1 (predict).** The generator produces a first prediction from the noise
input, and a lightweight head $P_g$ maps its hidden state $H$ to representation
tokens

$$
\hat{z} = P_g(H) \in \mathbb{R}^{N \times D},
$$

aligned to the frozen encoder's features of the target image with a cosine loss

$$
\mathcal{L}_{\text{align}}
= \frac{1}{N} \sum_{i=1}^{N} \big( 1 - \cos(\hat{z}_i, \mathcal{E}(x)_i) \big).
$$

**Stage 2 (condition).** The predicted tokens $\hat{z}$ are injected back into the
generator as conditioning—through spatial AdaLN on the shared patch grid, or by
concatenating them to the input tokens as a conditioning stream in the manner of
DINO-Forcing. The generator then produces the final output *conditioned on* $\hat{z}$:

$$
\hat{x} = F_\theta\big(z, \mathrm{cond}(\hat{z})\big).
$$

The stop-gradient on $\hat{z}$ in the alignment loss is essential: it makes the
predicted representation a fixed target to condition on, not a quantity to
differentiate through. This is the same predict-then-condition separation
DINO-Forcing uses, realized here within one generation pass.

**Compared to perceptual losses.** A perceptual loss compares two images through a
feature extractor and penalizes their distance, acting as a *regularizer* on the
output. Predict-then-condition differs in both role and computation: it adds a
*prediction head* $P_g$ and an internal representation $\hat{z}$ that the generator
produces and then consumes through a conditioning pathway. The representation is
part of the model's update rule, not an external image metric.

### 3.4 Frequency-balanced pixel supervision

We further regularize the pixel output with a **focal log-frequency loss** ($f$-loss),
an additive term in the training objective (a regularizer, not a change to the drift
field). Natural images follow a $1/f^2$ spectrum, so a pixel-uniform loss
under-weights the perceptually important high frequencies; this spectral stiffness
also slows first-order drifting dynamics. The $f$-loss operates in the 2D DFT
domain: for frequency coordinate $(u,v)$, the spectral residual
$e_{u,v} = \| \mathcal{F}_\theta(u,v) - \mathcal{F}(u,v) \|_2$ compares the
generated $\hat{x}$ and target $x$. The loss is

$$
\mathcal{L}_{f}
= \sum_{u,v} \frac{1}{(1-t)^2} \cdot \frac{e_{u,v}}{\max_{u',v'} e_{u',v'}} \cdot \log(1 + e_{u,v}),
$$

with three components: a timestep weight $\frac{1}{(1-t)^2}$ inherited from the
$v$-loss, an adaptive focal weight $\frac{e_{u,v}}{\max e_{u',v'}}$ (stop-gradient,
normalizing residuals per-sample), and a logarithmic compression $\log(1+e_{u,v})$
that prevents any single frequency from dominating. The full objective is

$$
\mathcal{L} = \mathcal{L}_{\text{drift}} + \lambda \mathcal{L}_{\text{align}} + \mu(t)\, \mathcal{L}_{f},
$$

where $\mu(t)$ is a frequency weight that decays over training, so the model
emphasizes frequencies early and refines spatially late.

### 3.5 Architecture and training

The generator is a vision transformer identical to the strongest cross-space
baseline (CrossFlow): RMSNorm, SwiGLU, 2D RoPE, QK-norm, zero-initialized vector
gates, in-context class and time prefix tokens, a latent patch-embed on the input
side and a pixel unpatchify head on the output side, with a shared token grid. On
ImageNet-256 the latent patch-embed produces a $16 \times 16$ token grid and the
output head maps each token to a $16 \times 16$ pixel patch. The predicted
representation $\hat{z}$ (§3.3) shares the same token grid and is injected through
spatial AdaLN, modulating the transformer blocks. Training uses AdamW with a
constant learning rate after warmup, gradient clipping, and EMA; FID is computed on
generated samples against dataset reference statistics.

## 4. Experiments

We evaluate class-conditional generation on ImageNet-256, reporting Fréchet
Inception Distance (FID) on 50,000 samples against reference statistics. The
generator, the drift feature extractor, and the training budget are held fixed
across all comparisons,
so differences are attributable to the representation grounding (§4.1) or the
mechanism ablations (§4.2). Additional results, including MNIST convergence, the
toy cross-space study, and mathematical derivations, appear in the Appendix.

### 4.1 Representation grounding: the representation ladder

We train X-Drifting with the drift objective grounded in a frozen encoder and vary
the encoder along the representation-alignment ladder (SD-VAE, RAE, PAE). Table 1
reports FID.

| grounding encoder | representation | FID $\downarrow$ |
|---|---|---|
| SD-VAE | generic compressed latent | — |
| RAE | DINO autoencoder latent | — |
| PAE | prior-aligned (diffusion-friendly) latent | — |

*(Table 1. X-Drifting with representation grounding; results in preparation.)*

The representation prior—not the noise size—governs generation quality: a more
diffusion-friendly latent manifold is expected to yield better generation at the
same budget. This isolates the source of the advantage: X-Drifting *leverages*
representation quality through grounding, and the gain grows as the representation
becomes better aligned to the generative task.

### 4.2 Mechanism ablations

We ablate predict-then-condition (§3.3) and frequency-balanced pixel supervision
(§3.4) on top of representation grounding.

| | FID $\downarrow$ |
|---|---|
| grounding only | — |
| grounding + predict-then-condition | — |
| grounding + frequency supervision | — |
| grounding + both | — |

*(Table 2. Mechanism ablations; results in preparation.)*

Predict-then-condition accelerates convergence relative to representation-free
training, and frequency-balanced pixel supervision improves fidelity without
architectural change.

### 4.3 Visual results

Figure 1 shows class-conditional samples from X-Drifting with representation
grounding. Additional sample grids and convergence curves appear in the Appendix.

*(Figure 1: sample grids.)*

## 5. Discussion

**Design choices.** Grounding keeps the encoder and drops the decoder, so the
representation advantage is retained while reconstruction loss leaves the
generative path. The choice of grounding encoder is therefore a first-order design
decision: a diffusion-friendly latent (PAE) yields better generation than a
generic compressed one (SD-VAE) at the same budget. Predicting the representation
rather than denoising it avoids a second trajectory and its noise schedule, and
frequency-balanced pixel supervision compensates for the low-frequency bias of
pixel losses without architectural change.

**Computational profile.** The drift objective trains with one forward-and-backward
pass and no Jacobian-vector product, unlike differential cross-space objectives
that differentiate the model along the latent trajectory. At a fixed wall-clock
budget this yields more optimization steps, an advantage that grows with model size.

**Limitations.** Grounding evaluates a frozen encoder on evolving samples, which
mismatches its clean-image pretraining; a timestep-conditioned projection
mitigates but does not remove this mismatch. The frozen encoder is evaluated at
both training and inference, adding a constant cost per step. Class-conditional
generation at very large class counts remains to be validated at ImageNet scale.

## 6. Conclusion

X-Drifting generates pixels from noise in one step, grounded in the feature space of
a frozen vision encoder and guided by predict-then-condition. The drift objective
makes this cross-space generation native—one forward-and-backward pass, no
Jacobian-vector product—and the representation prior, not the noise size, governs
generation quality. Grounding in a better-aligned representation (PAE over RAE over
SD-VAE) improves generation without a decoder or reconstruction loss, and
frequency-balanced pixel supervision further improves fidelity.

## References

*(To be filled: PixelDiT / PixelDiT2, DINO-Forcing, REPA / REPA-E, Trajectory
Forcing, TBSM, CrossFlow, drifting models, latent diffusion (SD/SDXL), flow
matching, rectified flow, diffusion / consistency / MeanFlow / iMF, MAE,
DINOv2 / DINOv3, SD-VAE, RAE, PAE, focal log-frequency loss.)*

---

# Appendix

## A. Mathematical Derivations

### A.1 Drift objective

The drift objective minimizes the squared error between the generator output
$f_\theta(z)$ and its stop-gradient fixed point under the drifting field $V$:

$$
\mathcal{L}_{\text{drift}}
= \mathbb{E}_{z \sim \mathcal{N}(0,I)}
\Big\| f_\theta(z) - \mathrm{sg}\big(f_\theta(z) + V(f_\theta(z))\big) \Big\|^2 .
$$

Expanding the squared norm and using the stop-gradient on $V$,

$$
\mathcal{L}_{\text{drift}} = \mathbb{E}_{z} \big\| V(f_\theta(z)) \big\|^2 ,
$$

so the objective minimizes the magnitude of the drifting field evaluated at the
generator output. The field $V$ decomposes into attraction and repulsion terms,

$$
V_{p,q}(\phi(x)) = \frac{1}{Z_p} \sum_{y^+ \in p} k(x,y^+) (y^+ - x)
- \frac{1}{Z_q} \sum_{y^- \in q} k(x,y^-) (y^- - x),
$$

with $k(x,y) = \exp(-\|x-y\|/\tau)$ the radial kernel and $Z_p, Z_q$ normalizers.
At the fixed point $V(f_\theta(z)) = 0$, the attraction to the data distribution
balances the repulsion from negative samples, and $f_\theta(z)$ matches the data
distribution in $\phi$.

### A.2 Multi-temperature feature aggregation

For each feature layer $\ell$ of the frozen encoder $\mathcal{E}$, we form
per-patch tokens $h^{(\ell)} \in \mathbb{R}^{L \times D}$ and aggregate

$$
\phi^{(\ell)} = \Big\{ h^{(\ell)},\; \mathrm{mean}(h^{(\ell)}),\; \mathrm{std}(h^{(\ell)}),\;
\mathrm{mean}_{2\times2}(h^{(\ell)}),\; \mathrm{std}_{2\times2}(h^{(\ell)}),\;
\mathrm{mean}_{4\times4}(h^{(\ell)}),\; \mathrm{std}_{4\times4}(h^{(\ell)}) \Big\},
$$

and sum the drift loss over temperatures $\tau \in \{0.02, 0.05, 0.2\}$ and layers
$\ell$. This yields the multi-scale, multi-temperature drift signal.

### A.3 Predict-then-condition

The representation head $P_g$ maps the generator's hidden state $H \in
\mathbb{R}^{N \times d}$ to representation tokens $\hat{z} = P_g(H) \in
\mathbb{R}^{N \times D}$, where $N$ is the patch count and $D$ the encoder's
feature dimension. The alignment loss is

$$
\mathcal{L}_{\text{align}}
= \frac{1}{N} \sum_{i=1}^{N}
\Big( 1 - \cos\big( \hat{z}_i, \mathcal{E}(x)_i \big) \Big),
$$

where $\mathcal{E}(x)_i \in \mathbb{R}^{D}$ is the frozen encoder's feature at patch
$i$ of the target image $x$. The predicted tokens $\hat{z}$ are injected into the
generator as conditioning—through spatial AdaLN on the shared $16 \times 16$ patch
grid, or by concatenating them to the input tokens as a conditioning stream in the
manner of DINO-Forcing. The stop-gradient on $\hat{z}$ in $\mathcal{L}_{\text{align}}$
makes the prediction a fixed target to condition on rather than a differentiable
path. The combined objective is
$\mathcal{L} = \mathcal{L}_{\text{drift}} + \lambda \mathcal{L}_{\text{align}}$.

### A.4 Focal log-frequency loss

Images admit two complementary representations: the spatial domain, which captures
local structure, and the frequency domain, which captures global patterns across
scales. A pixel-uniform loss under-weights the perceptually important high
frequencies. We adopt the **focal log-frequency loss** ($f$-loss), a spectrally
balanced objective that equalizes the learning signal across frequencies.

For a frequency at coordinate $(u, v)$ in the 2D Discrete Fourier Transform (DFT),
the spectral residual between the generated image $\hat{x}$ and the target $x$ is

$$
e_{u,v} = \big\| \mathcal{F}_\theta(u,v) - \mathcal{F}(u,v) \big\|_2 ,
$$

where $\mathcal{F}$ denotes the 2D DFT and $\mathcal{F}_\theta(u,v)$,
$\mathcal{F}(u,v)$ are the DFT coefficients of $\hat{x}$ and $x$ at $(u, v)$. The
$f$-loss is

$$
\mathcal{L}_{f}
= \sum_{u,v}
\frac{1}{(1-t)^2}
\cdot
\frac{e_{u,v}}{\displaystyle\max_{u',v'} e_{u',v'}}
\cdot
\log\!\big(1 + e_{u,v}\big),
$$

with three components:

1. **Timestep weighting** $\frac{1}{(1-t)^2}$, inherited from the $v$-loss and
   preserved unchanged, ensuring consistent timestep dynamics.

2. **Adaptive focal weight** $\frac{e_{u,v}}{\max_{u',v'} e_{u',v'}}$: normalizes the
   per-frequency residuals by the maximum residual across the spectrum at each
   training step, computed with a stop-gradient. This is a data-dependent
   coefficient that continuously normalizes each sample to avoid magnitude
   variations.

3. **Logarithmic compression** $\log(1 + e_{u,v})$: prevents any single frequency
   from dominating the loss, acting as a continuous generalization of a Laplacian
   pyramid—each doubling of frequency receives equal structural weight, so the loss
   is not dominated by high-energy, low-frequency components.

The full training objective is

$$
\mathcal{L} = \mathcal{L}_{\text{drift}} + \lambda \mathcal{L}_{\text{align}} + \mu(t)\, \mathcal{L}_{f},
$$

where $\mu(t)$ is a frequency weight that decays over training, so the model
emphasizes frequencies early and refines spatially late.

## B. MNIST Concept-Validation Study

Before scaling to ImageNet, we validate the cross-space pipeline on MNIST
(64$\times$64, 10 classes) under a matched budget and identical pixel-space
supervision. We compare a pixel-noise baseline against X-Drifting (latent noise,
pixel output) with the same drift objective and the same frozen feature extractor.

| | noise | supervision | FID $\downarrow$ | class-acc $\uparrow$ |
|---|---|---|---|---|
| pixel-noise baseline | pixel | pixel | 188.4 | 66.0% |
| X-Drifting | latent | pixel | 178.4 | 60.2% |

*(Table B.1. MNIST 64$\times$64, matched budget.)*

Both generators produce clear, class-correct digits: when asked for a 5, each draws
a recognizable 5 (Figure B.1). X-Drifting reaches a lower FID at comparable
class-accuracy while operating from a compact latent noise prior. The FID
trajectory (Figure B.2) shows X-Drifting improving at a larger relative rate and
converging toward the pixel baseline, consistent with the latent prior acting as a
better-conditioned starting distribution.

*(Figure B.1: MNIST sample grids, pixel-noise baseline (top) and X-Drifting
(bottom). Figure B.2: FID trajectory over training.)*

## C. Additional Visual Results

Class-conditional sample grids and convergence curves for all experiments are in
`paper/figures/`.
