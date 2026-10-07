# Conditional VAE and ResNet training roadmap

## Objective

Train a conditional VAE independently, then reuse its pretrained decoder inside a ResNet-style generator for the successor-digit task:

```text
input label y -> target label (y + 1) % 10
```

The final model should follow this flow:

```text
raw MNIST image [0,1]
        |
        v
ResNet-style encoder
        |
        v
mu and log_var
        |
        v
latent vector z
        |
        + target-label embedding
        |
        v
pretrained conditional decoder
        |
        v
generated MNIST image [0,1]
        |
        v
MNIST normalization
        |
        v
frozen MNIST classifier
```

The ResNet component should generate a latent representation rather than directly generating an image. The conditional decoder is responsible for turning that latent representation into a realistic digit.

## Tensor and value-range contract

Keep these domains distinct throughout the implementation:

```text
raw image:         [B,1,28,28], values in [0,1]
normalized image:  [B,1,28,28], approximately [-0.42,2.82]
mu:                [B,latent_dim]
log_var:           [B,latent_dim]
z:                 [B,latent_dim]
labels:            [B], dtype torch.long
```

Load MNIST using only:

```python
transform=torchvision.transforms.ToTensor()
```

Use raw images for:

- VAE encoder input;
- reconstruction targets;
- decoder output;
- total-variation loss;
- plotting.

Normalize only when passing images into the pretrained classifier:

```python
def normalize_mnist(images):
    return (images - 0.1307) / 0.3081
```

Do not overwrite the raw batch with its normalized version. Create a normalized view only where it is needed.

---

## Phase 1: standalone conditional-VAE pretraining

### 1.1 Construct the models

```python
latent_dim = 16
label_dim = 8

encoder = Encoder(
    latent_dim=latent_dim,
).to(device)

decoder = ConditionalDecoder(
    latent_dim=latent_dim,
    label_dim=label_dim,
).to(device)
```

The encoder is already a ResNet-style encoder because it contains residual blocks and produces `mu` and `log_var`.

### 1.2 Construct one optimizer for both components

```python
from torch.optim import Adam

optimizer = Adam(
    list(encoder.parameters())
    + list(decoder.parameters()),
    lr=1e-3,
)
```

Adam is a practical starting optimizer for VAE training.

### 1.3 Pretrain by reconstructing the original class

During pretraining, the decoder receives the original label:

```text
image of 7 -> encoder -> z + label 7 -> reconstruction of 7
```

The training step is:

```python
mu, log_var = encoder(imgs_raw)
z = rep_sample(mu, log_var)

reconstructed = decoder(z, labels)

reconstruction_loss = (
    nn.functional.binary_cross_entropy(
        reconstructed,
        imgs_raw,
        reduction="sum",
    )
    / imgs_raw.size(0)
)

latent_loss = KL_loss(mu, log_var)

loss = reconstruction_loss + beta * latent_loss
```

The KL loss should reduce latent dimensions first and then average the batch:

```python
def KL_loss(mu, log_var):
    return 0.5 * (
        log_var.exp()
        + mu.square()
        - 1
        - log_var
    ).sum(dim=1).mean()
```

### 1.4 Optional KL warm-up

Starting immediately with a strong KL term can cause posterior collapse before the decoder learns reconstruction.

An optional linear warm-up is:

```python
beta = min(1.0, epoch / 10)
```

This gradually changes `beta` from `0.1` to `1.0` over the first ten epochs.

### 1.5 Logging

Log these values independently:

```text
average total loss
average reconstruction loss
average KL loss
current beta
```

Protect short training runs with:

```python
log_every = max(1, epochs // 10)
```

---

## Phase 2: validate the pretrained VAE

Do not accept the model based only on total loss. Perform the following checks.

### 2.1 Deterministic reconstruction

Use `z = mu` to remove sampling noise:

```python
encoder.eval()
decoder.eval()

with torch.inference_mode():
    mu, log_var = encoder(imgs_raw)
    reconstructed = decoder(mu, labels)
```

The reconstructed images should be recognizable versions of the inputs.

### 2.2 Target-label swapping

Encode an image once and decode the same `z` with every label from `0` to `9`:

```text
same z + label 0 -> digit 0
same z + label 1 -> digit 1
...
same z + label 9 -> digit 9
```

This evaluates whether:

- `z` represents style and other variations;
- the label embedding controls digit identity;
- the decoder actually uses its condition.

If every result resembles the original class, `z` may be storing too much class information or the decoder may be ignoring the label.

### 2.3 Random prior sampling

Generate images from the prior:

```python
z = torch.randn(
    batch_size,
    latent_dim,
    device=device,
)

generated = decoder(z, target_labels)
```

Recognizable results indicate that the decoder works on the intended standard-normal latent distribution rather than only on encoded training examples.

### 2.4 Optional classifier validation

Pass generated images through the frozen classifier:

```python
logits = classifier(
    normalize_mnist(generated)
)

predicted_labels = logits.argmax(dim=1)
```

Use this together with visual inspection. Classifier accuracy alone does not prove that images look realistic.

---

## Phase 3: save a reusable VAE checkpoint

Save both encoder and complete conditional decoder:

```python
torch.save(
    {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "latent_dim": latent_dim,
        "label_dim": label_dim,
        "epochs": epochs,
    },
    "mnist_conditional_vae.pt",
)
```

The complete `ConditionalDecoder` must be saved, including:

- label embedding;
- latent/label input linear layer;
- convolutional decoder.

Saving only the inner `DecoderModule` is insufficient.

Although the decoder is the main reusable component, save the encoder as well. Its weights provide a valid initialization for the later ResNet-style latent generator.

---

## Phase 4: build the final successor generator

Wrap the pretrained encoder and decoder in one model:

```python
class SuccessorGenerator(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        imgs_raw,
        target_labels,
        sample=None,
    ):
        mu, log_var = self.encoder(imgs_raw)

        if sample is None:
            sample = self.training

        if sample:
            z = rep_sample(mu, log_var)
        else:
            z = mu

        generated = self.decoder(
            z,
            target_labels,
        )

        return generated, mu, log_var
```

The default behaviour becomes:

```text
model.train() -> sample z for stochastic training
model.eval()  -> use mu for deterministic inference
```

### Relationship with the old `ResNETSEQ`

The existing complete `ResNETSEQ` maps an image to another image:

```text
[B,1,28,28] -> [B,1,28,28]
```

The conditional decoder instead expects:

```text
z:             [B,latent_dim]
target labels: [B]
```

Therefore, the old `ResNETSEQ` cannot be placed unchanged before the decoder. For the preferred architecture, use the VAE `Encoder` as the ResNet component, or refactor `ResNETSEQ` so its final heads produce `mu` and `log_var` instead of an image.

The pretrained image-generating ResNet weights may not be desirable because they were optimized to produce classifier-targeted patterns rather than a regular latent distribution.

---

## Phase 5: load the pretrained components

```python
checkpoint = torch.load(
    "mnist_conditional_vae.pt",
    map_location=device,
    weights_only=True,
)

encoder = Encoder(
    latent_dim=checkpoint["latent_dim"],
).to(device)

decoder = ConditionalDecoder(
    latent_dim=checkpoint["latent_dim"],
    label_dim=checkpoint["label_dim"],
).to(device)

encoder.load_state_dict(
    checkpoint["encoder_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

model = SuccessorGenerator(
    encoder,
    decoder,
).to(device)
```

---

## Phase 6: test label swapping before fine-tuning

Try successor generation using only the pretrained conditional VAE:

```python
model.eval()

with torch.inference_mode():
    target_labels = (labels + 1) % 10

    generated, mu, log_var = model(
        imgs_raw,
        target_labels,
        sample=False,
    )
```

This establishes a baseline. Conditional VAE pretraining may already be sufficient for some label-swapping behaviour.

Record:

- visual quality;
- target-class accuracy;
- reconstruction quality;
- whether handwriting style is preserved;
- whether different source images collapse to similar outputs.

---

## Phase 7: train the successor task with a frozen decoder

Initially freeze the decoder so classifier loss cannot immediately damage its learned digit manifold:

```python
decoder.requires_grad_(False)
decoder.eval()

classifier.requires_grad_(False)
classifier.eval()
```

Freezing decoder parameters still allows gradients to pass through the decoder into the encoder. Do not execute the decoder inside `torch.no_grad()` or `torch.inference_mode()` during training, because that would interrupt the gradient path.

The training step is:

```python
imgs_raw = imgs.to(device)
labels = labels.to(device)
target_labels = (labels + 1) % 10

generated, mu, log_var = model(
    imgs_raw,
    target_labels,
    sample=True,
)

logits = classifier(
    normalize_mnist(generated)
)

class_loss = nn.functional.cross_entropy(
    logits,
    target_labels,
)

latent_loss = KL_loss(mu, log_var)
tv_loss = total_variation_loss(generated)

loss = (
    class_loss
    + beta * latent_loss
    + tv_weight * tv_loss
)
```

Conservative initial weights are:

```python
beta = 0.01
tv_weight = 0.001
```

These are only starting points. Log the unweighted values of all loss components before deciding their relative weights.

The KL term keeps encoder outputs near the latent distribution on which the decoder was pretrained. Without it, classification loss could drive the encoder into abnormal latent regions.

---

## Phase 8: optionally fine-tune decoder conditioning

If images remain realistic but target-class accuracy is insufficient, unfreeze only:

- `decoder.label_embedding`;
- `decoder.input_layer`.

Use a lower learning rate than the encoder:

```python
optimizer = Adam([
    {
        "params": encoder.parameters(),
        "lr": 1e-3,
    },
    {
        "params": decoder.label_embedding.parameters(),
        "lr": 1e-4,
    },
    {
        "params": decoder.input_layer.parameters(),
        "lr": 1e-4,
    },
])
```

Only unfreeze the convolutional decoder later if necessary. Updating it too early with classifier loss can recreate unnatural classifier-exploiting images.

---

## Phase 9: deterministic inference

```python
model.eval()
classifier.eval()

with torch.inference_mode():
    imgs_raw = imgs.to(device)
    labels = labels.to(device)
    target_labels = (labels + 1) % 10

    generated, _, _ = model(
        imgs_raw,
        target_labels,
        sample=False,
    )

    logits = classifier(
        normalize_mnist(generated)
    )

    predictions = logits.argmax(dim=1)
```

Using `z = mu` produces stable results for identical inputs. Sampling from the latent distribution can be exposed as an optional mode when multiple output variations are desired.

---

## Suggested checkpoints

Keep separate files for distinct stages:

```text
mnist_conditional_vae.pt
    standalone VAE pretraining checkpoint

mnist_successor_frozen_decoder.pt
    successor model after training only the encoder

mnist_successor_finetuned.pt
    final model after optional decoder fine-tuning
```

Do not overwrite the standalone VAE checkpoint during successor-task training. It is the reusable fallback if later fine-tuning damages image quality.

---

## Completion checklist

### Before successor training

- [ ] Dataset supplies raw `[0,1]` images.
- [ ] Reconstruction images are recognizable.
- [ ] KL loss is finite and does not immediately collapse to zero.
- [ ] Random prior samples resemble MNIST digits.
- [ ] Changing the target label changes the generated class.
- [ ] Both encoder and complete conditional decoder are saved.

### Before decoder fine-tuning

- [ ] Frozen-decoder baseline has been recorded.
- [ ] Class, KL and TV losses are logged separately.
- [ ] Generated images remain human-readable.
- [ ] Target accuracy is evaluated using normalized classifier inputs.

### Before final inference

- [ ] The model uses `mu` rather than random sampling by default.
- [ ] The decoder and classifier are in evaluation mode.
- [ ] Generated images remain in `[0,1]` for plotting.
- [ ] Only classifier inputs are MNIST-normalized.
- [ ] The standalone VAE checkpoint remains preserved.

---

## Future extensions

Possible later experiments:

- KL-weight scheduling other than linear warm-up;
- smaller or larger latent dimensions;
- denoising VAE pretraining;
- GroupNorm instead of BatchNorm;
- style-preservation or cycle-consistency losses;
- adversarial realism loss;
- multiple sampled outputs per input;
- class-conditional latent-space visualization;
- per-class reconstruction and successor accuracy metrics.
