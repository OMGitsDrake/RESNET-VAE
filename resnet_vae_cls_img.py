import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch import (
    cuda,
    inference_mode,
    device as dvc,
    load,
    save,
    cat,
    randn
)
from mlp_classifier import MLP
from resnet_classifier_img import ResNETSEQ
from datetime import datetime
from pathlib import Path
from utility import (
    save_training_preview,
    save_vae_reconstruction_preview,
    save_vae_reconstruction_inference_preview,
    total_variation_loss,
    MODELS_DIR,
    KL_loss,
    normalize_minst,
    rep_sample,
    update_beta
)
from definitions import (
    EncoderModule,
    DecoderModule,
)
from os import remove

# raw input
#    ↓
# ResNet-style feature extractor
#    ↓
# mu and log_var heads
#    ↓
# z + target-label embedding
#    ↓
# pretrained conditional decoder
#    ↓
# raw generated image


# Encoder:
# input image → μ and log_var → z
#                          ↓
# Decoder:
# z + target-label embedding → generated image

class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        label_dim: int = 8
    ):
        super().__init__()
        
        self.label_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=label_dim
        )

        self.input_layer = nn.Linear(
            latent_dim + label_dim,
            64 * 7 * 7
        )

        self.decoder = DecoderModule()

    def forward(self, z, target_labels):
        label_features = self.label_embedding(target_labels)
        cond_z = cat((z, label_features), dim=1)

        out = self.input_layer(cond_z)
        out = out.reshape(z.size(0), 64, 7, 7)
        out = self.decoder(out)

        return out
    

def pretrain_vae(
        loader,
        device,
        epochs: int = 50,
) -> tuple[float, float, float]:
    log_every = epochs // 10
    encoder = EncoderModule(latent_dim=16).to(device)
    decoder = ConditionalDecoder(label_dim=8).to(device)
    
    encoder.train()
    decoder.train()

    optim = Adam(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=1e-3
    )

    fixed_images, fixed_labels = next(iter(loader))
    
    fixed_images = fixed_images[:6].to(device)
    fixed_labels = fixed_labels[:6].to(device)

    prev_dir = Path(__file__).resolve().parent / 'logs' / 'vae_rec_preview'
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)
    for file in prev_dir.iterdir():
        remove(file)

    print(f'VAE PRETRAINING STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_KL_loss = 0.0
        epoch_samples = 0
        beta_i = update_beta(epoch=epoch, full_weight_epoch=epochs-epochs//3)

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            mu, log_var = encoder(imgs)
            z = rep_sample(mu, log_var)

            reconstructed = decoder(z, labels)
            rec_loss = nn.functional.binary_cross_entropy(reconstructed, imgs, reduction='sum') / imgs.size(0)
            latent_loss = KL_loss(mu, log_var)

            loss = rec_loss + beta_i * latent_loss
            loss.backward()

            optim.step()

            epoch_loss += loss.item() * imgs.size(0)
            epoch_rec_loss += rec_loss.item() * imgs.size(0)
            epoch_KL_loss += latent_loss.item() * imgs.size(0)
            epoch_samples += imgs.size(0)

        avg_loss = epoch_loss / epoch_samples
        avg_rec_loss = epoch_rec_loss / epoch_samples
        avg_KL_loss = epoch_KL_loss / epoch_samples
        
        if epoch % log_every == 0 or epoch == 1:
            print(
                f'epoch: {epoch}/{epochs}:\n'
                f'\t AVG: \t\t{avg_loss:.4f}\n'
                f'\t AVG KL: \t{avg_KL_loss:.4f}\n'
                f'\t AVG Rec: \t{avg_rec_loss:.4f}\n'
                f'\t beta: \t\t{beta_i:.4f}\n'
                f'\t LAST: \t\t{loss:.4f}\n'
            )

            save_vae_reconstruction_preview(
                encoder=encoder,
                decoder=decoder,
                input_images=fixed_images,
                input_labels=fixed_labels,
                epoch=epoch,
                average_loss=avg_loss,
                output_directory=prev_dir
            )

    print(f'VAE PRETRAINING FINISHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')

    save(
        encoder.state_dict(),
        f'{MODELS_DIR}/vae_encoder.pt'
    )
    save(
        decoder.state_dict(),
        f'{MODELS_DIR}/vae_decoder.pt'
    )

    return avg_loss, avg_KL_loss, avg_rec_loss


def vae_inference(
    loader,
    device,
    generation: int = 0
) -> tuple[float, float, float]:
    encoder = EncoderModule(latent_dim=16)
    decoder = ConditionalDecoder(label_dim=8)

    state_enc = load(
        f'{MODELS_DIR}/vae_encoder.pt',
        map_location=device,
        weights_only=True
    )
    state_dec = load(
        f'{MODELS_DIR}/vae_decoder.pt',
        map_location=device,
        weights_only=True
    )
    encoder.load_state_dict(state_enc)
    decoder.load_state_dict(state_dec)
    
    encoder.to(device).eval()
    decoder.to(device).eval()

    prev_dir = Path(__file__).resolve().parent / 'logs' / 'vae_rec_inference'
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)
    for file in prev_dir.iterdir():
        remove(file)

    total_loss = 0.0
    total_rec_loss = 0.0
    total_KL_loss = 0.0
    total_samples = 0

    loss = float('nan')

    print(f'VAE INFERENCE STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    with inference_mode():
        for i, (imgs, labels) in enumerate(loader, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)
    
            mu, log_var = encoder(imgs)

            match generation:
                case 0:
                    # sampled z
                    z = rep_sample(mu, log_var)
                    
                    reconstructed = decoder(z, labels)
                    rec_loss = nn.functional.binary_cross_entropy(
                        reconstructed,
                        imgs,
                        reduction='sum'
                    ) / imgs.size(0)

                    latent_loss = KL_loss(mu, log_var)

                    loss = rec_loss + latent_loss

                    batch_size = reconstructed.size(0)
                    total_rec_loss += rec_loss.item() * batch_size
                    total_KL_loss += latent_loss.item() * batch_size
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                case 1:
                    # generate from random z and good label
                    z = randn(imgs.size(0), 16, device=device)
                    
                    reconstructed = decoder(z, labels)
                case 2:
                    # change label and maintain the embeddings
                    z = rep_sample(mu, log_var)
                    shift_labels = (labels + 1) % 10

                    reconstructed = decoder(z, shift_labels)
                case _:
                    return float('nan'), float('nan'), float('nan')

            if i % 10 == 0 or i == 1:
                print(f'batch - {i} | loss: {loss}')

                save_vae_reconstruction_inference_preview(
                    input_images=imgs,
                    rec_images=reconstructed,
                    labels=labels,
                    target_labels=labels,
                    batch=i,
                    loss=loss,
                    output_directory=prev_dir
                )
                
    print(f'VAE INFERENCE FINISHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    if generation == 0:
        avg_loss = total_loss / total_samples
        avg_rec_loss = total_rec_loss / total_samples
        avg_KL_loss = total_KL_loss / total_samples

        return avg_loss, avg_KL_loss, avg_rec_loss
    else:
        return float('nan'), float('nan'), float('nan')


def train(
    model: nn.Module,
    classifier: MLP,
    loader,
    optim: SGD,
    loss_fn: nn.CrossEntropyLoss,
    epochs: int,
    device,
    log_every: int = 10,
) -> float:
    classifier.eval()
    classifier.requires_grad_(False)
    model.train()

    fixed_images, fixed_labels = next(iter(loader))

    fixed_images = fixed_images[:6].to(device)
    fixed_labels = fixed_labels[:6].to(device)

    prev_dir = Path(__file__).resolve().parent / 'logs' / 'ex2_previews'
    prev_dir.mkdir(parents=True, exist_ok=True)
    for file in prev_dir.iterdir():
        remove(file)

    print(f'TRAINING STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    avg_loss = float('nan')
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_samples = 0

        for imgs, labels in loader:
            imgs = normalize_minst(imgs)
            imgs = imgs.to(device)
            labels = labels.to(device)
            target_labels = (labels + 1) % 10

            optim.zero_grad()

            pred_imgs = model(imgs)
            logits = classifier(normalize_minst(pred_imgs))

            tv_loss = total_variation_loss(pred_imgs)

            class_loss = loss_fn(logits, target_labels)

            loss = 12 * class_loss + 5 * tv_loss

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * imgs.size(0)
            epoch_samples += imgs.size(0)
        
        avg_loss = epoch_loss / epoch_samples
        if epoch % log_every == 0 or epoch == 1:
            print(
                f'epoch: {epoch}/{epochs}:\n'
                f'\t AVG: \t\t\t{avg_loss}\n'
                f'\t Classification: \t{class_loss}\n'
                f'\t Tot. Variation: \t{tv_loss}\n'
                f'\t TOTAL: \t\t{loss}\n'
            )

            save_training_preview(
                model=model,
                classifier=classifier,
                input_images=fixed_images,
                input_labels=fixed_labels,
                epoch=epoch,
                average_loss=avg_loss,
                output_directory=prev_dir
            )

    print(f'TRAINING FINISHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    return avg_loss


def test(
    model: nn.Module,
    classifier: MLP,
    loader,
    loss_fn: nn.CrossEntropyLoss,
    device
) -> tuple[float, float]:
    classifier.eval()
    classifier.requires_grad_(False)
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    correct = 0

    print(f'TEST STARTED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    with inference_mode():
        for i, (imgs, labels) in enumerate(loader, start=1):
            imgs = normalize_minst(imgs)
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred_imgs = model(imgs)

            logits = classifier((pred_imgs - 0.1307) / 0.3081)
            target_labels = (labels + 1) % 10
            loss = loss_fn(logits, target_labels)

            pred_labels = logits.argmax(dim=1)
            correct += (pred_labels == target_labels).sum().item()

            batch_size = pred_imgs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if i % 10 == 0 or i == 1:
                print(f'batch - {i} | loss: {loss}')

    print(f'TEST FINISHED AT {datetime.now().astimezone().strftime("%H:%M:%S")}')
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy


def main_full():
    device = dvc(
        'cuda' if cuda.is_available() else 'cpu'
    )
    print(f'using device: {device}')

    test_mnist = torchvision.datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # Normalize when needed!
            # torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    test_loader = DataLoader(
        dataset=test_mnist,
        batch_size=64,
        shuffle=False
    )

    classifier = MLP()
    state = load(
        f'{MODELS_DIR}/mnist_classifier.pt',
        map_location=device,
        weights_only=True
    )
    classifier.load_state_dict(state)
    classifier.to(device)
    classifier.eval()

    decoder = ConditionalDecoder(label_dim=8)
    state_dec = load(
        f'{MODELS_DIR}/vae_decoder.pt',
        map_location=device,
        weights_only=True
    )
    decoder.load_state_dict(state_dec)
    decoder.to(device)
    decoder.eval()

    prev_dir = Path(__file__).resolve().parent / 'logs' / 'predicted_successors'
    prev_dir.mkdir(parents=True, exist_ok=True)
    for file in prev_dir.iterdir():
        remove(file)

    with inference_mode():
        for i, (imgs, labels) in enumerate(test_loader, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = classifier((imgs - 0.1307) / 0.3081)
            predicted_labels = logits.argmax(dim=1)
            successor_labels = (predicted_labels + 1) % 10

            z = randn(imgs.size(0), 16, device=device)
            reconstructed_imgs = decoder(z, successor_labels)

            if i % 10 == 0 or i == 1:
                print(f'batch - {i:03d} | predicted label: {predicted_labels[0]} (was {labels[0]})')
                
                save_vae_reconstruction_inference_preview(
                    input_images=imgs[:5],
                    rec_images=reconstructed_imgs[:5],
                    labels=labels[:5],
                    target_labels=successor_labels[:5],
                    batch=i,
                    loss=float('nan'),
                    output_directory=prev_dir
                )

def main_vae():
    device = dvc(
        'cuda' if cuda.is_available() else 'cpu'
    )
    print(f'using device: {device}')

    train_mnist = torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    train_loader = DataLoader(
        dataset=train_mnist,
        batch_size=64,
        shuffle=True
    )

    epochs = 100
    avg, KL, rec = pretrain_vae(train_loader, device, epochs=epochs)

    log_file = Path(__file__).stem
    with open(f'logs/train/train_{log_file}.log', 'a', encoding='utf-8') as logfile:
        logfile.write(
            f'[{datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S")}] '
            f'Average loss: \t\t\t{avg:.4f}\n'
            f'KL loss: \t\t\t{KL:.4f}\n'
            f'Reconstruction loss: \t{rec:.4f}\n'
        )

if __name__ == '__main__':
    # main_vae()
    # test_mnist = torchvision.datasets.MNIST(
    #     './data',
    #     train=False,
    #     download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         # Normalize when needed!
    #         # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # )
    # test_loader = DataLoader(
    #     dataset=test_mnist,
    #     batch_size=64,
    #     shuffle=False
    # )

    # device = dvc(
    #     'cuda' if cuda.is_available() else 'cpu'
    # )
    # print(f'using device: {device}')

    # vae_inference(test_loader, device, generation=2)
    main_full()