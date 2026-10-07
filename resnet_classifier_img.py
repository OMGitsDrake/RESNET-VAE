import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import (
    cuda,
    inference_mode,
    device as dvc,
    load,
    save
)
from mlp_classifier import MLP
from datetime import datetime, UTC
from pathlib import Path
from utility import save_training_preview, total_variation_loss, MODELS_DIR
from os import remove
from definitions import ResidualBlock

class ResNETSEQ(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.res_blocks = nn.Sequential(
            # Considerare di aumentare gli shortcut e subito a 32
            ResidualBlock(32, 32),

            ResidualBlock(32, 32),

            ResidualBlock(32, 64),

            ResidualBlock(64, 64),

            ResidualBlock(64, 32),

            ResidualBlock(32, 32),

            ResidualBlock(32, 32),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.input_layer(x)

        out = self.res_blocks(out)

        return self.output_layer(out)


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
    for file in prev_dir.iterdir():
        remove(file)

    print(f'TRAINING STARTED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    avg_loss = float('nan')
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_samples = 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            target_labels = (labels + 1) % 10

            optim.zero_grad()

            pred_imgs = model(imgs)
            logits = classifier((pred_imgs - 0.1307) / 0.3081)

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

    print(f'TRAINING FINISHED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
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

    print(f'TEST STARTED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    with inference_mode():
        for i, (imgs, labels) in enumerate(loader, start=1):
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

    print(f'TEST FINISHED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy

def main():
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
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
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

    model = ResNETSEQ().to(device)

    loss_fn = nn.CrossEntropyLoss()

    saved_dict = Path(f'{MODELS_DIR}/mnist_resnet.pt').resolve()
    if not saved_dict.exists():
        train_mnist = torchvision.datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        train_loader = DataLoader(
            dataset=train_mnist,
            batch_size=64,
            shuffle=True
        )

        epochs = 25
        lr = 0.005
        optim = SGD(model.parameters(), lr=lr)

        last_epoch_loss = train(
            model=model,
            classifier=classifier,
            loader=train_loader,
            optim=optim,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
            log_every=5
        )

        log_file = Path(__file__).stem
        with open(f'Lecture 3/logs/train_{log_file}.log', 'a', encoding='utf-8') as logfile:
            logfile.write(
                f'[{datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")}] '
                f'[{model.__class__.__name__}] - Last epoch loss: {last_epoch_loss}\n'
            )

        save(
            model.state_dict(),
            f'{MODELS_DIR}/mnist_resnet.pt'
        )

    state = load(
        f'{MODELS_DIR}/mnist_resnet.pt',
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device=device)

    avg_loss, accuracy = test(model, classifier, test_loader, loss_fn, device)

    log_file = Path(__file__).stem
    with open(f'Lecture 3/logs/test_{log_file}.log', 'a', encoding='utf-8') as logfile:
        logfile.write(
            f'[{datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")}] '
            f'[{model.__class__.__name__}]\n'
            f'\tAVG test loss: {avg_loss}\n'
            f'\tTest Accuracy: {accuracy}\n'
        )

    print(f'AVG LOSS: {avg_loss}\nACCURACY: {accuracy}')

if __name__ == '__main__':
    main()
