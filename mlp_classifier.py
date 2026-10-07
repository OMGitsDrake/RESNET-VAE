import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from datetime import datetime, UTC
from pathlib import Path
from utility import MODELS_DIR

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 10),
        )

    def forward(self, x):
        return self.net(x)

def train(
    model: MLP,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.SGD,
    device: torch.device,
    epochs: int = 100,
    log_every: int = 10,
) -> float:
    model.train()

    print(f'TRAIN STARTED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    avg_loss = float('nan')
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_samples = 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            pred = model(imgs)

            loss = loss_fn(pred, labels)
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * imgs.size(0)
            epoch_samples += imgs.size(0)

        avg_loss = epoch_loss / epoch_samples
        if epoch % log_every == 0 or epoch == 1:
            print(f'epoch: {epoch}/{epochs} - loss: {avg_loss:.6f}')

    print(f'TRAIN FIINSHED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    return avg_loss

def test(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    print(f'TEST STARTED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.inference_mode():
        for i, (imgs, labels) in enumerate(loader, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)

            loss = loss_fn(logits, labels)

            batch_size = logits.size(0)
            total_loss += loss.item() * batch_size

            predicted_labels = logits.argmax(dim=1)

            correct += (predicted_labels == labels).sum().item()

            total_samples += batch_size

            if i % 10 == 0 or i == 1:
                print(f'batch - {i} | loss: {loss}')

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    print(f'TEST FIINSHED AT {datetime.now(UTC).strftime("%H:%M:%S")}')
    return avg_loss, accuracy

def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
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

    loss_fn = nn.CrossEntropyLoss()
    model = MLP().to(device)

    saved_dict = Path(f'{MODELS_DIR}/mnist_classifier.pt').resolve()
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
        lr = 0.1
        epochs = 500
        optim = torch.optim.SGD(model.parameters(), lr=lr)
        last_epoch_loss = train(model, train_loader, loss_fn, optim, device, epochs)

        log_file = Path(__file__).stem
        with open(f'Lecture 3/logs/train_{log_file}.log', 'a', encoding='utf-8') as logfile:
            logfile.write(
                f'[{datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")}] '
                f'[{model.__class__.__name__}] - Last epoch loss: {last_epoch_loss}\n'
            )


        torch.save(
            model.state_dict(),
            f'{MODELS_DIR}/mnist_classifier.pt'
        )

    state = torch.load(
        f'{MODELS_DIR}/mnist_classifier.pt',
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device=device)

    avg_loss, accuracy = test(model, test_loader, loss_fn, device)

    log_file = Path(__file__).stem
    with open(f'Lecture 3/logs/test_{log_file}.log', 'a', encoding='utf-8') as logfile:
        logfile.write(
            f'[{datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")}] '
            f'[{model.__class__.__name__}]\n'
            f'\tAVG test loss: {avg_loss}\n'
            f'\tTest Accuracy: {accuracy}\n'
        )

    print(f'AVG test loss: {avg_loss}')
    print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    main()
