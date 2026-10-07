from matplotlib import pyplot as plt
from torch import inference_mode

MODELS_DIR = 'trained_models'

def save_training_preview(
    model,
    classifier,
    input_images,
    input_labels,
    epoch,
    average_loss,
    output_directory,
):
    model.eval()
    classifier.eval()

    with inference_mode():
        generated_images = model(input_images)
        predicted_labels = classifier(
            generated_images
        ).argmax(dim=1)

    # Denormalization
    input_display = (
        input_images * 0.3081 + 0.1307
    ).clamp(0, 1)

    generated_display = (
        generated_images #* 0.3081 + 0.1307
    ).clamp(0, 1)

    number_of_images = input_images.size(0)

    figure, axes = plt.subplots(
        2,
        number_of_images,
        figsize=(2 * number_of_images, 4),
    )

    for index in range(number_of_images):
        expected_label = (
            input_labels[index].item() + 1
        ) % 10

        axes[0, index].imshow(
            input_display[index, 0].cpu(),
            cmap="gray",
        )
        axes[0, index].set_title(
            f"Input: {input_labels[index].item()}"
        )
        axes[0, index].axis("off")

        axes[1, index].imshow(
            generated_display[index, 0].cpu(),
            cmap="gray",
        )
        axes[1, index].set_title(
            f"Expected: {expected_label}\n"
            f"Predicted: {predicted_labels[index].item()}"
        )
        axes[1, index].axis("off")

    figure.suptitle(
        f"Epoch {epoch} — average loss: {average_loss:.6f}"
    )
    figure.tight_layout()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_directory / f"epoch_{epoch:04d}.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Continue training
    model.train()