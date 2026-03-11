import torch
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def train_unet(model, train_loader, device='cpu', epochs=20, lr=0.0005, save_plots=None, plot_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for i, (images, annotations, dist_maps) in enumerate(train_loader):
            images, annotations, dist_maps = images.to(device), annotations.to(device), dist_maps.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # shape => (batch, 1, H, W)

            # Optional visualization every 5 epochs on the first batch
            if save_plots and i == 0 and (epoch % plot_every == 0):
                # We'll visualize the first sample in the batch
                img_np = images[0, 0].detach().cpu().numpy()
                ann_np = annotations[0, 0].detach().cpu().numpy()
                dist_gt_np = dist_maps[0, 0].detach().cpu().numpy()
                dist_pred_np = outputs[0, 0].detach().cpu().numpy()

                # Skeletonization: threshold the predicted distance, then skeletonize
                threshold = 0.8
                dist_thresh = dist_pred_np > threshold
                skeleton = skeletonize(dist_thresh)

                plt.clf()
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                # (1) Raw image (grayscale)
                axes[0, 0].imshow(img_np, cmap='gray')
                axes[0, 0].set_title("Raw Image")
                axes[0, 0].axis("off")
                plt.colorbar(axes[0, 0].imshow(img_np, cmap='gray'), ax=axes[0, 0])

                # (2) Annotation (binary mask)
                im = axes[0, 1].imshow(dist_gt_np, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title("Annotation (distance)")
                axes[0, 1].axis("off")
                plt.colorbar(im)

                # (3) Distance map (rainbow colormap)
                im = axes[1, 0].imshow(dist_pred_np, cmap='gray', vmin=0, vmax=1)  # or "turbo", "rainbow", etc.
                axes[1, 0].set_title("Predicted Distance")
                axes[1, 0].axis("off")
                plt.colorbar(im)

                # (4) Skeleton of thresholded distance
                axes[1, 1].imshow(skeleton, cmap='gray')
                axes[1, 1].set_title(f"Skeleton (dist > {threshold})")
                axes[1, 1].axis("off")
                plt.colorbar(axes[1, 1].imshow(skeleton, cmap='gray'), ax=axes[1, 1])

                plt.tight_layout()
                plt.savefig(f'{save_plots}/{epoch:05d}.png')
                plt.close()


            # Use a regression loss (MSE) for distance maps
            loss = torch.sum((0.1 + dist_maps) * (outputs - dist_maps) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
