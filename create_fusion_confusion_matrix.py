import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


cm = np.array([[123, 40], [42, 147]])
labels = ["Control", "Alzheimer's"]

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=180)
im = ax.imshow(cm, cmap="Blues")

ax.set_title("Fusion Model Confusion Matrix", fontsize=15, pad=14)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(range(2), labels=labels)
ax.set_yticks(range(2), labels=labels)

threshold = cm.max() / 2
for row in range(2):
    for col in range(2):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
            fontsize=18,
            color="white" if cm[row, col] > threshold else "black",
        )

fig.colorbar(im, ax=ax, label="Number of recordings", fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("fusion_confusion_matrix.png", bbox_inches="tight")
plt.close(fig)
print("Saved fusion_confusion_matrix.png")
