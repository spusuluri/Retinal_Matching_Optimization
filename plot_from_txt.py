import re
import matplotlib.pyplot as plt
import json
with open("results.txt", "r") as file:
    data = file.read()

epochs = re.findall(r"Epoch (\d+)/\d+", data)
train_losses = re.findall(r"train overall loss: ([\d.]+)", data)
train_detector_losses = re.findall(r"train.*detector_loss: ([\d.]+)", data)
train_descriptor_losses = re.findall(r"train.*descriptor_loss: ([\d.]+)", data)
val_losses = re.findall(r"val overall loss: ([\d.]+)", data)
val_detector_losses = re.findall(r"val.*detector_loss: ([\d.]+) of \d+ nums", data)
val_descriptor_losses = re.findall(r"val.*descriptor_loss: ([\d.]+) of \d+ nums", data)

epochs = list(map(int, epochs))
train_losses = list(map(float, train_losses))
train_detector_losses = list(map(float, train_detector_losses))
train_descriptor_losses = list(map(float, train_descriptor_losses))
val_losses = list(map(float, val_losses))
val_detector_losses = list(map(float, val_detector_losses))
val_descriptor_losses = list(map(float, val_descriptor_losses))

out = {"train": {"overall_loss": train_losses, "detector_loss": train_detector_losses, "descriptor_loss": train_descriptor_losses}, "val": {"overall_loss": val_losses, "detector_loss": val_detector_losses, "descriptor_loss": val_descriptor_losses}}
json.dump(out, open("metrics_baseline.json", "w"))

# plt.plot(epochs, train_losses, label="Train Loss")
# plt.plot(epochs, train_detector_losses, label="Train Detector Loss")
# plt.plot(epochs, train_descriptor_losses, label="Train Descriptor Loss")
# plt.plot(epochs, val_losses, label="Validation Loss")
# plt.plot(epochs, val_detector_losses, label="Validation Detector Loss")
# plt.plot(epochs, val_descriptor_losses, label="Validation Descriptor Loss")

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss vs. Epoch")
# plt.legend()
# plt.show()
