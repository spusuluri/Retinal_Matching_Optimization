import json
import matplotlib.pyplot as plt
import numpy as np

metrics = json.loads(open('metrics_final_fourier_rings.json').read())
metrics_baseline = json.loads(open('metrics_baseline.json').read())
metrics_list = [metrics_baseline, metrics]


def plot_losses():
    for key in list(metrics.keys()):
        overall_loss = metrics[key]['overall_loss']
        detector_loss = metrics[key]['detector_loss']
        descriptor_loss = metrics[key]['descriptor_loss']
        linestyle = "solid"
        postfix = ""
        if key == "val":
            linestyle = "dashed"
            postfix = " (val)"
        plt.plot(overall_loss, label="Overall Loss" + postfix, color='blue', linestyle=linestyle)
        plt.plot(detector_loss, label="Detector Loss" + postfix, color='red', linestyle=linestyle)
        plt.plot(descriptor_loss, label="Descriptor Loss" + postfix, color='green', linestyle=linestyle)

    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def plot_weights():
    startw = softmax(metrics['train']['fourier_weights'][0])
    endw = softmax(metrics['train']['fourier_weights'][-1])
    print(endw)
    radii = ["h//128", "h//64", "h//32", "h//16", "h//8", "h//4",]
    plt.bar(radii, endw)
    plt.xlabel("Mask Index")
    plt.ylabel("Probability")
    plt.title("Ending Weight Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_losses()
    plot_weights()