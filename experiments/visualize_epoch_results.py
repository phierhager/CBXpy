import matplotlib.pyplot as plt

def visualize_results():
    # Data for QCBO
    epochs_qcbo = [1, 2, 3, 4, 5, 6]
    accuracy_qcbo = [0.4225, 0.4899, 0.5282, 0.5646, 0.5875, 0.5875]

    # Data for CBO
    epochs_cbo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    accuracy_cbo = [0.526, 0.6219, 0.6877, 0.72, 0.7574, 0.7655, 0.781, 0.7884, 0.7929, 0.7947]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_qcbo, accuracy_qcbo, label='QCBO', marker='o', linestyle='-')
    plt.plot(epochs_cbo, accuracy_cbo, label='CBO', marker='o', linestyle='-')
    
    # Adding titles and labels
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Show plot
    plt.grid(True)
    plt.savefig('results/images/accuracy_over_epochs.png')
    plt.show()

# Call the function to visualize the results
visualize_results()
