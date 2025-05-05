import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score

from decision_tree import ConvNet, DecisionTree, get_features_and_labels, get_features_and_paths
from CNN import CNN, train, validate, test
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset, plot

"""
Notice:
    1) You can't add any additional package
    2) You can ignore the suggested data type if you want
"""

def main():
    """
    load data
    """
    #logger.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    train_len = int(0.8 * len(images))

    train_images, val_images = images[:train_len], images[train_len:]
    train_labels, val_labels = labels[:train_len], labels[train_len:]
    
    test_images = load_test_dataset()

    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = TrainDataset(val_images, val_labels)
    test_dataset = TestDataset(test_images)
    
    """
    CNN - train and validate
    """
    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=0.001)

    train_losses = []
    val_losses = []
    max_acc = 0
    maxe = 0
    EPOCHS = 10
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if val_acc > max_acc:
            max_acc = val_acc
            maxe = epoch+1
            torch.save(model, 'cnn_model.pt')
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # print(f"{train_loss:.4f}, {val_loss:.4f}, {val_acc:.4f}")
        # Print the training log to help you monitor the training process
        # You can save the model for future usage

    logger.info(f"Best Accuracy: {max_acc:.4f}")#, at {maxe}th")
    """
    CNN - plot
    """
    plot(train_losses, val_losses, EPOCHS)

    """
    CNN - test
    """

    model = torch.load('cnn_model.pt', weights_only=False)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test(model, test_loader, criterion, device)

    """
    Decision Tree - grow tree and validate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Start training Decision Tree")

    conv_model = ConvNet().to(device)
    tree_model = DecisionTree(max_depth=7)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_features, train_labels = get_features_and_labels(conv_model, train_loader, device)
    tree_model.fit(train_features, train_labels)

    val_features, val_labels = get_features_and_labels(conv_model, val_loader, device)
    val_predictions = tree_model.predict(val_features)

    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.info(f"Validation Accuracy: {val_accuracy*100:.2f}%")

    """
    Decision Tree - test
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_features, test_paths = get_features_and_paths(conv_model, test_loader, device)

    test_predictions = tree_model.predict(test_features)

    results = []
    for image_name, prediction in zip(test_paths, test_predictions.cpu().numpy()):
        results.append({'id': image_name, 'prediction': prediction})
    df = pd.DataFrame(results)
    df.to_csv('DecisionTree.csv', index=False)
    print(f"Predictions saved to 'DecisionTree.csv'")


if __name__ == '__main__':
    main()
