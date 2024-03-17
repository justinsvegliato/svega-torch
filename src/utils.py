import torch


def train_model(model, dataloader, loss_function, optimizer):
        model.train()

        dataset_size = len(dataloader.dataset)

        for batch_id, (X, y) in enumerate(dataloader):
            y_hat = model(X)
            loss = loss_function(y_hat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_id % 100 == 0:
                loss = loss.item()
                current_training_example_id = batch_id * dataloader.batch_size + len(X)
                print(f"Loss: {loss:>7f} [{current_training_example_id:>5d}{dataset_size:>5d}]")


def test_model(model, dataloader, loss_function):
    model.eval()

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    cumulative_loss = 0
    num_correct = 0

    with torch.no_grad():
         for X, y in dataloader:
              y_hat = model(X)
              cumulative_loss += loss_function(y_hat, y).item()
              num_correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    accuracy = (num_correct / dataset_size) * 100
    average_batch_loss = cumulative_loss / num_batches

    print(f"Accuracy: {(accuracy):>0.1f}%")
    print(f"Average Loss: {average_batch_loss:>8f}")
