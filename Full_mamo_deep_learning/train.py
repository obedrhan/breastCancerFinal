import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, device, save_path, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch [{epoch+1}/{num_epochs}] -------------------------")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 🔍 Print every few batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f" 🧪 Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"✅ Epoch [{epoch+1}] Completed | Avg Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")