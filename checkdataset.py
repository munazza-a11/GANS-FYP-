import os

# Point to the dataset
data_dir = "./data/tiny-imagenet-200"

# Check training set
train_dir = os.path.join(data_dir, "train")
train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
print(f"Train classes: {len(train_classes)}")
print(f"Sample classes: {train_classes[:5]}")

# Count images in train
total_train_images = sum(
    len(files) for _, _, files in os.walk(train_dir)
)
print(f"Total train images: {total_train_images}")

# Check test set
test_dir = os.path.join(data_dir, "test")
test_images = [f for f in os.listdir(test_dir) if f.endswith(".JPEG")]
print(f"Total test images: {len(test_images)}")
