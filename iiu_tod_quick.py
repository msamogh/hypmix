from datasets import load_dataset

# Load the dataset
dataset = load_dataset("msamogh/indirect-requests")

# Get the number of samples in each split
train_samples = len(dataset['train'])
validation_samples = len(dataset['validation'])
test_samples = len(dataset['test'])

# Print the number of samples
print(f"Number of samples in the train set: {train_samples}")
print(f"Number of samples in the validation set: {validation_samples}")
print(f"Number of samples in the test set: {test_samples}")
