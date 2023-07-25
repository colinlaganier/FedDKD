import os
import random
import shutil
import argparse
from torchvision.datasets import ImageFolder

def split_dataset(num_split):
    # Get the total number of samples in the dataset

    data_dir = "cinic-10/valid/"
    out_dir = "cinic-10/synthetic/"
    dataset = ImageFolder(data_dir)
    num_datasets = num_split
    total_samples = len(dataset)

    # Shuffle the indices to randomize the data
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Calculate the number of samples in each smaller dataset
    samples_per_dataset = total_samples // num_datasets

    # Split the indices into smaller chunks
    split_indices = [indices[i * samples_per_dataset: (i + 1) * samples_per_dataset] for i in range(num_datasets)]

    # Create directories for each smaller dataset
    output_dirs = [os.path.join(out_dir, f"round_{i}") for i in range(num_datasets)]
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    # Move the files to their respective smaller datasets
    for i, indices in enumerate(split_indices):
        for idx in indices:
            file_path, class_id = dataset.samples[idx]
            class_name = dataset.classes[class_id]
            dst_dir = os.path.join(output_dirs[i], class_name)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(file_path, dst_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-split", type=int, default=20)
    
    args = parser.parse_args()
    
    split_dataset(args.num_split)
    print("Dataset successfully split")
