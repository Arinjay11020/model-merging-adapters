#!/usr/bin/env python3
"""
Download all splits of the GSM8K dataset from Hugging Face.
GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems.
"""

import os
import json
from datasets import load_dataset
from typing import Optional

def download_gsm8k(cache_dir: Optional[str] = None, save_local: bool = False, output_dir: str = "gsm8k_data"):
    """
    Download all splits of the GSM8K dataset from Hugging Face.
    
    Args:
        cache_dir: Directory to cache the dataset (optional)
        save_local: Whether to save the dataset locally as JSON files
        output_dir: Directory to save local copies (if save_local=True)
    """
    
    print("Downloading GSM8K dataset from Hugging Face...")
    
    try:
        # Load the dataset with all splits
        dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
        
        print(f"Successfully loaded GSM8K dataset!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Print information about each split
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} split:")
            print(f"  - Number of examples: {len(split_data)}")
            if len(split_data) > 0:
                print(f"  - Features: {list(split_data.features.keys())}")
                print(f"  - Example question: {split_data[0]['question'][:100]}...")
        
        # Save locally if requested
        if save_local:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving dataset locally to {output_dir}/...")
            
            for split_name, split_data in dataset.items():
                output_file = os.path.join(output_dir, f"{split_name}.json")
                
                # Convert to list of dictionaries for JSON serialization
                data_list = []
                for example in split_data:
                    data_list.append({
                        "question": example["question"],
                        "answer": example["answer"]
                    })
                
                # Save as JSON
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)
                
                print(f"  - Saved {len(data_list)} examples to {output_file}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading GSM8K dataset: {e}")
        raise

def explore_dataset(dataset):
    """
    Explore and display sample data from the dataset.
    """
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} SPLIT SAMPLE:")
        print("-" * 30)
        
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print()

def main():
    """Main function to download and explore GSM8K dataset."""
    
    # Configuration
    CACHE_DIR = None  # Use default cache directory
    SAVE_LOCAL = True  # Save as local JSON files
    OUTPUT_DIR = "gsm8k_Dataset"
    
    try:
        # Download the dataset
        dataset = download_gsm8k(
            cache_dir=CACHE_DIR,
            save_local=SAVE_LOCAL,
            output_dir=OUTPUT_DIR
        )
        
        # Explore the dataset
        explore_dataset(dataset)
        
        print(f"\n✅ Successfully downloaded GSM8K dataset!")
        if SAVE_LOCAL:
            print(f"📁 Local files saved to: {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import datasets
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
    
    # Run the main function
    exit(main())