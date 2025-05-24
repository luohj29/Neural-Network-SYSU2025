import os
import glob
import pandas as pd

def make_csv_cub(input_path, csv_path, index_path = None):
    '''
    Make CUB 200 2011 csv file.
    '''
    
    info = []
    for subdir in os.scandir(input_path): # 
        label = int(subdir.name.split('.')[0])
        path_list = glob.glob(os.path.join(subdir.path, "*.jpg")) 
        sub_info = [[item, label] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data['label'] = info_data['label'] - 1

    # if index_path is not None:
    #     index = pd.read_csv(index_path, header=None, sep=' ').loc[:,1]
    #     index = input_path + index
    #     info_data.index = info_data['id']
    #     info_data = info_data.loc[index]
        
    info_data.to_csv(csv_path, index=False)
    
def split_csv_cub(split_path, csv_path):
    '''
    Split CUB 200 2011 csv file.
    '''
    
    split = pd.read_csv(split_path, header=None, sep=' ').loc[:,1]
    info_data = pd.read_csv(csv_path)
 
    train_data = info_data.loc[split == 1]
    test_data = info_data.loc[split == 0]
    
    train_data.to_csv(csv_path + '_train.csv', index=False)
    test_data.to_csv(csv_path + '_test.csv', index=False)

def make_train_test_csv(input_path, train_csv_path, test_csv_path, split_path=None, train_ratio=0.8, random_seed=42):
    '''
    Create train and test CSV files directly from the input directory.
    
    Parameters:
    -----------
    input_path : str
        Path to the directory containing image subdirectories
    train_csv_path : str
        Path to save the train CSV file
    test_csv_path : str
        Path to save the test CSV file
    split_path : str, optional
        Path to the split file (if None, random split will be used)
    train_ratio : float, optional
        Ratio of data to use for training (default: 0.8)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    '''
    
    # Collect all image paths and labels
    info = []
    for subdir in os.scandir(input_path):
        if subdir.is_dir():
            label = int(subdir.name.split('.')[0])
            path_list = glob.glob(os.path.join(subdir.path, "*.jpg"))
            sub_info = [[item, label] for item in path_list]
            info.extend(sub_info)
    
    # Create DataFrame
    col = ['id', 'label']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data['label'] = info_data['label'] - 1
    
    # Split data into train and test sets
    if split_path is not None:
        # Use provided split file
        split = pd.read_csv(split_path, header=None, sep=' ').loc[:,1]
        train_data = info_data.loc[split == 1]
        test_data = info_data.loc[split == 0]
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            info_data, 
            train_size=train_ratio, 
            random_state=random_seed,
            stratify=info_data['label']  # Ensure balanced class distribution
        )
    
    # Save to CSV files
    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)
    
    print(f"Created train set with {len(train_data)} samples")
    print(f"Created test set with {len(test_data)} samples")
    
    return train_data, test_data

if __name__ == "__main__":
    
    # make csv file
    # if not os.path.exists('./csv_file'):
    #     os.makedirs('./csv_file')

    input_path = './datasets/CUB_200_2011/CUB_200_2011/images/'
    csv_path = './csv_file/cub_200_2011.csv'
    index_path = './datasets/CUB_200_2011/CUB_200_2011/images.txt'
    make_csv_cub(input_path, csv_path, index_path)
    
    # split csv file
    split_path = './datasets/CUB_200_2011/CUB_200_2011/train_test_split.txt'
    split_csv_cub(split_path, csv_path)
    
    # Example of using the new function
    train_csv_path = './csv_file/cub_200_2011.csv_train.csv'
    test_csv_path = 'csv_file/cub_200_2011.csv_test.csv'
    make_train_test_csv(input_path, train_csv_path, test_csv_path, split_path)
    
    # Example of random split without using split file
    # make_train_test_csv(input_path, './csv_file/cub_200_2011_train_random.csv', './csv_file/cub_200_2011_test_random.csv')

    
