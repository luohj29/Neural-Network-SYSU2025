import shutil
import os

def get_weight_path(ckpt_path):
    """
    Retrieves the path to the latest weight file from a given checkpoint directory.

    Args:
        ckpt_path (str): The path to the checkpoint directory.

    Returns:
        str or None: The full path to the latest weight file if the directory is not empty,
                     otherwise None. If the provided path is not a directory, returns None.

    Notes:
        - The function assumes that the filenames in the directory contain a numeric value
          that can be extracted and used for sorting.
        - Sorting is performed based on the numeric value extracted from the filenames,
          specifically from the portion after the last '=' and before the first '-'.
    """

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None


def get_weight_list(ckpt_path,choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None and eval(str(fold.name)[-1]) not in choice:
            continue
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            # print(weight_path)
            if len(weight_path) > 0:
                weight_path.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
                path_list.append(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list


def remove_weight_path(ckpt_path,retain=5):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain=retain)
        else:
            remove_weight_path(ckpt_path,retain=retain)
            break  


if __name__ == '__main__':
    ckpt_path = './ckpt/'
    dfs_remove_weight(ckpt_path)
    