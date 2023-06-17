import os

def create_output_directories(path='outputs'):
    """
    Creates output directories for storing results and plots.

    Args:
        path (str): Path to the parent directory where output directories will be created.
                    Defaults to 'outputs'.

    Returns:
        str: Path of the newly created outputs directory.

    Raises:
        OSError: If there is an error in creating the directories.

    Example:
        >>> create_output_directories()
        Output will be in outputs/outputs-1
        'outputs/outputs-1'
    """
    # Find the latest number used in 'outputs' directory
    max_output_num = 0
    for name in os.listdir(path):
        if name.startswith('outputs-') and os.path.isdir(os.path.join(path, name)):
            output_num = int(name.split('-')[1])
            max_output_num = max(max_output_num, output_num)
    
    # Increment the number for the new outputs directory
    new_output_num = max_output_num + 1
    new_outputs_dir = os.path.join(path, f'outputs-{new_output_num}')
    plots_dir = os.path.join(new_outputs_dir, 'plots')
    
    os.makedirs(new_outputs_dir)
    
    os.makedirs(plots_dir)
    print(f"Output will be in {new_outputs_dir}")
    return new_outputs_dir

