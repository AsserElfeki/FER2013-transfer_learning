import os

def create_output_directories(path='outputs'):
    
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

