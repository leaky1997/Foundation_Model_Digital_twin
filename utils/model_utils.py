
def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows

# 反归一化处理
def denormalize(x, means, stdev):
    """反归一化处理"""
    x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
    x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
    return x


