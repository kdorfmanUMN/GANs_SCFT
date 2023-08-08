def Cov3d(input_size, in_channel, out_channel, kernel, stride, padding):
    [N,C,D,H,W] = input_size
    D = (D + 2*padding - 1*(kernel-1)-1)/stride +1
    H = D
    W = D
    output_size = [N,C,D,H,W]
    return output_size

