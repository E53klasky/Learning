import numpy as np
import math

# -----------------------------------------
# Helper: pad any tensor to 5D (1,1,D,H,W)
# -----------------------------------------
def to_5d_and_pad(arr):
    original_shape = arr.shape
    flat = arr.flatten()
    N = flat.size

    # choose H=256 and W=256 unless too small
    W = 256
    H = 256

    # number of slices D needed
    D = math.ceil(N / (H * W))
    padded_length = D * H * W

    padded = np.zeros(padded_length, dtype=arr.dtype)
    padded[:N] = flat  # copy original into padded

    # reshape into 5D
    padded_5d = padded.reshape(1, 1, D, H, W)

    return padded_5d, original_shape, N, (1,1,D,H,W)


# -----------------------------------------
# Helper: restore 5D array to original shape
# -----------------------------------------
def from_5d_and_unpad(arr_5d, original_shape, original_N):
    flat = arr_5d.flatten()
    restored_flat = flat[:original_N]  # remove padding
    return restored_flat.reshape(original_shape)


# -----------------------------------------
# Test function
# -----------------------------------------
def test_tensor(arr):
    print("\n=============================================")
    print("TESTING ARRAY")
    print(f"Original shape: {arr.shape}")
    print(f"Original number of elements: {arr.size}")

    padded_5d, original_shape, N, padded_shape = to_5d_and_pad(arr)

    print(f"Padded 5D shape: {padded_shape}")
    print(f"Padded total elements: {padded_5d.size}")

    # "Decompress" (just an identity here)
    decompressed_5d = padded_5d.copy()

    restored = from_5d_and_unpad(decompressed_5d, original_shape, N)

    print(f"Restored shape: {restored.shape}")
    print("Restoration correct? ", np.allclose(restored, arr))
    print("=============================================")


# -----------------------------------------
# Generate many random test arrays
# -----------------------------------------
if __name__ == "__main__":
    tests = []

    # 1D tests
    tests.append(np.random.rand(90121933))
    tests.append(np.random.rand(257,2267,1))
    tests.append(np.random.rand(9999))
    tests.append(np.random.rand(524289))   # boundary case

    # 2D tests
    tests.append(np.random.rand(9000,20, 20))
    tests.append(np.random.rand(10,1000, 257))
    tests.append(np.random.rand(2599, 531))  # your example

    # 3D tests
    tests.append(np.random.rand(8, 256, 256))
    tests.append(np.random.rand(31, 129, 257))

    # 4D tests
    tests.append(np.random.rand(4, 6, 128, 128))
    tests.append(np.random.rand(100,10,1,1000))
    tests.append(np.random.rand(2, 3, 75, 89,))   # close to your data

    # Run tests
    for t in tests:
        test_tensor(t)

