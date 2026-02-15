import numpy as np
import math


# Convert ANY ND array padded 5D tensor (1,1,D,256,256)
def to_5d_and_pad(arr, H=256, W=256):
    original_shape = arr.shape
    flat = arr.flatten()
    N = flat.size  # number of elements

    # Number of slices (D) needed to fit everything into D*x*y blocks
    D = math.ceil(N / (H * W))
    padded_length = D * H * W

    # Create padded array
    padded = np.zeros(padded_length, dtype=arr.dtype)
    padded[:N] = flat  # copy original linear data

    # Convert to 5D
    padded_5d = padded.reshape(1, 1, D, H, W)

    info = {
        "original_shape": original_shape,
        "original_length": N,
        "padded_shape": (1, 1, D, H, W),
        "H": H,
        "W": W
    }

    return padded_5d, info


def restore_from_5d(padded_5d, info):
    flat = padded_5d.flatten()
    N = info["original_length"]
    original_shape = info["original_shape"]
    trimmed = flat[:N]
    restored = trimmed.reshape(original_shape)
    return restored


def generate_test_shapes():
    shapes = []

    shapes += [
        (10,),
        (9999,),
        (524289,),
        (90121933,),
    ]

    shapes += [
        (2599, 531),
        (75000, 89),
        (1000, 257),
        (2267, 257),
    ]

    shapes += [
        (8, 256, 256),
        (31, 129, 257),
        (2, 3, 128),
        (9000, 20, 20),
    ]

    shapes += [
        (4, 6, 128, 128),
        (100, 10, 1, 1000),
        (2, 3, 75, 89),
        (5, 7, 9, 11),
    ]

    for dims in range(5, 11):
        shape = tuple(np.random.randint(2, 10) for _ in range(dims))
        shapes.append(shape)

    return shapes



def run_tests():
    shapes = generate_test_shapes()

    print("\n==================== ND PADDING TEST  =======================\n")

    for shape in shapes:
        arr = np.random.rand(*shape)

        print("---------------------------------------------------------------")
        print(f"TESTING ARRAY SHAPE: {shape}")
        print(f"Total elements: {arr.size}")

        padded, info = to_5d_and_pad(arr)

        print("Padded 5D shape:", info["padded_shape"])

        restored = restore_from_5d(padded, info)

        print("Restored shape:", restored.shape)

        correct = np.allclose(arr, restored)

        print("Restoration correct?:", correct)

        if arr.size < 256 * 256:
            print("Compression worth it?: NO (block too small)")
        else:
            print("Compression worth it?: YES")

        print("RESULT:", "PASS" if correct else "FAIL")
        print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    run_tests()
