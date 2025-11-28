from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------------------
# STEP 1: Ask MPI to pick a 2D decomposition (Px, Py)
# ----------------------------------------
dims = MPI.Compute_dims(size, 2)   # same as MPI_Dims_create in C

Px, Py = dims[0], dims[1]

# ----------------------------------------
# STEP 2: Create a 2D Cartesian communicator
# ----------------------------------------
periods = [False, False]  # no wraparound
reorder = True

cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)

coords = cart_comm.Get_coords(rank)
px, py = coords[0], coords[1]

# ----------------------------------------
# STEP 3: Create a dummy 3D data array on each rank
# ----------------------------------------
# Let's pretend our full domain is 30 × 30 × 30
Nx, Ny, Nz = 129, 153, 20

# For 2D process grid, we decompose along x and y only:
local_Nx = Nx // Px
local_Ny = Ny // Py
local_Nz = Nz  # no decomposition in z

# Dummy data block: fill it with rank number
local_data = np.full((local_Nx, local_Ny, local_Nz), rank, dtype=int)

# ----------------------------------------
# STEP 4: Each rank prints info (good for understanding layout)
# ----------------------------------------
print(f"Rank {rank:02d} | coords = ({px}, {py}) "
      f"| local shape = {local_data.shape} "
      f"| Px={Px}, Py={Py}")

MPI.Finalize()

