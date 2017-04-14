# comp-photo-assignment5

#### Task 1: Seam Carving
For this task, I had to implement seam carving and seam expansion. The algorithm implemented is essentially the exact same as in the paper, except for the fact that there's no optimal seam ordering for expansion/removal.

Notably, for carving horizontal seams, I just transposed the image and applied vertical seam carving.

For seam expansion, to prevent the top k seams from overlapping, whenever I would find an optimal seam I would set the energy at that location to be infinite. This would ensure that the next k-1 seams will not overlap.
