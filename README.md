# Dual side sparse tensorcore [LINK](https://dl.acm.org/doi/10.1109/ISCA52012.2021.00088) implementation in python

## Sections of this code conatins: 
1. Matrix creation:
    - First: creates matrices A, B, C
    - Second: make sparse matrices with specific ratio defined in config file
    - Third: condense matrices in row and column major and save bitmaps

2. Cpu implementation of mma 
    - This section is computes mma of matrix with 3 simple for loop

3. GPU implementation of dense tensorcore
    - This section is computes mma of matrix with tiling size 32X32XK
    - Hierarchy of computation is ([LINK](https://ieeexplore.ieee.org/document/8695642)):
        - Subcore
        - Tensorcore
        - Octet
        - Threadgroup
        - FEOP

4. GPU implementation of sparse tensorcore
    - This section is computes mma of matrix with tiling size 32X32XK
    - Three major parts of this computation is: 
        - For each warp, computes popc of vectorA and popc of vectorB to determines which OHMMAs active
        - Computes outer product of vectorA(32X1) and vectorB(32X1) to Gather a section of matrixC
        - Accumultes section of matrixC and result of OHMMAs
        - Scatter and writeback the result into memory
    - Hierarchy of computation is ([LINK](https://ieeexplore.ieee.org/document/8695642)): 
        - Subcore
        - Tensorcore
        - Octet
        - Threadgroup
        - FEOP

---

<small>Written by S.M.R</small>