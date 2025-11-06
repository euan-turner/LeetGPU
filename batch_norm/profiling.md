Initial ideas:
- Use shared memory for parallel reduction
- Thread block per channel for easy synchronisation and sharing
- 

v1:
Assign a thread block to a channel
Perform separate mean and variance reductions in shared memory,
in parallel over the block. Use tree reduction to coalesce accesses.
Warp shuffle for final 32 elements to reduce divergence.

NCU reported:
- Under-utilised SMs, launching max. 64 blocks, with 170 SMs available
- Long scoreboard stalls
- Limited theoretical occupancy due to volume of shared memory

v2:
Calculate mean and variance in one pass, with E[X] - E[x^2]
Tile the reductions to reduce the shared memory required

NCU reported:
...