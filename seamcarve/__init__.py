"""Seam carving (content-aware image resizing) using dynamic programming.

Assignment 2, Question 1:
  a) energy map        -> seamcarve.energy
  b) DP minimum seam   -> seamcarve.dp
  c) remove and repeat -> seamcarve.carve

Alternatives to DP:
  greedy seam          -> seamcarve.greedy   (Question 3)
  graph cut / max-flow -> seamcarve.graphcut
"""
