# Sudoku-Solver
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) <br />
Sudoku solver using Genetic Algorithm AI

## Approach
<ol>
  <li>Created a helper matrix which is a list of list containing all possible values a cell can contain.</li>
  <li>Initialized population of size 200 randomly using the helper.</li>
  <li>Calculated the fitness of each candidate and sorted them.</li>
  <li>Took first 120 (having greater fitness) as elites. Elites would definately be part of the next generation.</li>
  <li>For the remaining 80, randomly selected two candidates from the population and performed crossover among them. Mutation is
  done on each candidate.</li>
  <li>We continue doing this until we get a fitness value of 1. This indicates we have successfully found a solution to the sudoku problem.</li>
</ol>

## Output
![image](https://user-images.githubusercontent.com/30633549/57671798-eb526a00-7632-11e9-86a9-e95132826dd0.png)
