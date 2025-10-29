# Berkeley CS 188 Pacman Project
<div align="center">
  <img width="80%" height="80%" alt="Image" src="https://github.com/user-attachments/assets/991cbace-9eea-43a3-8272-00efdfaafe8c" />
</div>

**Implementation of Berkely AI's Pacman Project with Multi-Agent Search (CS 188)**  
Original project instruction link is available at : https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj2/  
<br />
All important implementation details are presented in `multiAgents.py`.
## Q1 : Reflex Agent
A **"Reflex Agent"** is an agent that simply considers the current state of the environment and selects the next action, and does not consider changes in the future state of the environment according to the action.  
<br />
see `class ReflexAgent`'s `evaluationFunction()`.

## Q2 : Minimax Agent
The **"Minimax Search"** represents a competitive search process between agent (**Pacman**) that seek to **maximize** their utility in an **adversarial search problem**, and agents (**Ghosts**) that seek to **minimize** it.  
<br />
see `class MinimaxAgent`.

## Q3 : Alpha-Beta Pruning
**"Alpha-Beta Pruning"** algorithm is an algorithm that prunes edges **(transitions according to actions in tree search)** that do not ultimately affect the agent's selection, enabling more efficient tree search in minimax search.  
<br />
see `class AlphaBetaAgent`.

## Q4 : Expectimax Agent
**Expectimax Search** is a variant of minimax search, which **maximizes the expected utility** from all possible actions from the current state. This method is more effective when the **opposing agent is not optimal**, and in such cases, a slightly more **"risky"** form of search is performed.  
<br />
see `class ExpectimaxAgent`.

## Q5 : Evaluation Function
Write a **"better evaluation function"** for Pacman in `betterEvaluationFunction()`.  
We need to implement a function that evaluates the actions of the pacman agent, taking into account various situations that may occur in an adversarial search process.

---
## Tips for Autograder
You can find the test cases for each problem in the folders named as: `q1`, ..., `q5` in the `test_cases` folder. Additionally, for `q1` and `q5`, you can adjust the number of tests by modifying the `numGames` value in the `grade-agent.test` file.  
Also you can handle with the number of ghosts with `ghosts` attribute, or you can set the map configuration by modifying `layoutName`.

```
class: "EvalAgentTest"

agentName: "ReflexAgent"      // Target agent to be tested
layoutName: "openClassic"     // Map layout
maxTime: "120"
numGames: "10"                // # of games for evaluation of your agent implementation


nonTimeoutMinimum: "10"

scoreThresholds: "500 1000"

winsMinimum: "1"
winsThresholds: "5 10"


randomSeed: "0"              // Random seeds for world's configuration
ghosts: "[RandomGhost(1)]"   // Ghosts
```
