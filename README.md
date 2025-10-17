# CS 370 - Current/Emerging Trends in Computer Science

## Pirate Intelligent Agent - Treasure Hunt Game

### Project Overview

This repository contains my implementation of a deep Q-learning algorithm to train an intelligent agent (pirate) to navigate a maze and find treasure. The project demonstrates the application of reinforcement learning and neural networks to solve a pathfinding problem autonomously.

---

## Project Reflection

### Work Completed

**Code Provided:**
The project began with substantial foundational code that formed the environment and support structures:
- **TreasureMaze.py**: A complete Python class representing the game environment, including the maze structure (8×8 matrix), state management, reward system, and action validation methods
- **GameExperience.py**: A fully implemented experience replay class that stores episodes and manages the agent's memory, including methods for sampling batches and generating training data
- **Neural Network Architecture**: The `build_model()` function that constructs a neural network with two hidden layers using PReLU activation functions
- **Helper Functions**: Visualization tools (`show()`), game simulation (`play_game()`), and completion verification (`completion_check()`)
- **Training Infrastructure**: The `train_step()` function using TensorFlow's gradient tape for backpropagation

**Code Created by Me:**
My primary contribution was implementing the core deep Q-learning training algorithm within the `qtrain()` function. This involved:

1. **Training Loop Architecture**: Designed the main epoch loop that runs until the agent achieves mastery or reaches the maximum iteration limit

2. **Epsilon-Greedy Strategy**: Implemented the exploration vs. exploitation decision mechanism using epsilon-greedy action selection, including:
   - Random action selection for exploration (when `np.random.rand() < epsilon`)
   - Model-based action selection for exploitation (using `experience.predict()`)
   - Dynamic epsilon decay schedule that adapts based on performance

3. **Game Episode Management**: Created the episode loop that:
   - Randomly selects starting positions for each training game
   - Tracks state transitions throughout gameplay
   - Records wins and losses in the history
   - Manages game-over conditions

4. **Experience Replay Integration**: Implemented the storage and retrieval of episodes:
   - Constructed episode tuples containing `[previous_state, action, reward, new_state, game_status]`
   - Called `experience.remember()` to store experiences
   - Retrieved training batches using `experience.get_data()`

5. **Neural Network Training**: Integrated the training process:
   - Converted training data to TensorFlow tensors
   - Called the `train_step()` function to perform backpropagation
   - Tracked loss values across training

6. **Target Network Updates**: Implemented periodic synchronization of the target network with the main network every 50 epochs for training stability

7. **Performance Monitoring**: Created comprehensive tracking and reporting:
   - Calculated moving average win rates over a sliding window
   - Formatted and displayed epoch statistics (loss, episodes, win count, win rate, time)
   - Implemented convergence criteria checking

8. **Adaptive Learning Parameters**: Developed the logic to adjust epsilon based on performance:
   - Aggressive decay during poor performance phases
   - Reduction to minimum exploration when win rate exceeds 90%

**Results Achieved:**
The implementation successfully trained the agent to achieve a 100% win rate at epoch 361 after approximately 8.9 minutes of training. The agent demonstrated the ability to navigate from any starting position to the treasure, as verified by the completion check that tested all 53 possible free cell starting positions.

---

## Computer Science and This Project

### What Computer Scientists Do and Why It Matters

Computer scientists solve complex problems by designing algorithms, building systems, and creating intelligent solutions that impact every aspect of modern life. In this project, I engaged in several core computer science activities:

**Algorithm Design**: I implemented a reinforcement learning algorithm that enables autonomous decision-making. This type of work has real-world applications in robotics (autonomous navigation), game AI (adaptive opponents), resource optimization (logistics and routing), and autonomous vehicles (pathfinding and obstacle avoidance).

**System Integration**: I connected multiple components—neural networks, experience replay, environment simulation, and training loops—into a cohesive learning system. This integration mirrors the work computer scientists do when building complex software systems where different modules must communicate effectively.

**Performance Optimization**: Through careful parameter tuning (epsilon decay rates, learning rates, batch sizes, target network update frequency), I optimized the training process. This reflects the computer scientist's role in making systems efficient, whether optimizing algorithms for speed, memory usage, or learning efficiency.

**Problem Abstraction**: The maze navigation problem is an abstraction of many real-world challenges. By solving it with deep Q-learning, I demonstrated how computer scientists generalize solutions to apply across domains—the same principles used here apply to warehouse robots, video game NPCs, or traffic routing systems.

This work matters because artificial intelligence and machine learning are transforming industries. Autonomous systems need to make decisions in complex, dynamic environments. By developing and understanding these algorithms, computer scientists create technologies that improve efficiency, safety, and capability across healthcare, transportation, finance, manufacturing, and entertainment.

### Approaching Problems as a Computer Scientist

Through this project, I learned to approach problems with a structured, analytical methodology:

**1. Problem Decomposition**
I broke down the complex challenge of "teach an agent to navigate a maze" into manageable sub-problems:
- How does the agent perceive its environment? (State representation)
- What actions can it take? (Action space definition)
- How do we measure success? (Reward structure)
- How does it learn from experience? (Q-learning algorithm)
- How do we prevent overfitting? (Experience replay and target networks)

**2. Research and Knowledge Application**
I studied reinforcement learning concepts—Markov Decision Processes, Q-learning, epsilon-greedy strategies, experience replay, and target networks—and applied them systematically. Computer scientists must continuously learn new techniques and understand when and how to apply them.

**3. Iterative Development and Testing**
I didn't build the complete solution at once. The development process involved:
- Implementing the basic training loop
- Testing and debugging errors (like incorrect parameter names)
- Monitoring performance metrics
- Adjusting hyperparameters based on results
- Validating with the completion check

This iterative approach—implement, test, analyze, refine—is fundamental to computer science.

**4. Algorithmic Thinking**
I reasoned about time complexity (how many epochs needed?), space complexity (how much memory for experience replay?), and convergence guarantees (will epsilon decay lead to optimal policies?). This mathematical and logical reasoning is central to computer science.

**5. Debugging and Problem-Solving**
When errors occurred (like the `TypeError` with `get_data()`), I systematically:
- Read error messages carefully
- Examined the source code of supporting classes
- Identified the root cause (parameter naming)
- Implemented and tested the fix

**6. Performance Analysis**
I analyzed training metrics to understand:
- Why the agent initially failed (insufficient exploration)
- When learning accelerated (epochs 140-165)
- How to verify robust learning (completion check from all positions)

This data-driven approach to validating solutions is essential in computer science.

**7. Documentation and Communication**
I documented my code with clear comments and created comprehensive explanations of my approach. Computer scientists must communicate technical concepts to both technical and non-technical audiences.

### Ethical Responsibilities

Working on AI and intelligent systems brings significant ethical responsibilities to both end users and organizations:

**Responsibility to End Users:**

**Safety and Reliability**: In this project, the agent must consistently reach the goal without getting stuck or causing system failures. In real-world applications—autonomous vehicles, medical diagnosis systems, financial trading algorithms—failures can have serious consequences. I have a responsibility to thoroughly test systems, implement safeguards, and be transparent about limitations.

**Fairness and Bias**: While my maze agent doesn't face bias concerns, many AI systems do. If I were developing a pathfinding system for a delivery robot, I'd need to ensure it doesn't avoid certain neighborhoods or discriminate in route selection. Computer scientists must actively work to identify and mitigate bias in training data and algorithms.

**Privacy**: AI systems often process personal data. Even in this project, if the maze represented a real environment with user movement patterns, I'd need to handle that data responsibly, implement privacy protections, and obtain proper consent.

**Transparency**: Users deserve to understand how AI systems make decisions. While deep learning models can be "black boxes," I have a responsibility to provide explanations when possible, especially in high-stakes applications like healthcare or criminal justice.

**Accessibility**: Technology should be accessible to all users, including those with disabilities. In developing AI systems, I must consider diverse user needs and ensure equitable access.

**Responsibility to Organizations:**

**Integrity and Honesty**: I must accurately represent the capabilities and limitations of my systems. If my pirate agent only works in 8×8 mazes with specific obstacle patterns, I shouldn't claim it generalizes to all navigation problems. Overpromising AI capabilities can lead to misallocated resources and failed projects.

**Security**: AI systems can be vulnerable to adversarial attacks or exploitation. I'm responsible for implementing security measures, testing for vulnerabilities, and following best practices to protect organizational assets and user data.

**Resource Efficiency**: Training AI models consumes computational resources and energy. I have a responsibility to optimize algorithms, avoid unnecessary computation, and consider the environmental impact of large-scale training.

**Intellectual Property**: I must respect copyright, use open-source licenses appropriately, and protect proprietary algorithms. In this project, I built upon provided code while creating original implementations—understanding these boundaries is crucial.

**Continuous Monitoring**: AI systems can degrade over time or behave unexpectedly in new situations. I'm responsible for implementing monitoring systems, planning for maintenance, and updating models as needed.

**Ethical Use Cases**: I must consider whether an AI application should be built at all. Some applications of AI—like autonomous weapons, invasive surveillance, or manipulative persuasion systems—raise serious ethical concerns. Computer scientists have a responsibility to question whether a project aligns with ethical principles and to decline work on harmful applications.

**Broader Societal Impact**: The work we do as computer scientists shapes society. Automation through AI affects employment, algorithmic decision-making affects justice and opportunity, and intelligent systems affect privacy and autonomy. I have a responsibility to consider these broader impacts, engage in policy discussions, and advocate for responsible AI development.

In this project, these ethical considerations were limited in scope, but they provided valuable practice in thinking through the implications of AI systems. As I continue in computer science, I'm committed to prioritizing ethical considerations alongside technical excellence, ensuring that the systems I build benefit users and society while minimizing potential harms.

---

## Technical Details

**Technologies Used:**
- Python 3.11
- TensorFlow 2.19
- Keras 3.11
- NumPy 1.26.4
- Matplotlib for visualization

**Algorithm:**
Deep Q-Learning with Experience Replay and Target Network

**Training Results:**
- Epochs to convergence: 361
- Training time: 8.90 minutes
- Final win rate: 100%
- Successfully navigates from all 53 possible starting positions

**Key Hyperparameters:**
- Initial epsilon: 1.0
- Epsilon decay: 0.995
- Minimum epsilon: 0.05
- Discount factor (gamma): 0.95
- Experience replay buffer: 512 episodes
- Batch size: 32
- Target network update frequency: 50 epochs

---

## Repository Structure

```
CS370-Portfolio/
├── YourName_ProjectTwo.ipynb    # Jupyter notebook with complete implementation
├── TreasureMaze.py               # Environment class (provided)
|── Requirements.txt
├── GameExperience.py             # Experience replay class (provided)
└── README.md                     # This file
```

---

## How to Run

1. Ensure Python 3.11+ is installed
2. Install required packages:
   ```bash
   pip install numpy==1.26.4 tensorflow==2.19.0 keras matplotlib
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Javaney_Thomas_ProjectTwo.ipynb
   ```
4. Run all cells to train the agent and visualize results

---

## Future Enhancements

Potential improvements to explore:
- Implement Double DQN to reduce overestimation bias
- Add prioritized experience replay for more efficient learning
- Test on larger mazes or dynamic environments
- Implement dueling network architecture
- Add curriculum learning for faster convergence
- Visualize Q-value heatmaps to understand learned policies

---

## Acknowledgments

This project was completed as part of CS 370: Current/Emerging Trends in Computer Science at Southern New Hampshire University. The foundational code framework was provided by the course, and the deep Q-learning algorithm implementation was developed independently.

---

## Contact

Javaney Thomas
javaneythomas4@gmail.com

---

*Last Updated: [Current Date]*
