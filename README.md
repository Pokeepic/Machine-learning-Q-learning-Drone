# 🛸 Q-Learning Delivery Drone Simulation

This project demonstrates the use of **Q-learning (Reinforcement Learning)** to train an autonomous delivery drone to navigate a grid-based environment and deliver food to a customer while avoiding obstacles and preferring safe zones.

The simulation includes an **animated visualization** showing how the drone’s behavior improves over time through learning.

---

## 📌 Problem Description

The delivery drone operates in a **4×4 grid environment** with the following elements:

- **S** – Distribution Hub (starting position)
- **G** – Customer Location (goal)
- **O** – Obstacles / No-fly zones (humans, pets, poles, construction sites)
- **R** – Safe zones (low-traffic areas, designated robot lanes)
- **Empty cells** – Normal movement areas

The objective of the drone is to **reach the customer as efficiently and safely as possible**, maximizing cumulative reward.

---

## 🧠 Reinforcement Learning Model

### States
Each grid cell represents a state.

### Actions
The drone can take four actions:
- Left (←)
- Down (↓)
- Right (→)
- Up (↑)

### Reward Function
| Event | Reward |
|------|--------|
| Move to empty cell | -1 |
| Enter safe zone (R) | +2 |
| Hit obstacle (O) | -5 (episode ends) |
| Reach goal (G) | +50 (episode ends) |

This reward design encourages:
- Shorter delivery paths
- Avoidance of obstacles
- Preference for safer routes

---

## 📐 Learning Algorithm

The agent is trained using **Q-learning**, updating values with:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big(r + \gamma \max Q(s',a') - Q(s,a)\big)
\]

Where:
- α (learning rate) = 0.8  
- γ (discount factor) = 0.93  
- ε-greedy strategy is used for exploration vs exploitation

---

## 🎥 Animation

The project includes a **Matplotlib animation** that visualizes:
- The drone’s movement per episode
- Early random exploration
- Gradual convergence to an optimal delivery path

This animation can also be exported as a **GIF** for reports or presentations.

---

## 📊 Output Example

- Learned Q-table showing action values per state
- Greedy policy represented with arrows
- Animated path improvement across episodes

---

## 🛠 Technologies Used

- Python
- NumPy
- Matplotlib
- Reinforcement Learning (Q-learning)

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Pokeepic/Machine-learning-Q-learning-Drone
pip install numpy matplotlib pillow
python reze_animated.py

