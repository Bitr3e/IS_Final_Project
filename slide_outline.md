# Slide Outline (Presentation: November 10, 2026)

Split speaking time evenly. Team Collaboration & Q&A is 15%, so every member must be able to answer questions on any part of the code.

1. **Title and team** (all members)
2. **Problem and goal**: one game, three ways of learning; what machine learning is (Module 9 overview)
3. **Paradigm comparison**: labeled data vs. no labels vs. rewards (one table)
4. **Supervised: the idea and why SVM** (max margin, support vectors, kernel trick); dataset of 4,520 labeled positions
5. **Supervised: code walkthrough** (`encode`, `optimal_moves`, `run_supervised`); feature engineering (90.5% to 95.2%); results chart
6. **Unsupervised: the idea and why K-Means + PCA**; 5,478 unlabeled positions, 8 features
7. **Unsupervised: code walkthrough**; choosing K (silhouette), cluster table, PCA plot
8. **Reinforcement: the idea and why Q-Learning**; agent, environment, action, reward, policy
9. **Reinforcement: code walkthrough** (the update rule, epsilon-greedy, self-play); learning curve and final strength
10. **Live demo**: menu options 1, 2, 3, then 4 (let a panelist play the bot)
11. **Limitations and next steps**: bigger games (Connect 4), deep RL, comparing the supervised and RL bots head to head
12. **Q&A**

## Likely Q&A questions to prepare

- Why scale or encode the board as 45 numbers instead of just 9?
- What is a support vector? What does the RBF kernel do?
- Why did adding "win here / block here" features help?
- Why is "optimal move" accuracy higher than "exact label" accuracy?
- Why does K-Means need standardized features? Why did silhouette pick K=7 when K=3 is close?
- What does PCA lose when going from 8 features to 2?
- What do alpha, gamma and epsilon do in the Q-learning update?
- Why does the bot need exploration if it only plays against itself?
- Why can't the bot lose, and why does it only draw a perfect player?
- Which paradigm would you pick for a new game, and why?
