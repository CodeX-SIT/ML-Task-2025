# The Gradient Quest - ML Assignment

## 📋 Overview
Welcome to **The Gradient Quest**! This assignment is designed to help you understand Gradient Descent step-by-step through a hands-on coding adventure. You'll implement gradient descent from scratch and compare your results with scikit-learn's built-in implementation.

## 🎯 Learning Objectives
By completing this assignment, you will:
- Understand the fundamentals of gradient descent optimization
- Implement cost functions for linear regression
- Manually code gradient descent algorithm
- Visualize the convergence of gradient descent
- Experiment with different learning rates
- Compare custom implementations with scikit-learn
- Gain insights into feature scaling and learning rate selection

## 📊 Dataset
**Dataset:** Car Price Prediction  
**Source:** [Kaggle - Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)  
**File:** `CarPrice_Assignment.csv`

### Getting the Dataset
1. Download the dataset from the link above
2. Place `CarPrice_Assignment.csv` in the same directory as the notebook
3. Ensure the file is named exactly as specified

## 📁 Repository Structure
```
ML-Task-2025/
├── Assignment_Post_ML_Session.ipynb   # Main assignment notebook
├── CarPrice_Assignment.csv            # Dataset (download separately)
└── README.md                          # This file
```

## 🚀 Getting Started

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Setup Instructions
1. **Fork this repository** (automatically done via GitHub Classroom)
2. **Clone the forked repository**
4. **Download the dataset** from Kaggle (link above)
5. **Open the Jupyter notebook**
6. **Work through each stage sequentially**
7. **Push the changes to your fork to submit**

## 📝 Assignment Stages

### Stage 1: The Gatekeeper
**Objective:** Load the dataset and explore its structure  
**Task:** Load `CarPrice_Assignment.csv` and print its shape  
**Expected Output:** `Shape: (205, 26)`  
**Clue:** The number of columns

### Stage 2: The Feature Forge
**Objective:** Feature selection for price prediction  
**Task:** Select 3 numerical features most relevant to car price  
**Examples:** enginesize, horsepower, curbweight  
**Clue:** Number of features × 2

### Stage 3: The Cost Chamber
**Objective:** Implement the cost function  
**Task:** Create `compute_cost(X, y, theta)` function  
**Formula:** J(θ) = (1/2m) × Σ(h_θ(x_i) - y_i)²  
**Test with dummy data** to verify correctness  
**Clue:** Rounded cost value

### Stage 4: The Gradient Gate
**Objective:** Implement one gradient descent step  
**Task:** Create `gradient_step(X, y, theta, alpha)` function  
**Formula:** θ := θ - (α/m) × (X^T · (Xθ - y))  
**Clue:** Sum of theta elements (rounded to 2 decimals)

### Stage 5: The Descent Spiral
**Objective:** Full gradient descent implementation  
**Task:** Implement complete `gradient_descent()` function  
**Requirements:**
- Run for multiple iterations
- Record cost at each iteration
- Plot cost vs. iteration to visualize convergence  
**Clue:** Final cost value (rounded to 2 decimals)

### Stage 6: The Learning Rate Trial
**Objective:** Experiment with different learning rates  
**Task:** Test α = [0.001, 0.01, 0.1] for 500 iterations each  
**Requirements:**
- Plot all three learning rates on the same graph
- Identify which α converges fastest without diverging  
**Clue:** Best learning rate value

### Stage 7: The Twin Peaks
**Objective:** Compare with scikit-learn  
**Task:** Train a LinearRegression model using sklearn  
**Requirements:**
- Use the same features as your custom implementation
- Compare coefficients (theta values)
- Determine if they match numerically  
**Clue:** "MATCH" or "TRY AGAIN"

### Stage 8: Reflection
**Objective:** Demonstrate understanding  
**Task:** Answer these questions in markdown cells:
1. What happens if the learning rate is too high?
2. Why does feature scaling help gradient descent?
3. How do your results compare to sklearn's model?

**Final Step:** Print "THE QUEST IS COMPLETE"

## ✅ Submission Guidelines

### What to Submit
1. Completed Jupyter notebook (`Assignment_Post_ML_Session.ipynb`)
2. All cells should be executed with outputs visible
3. All clues should be printed correctly
4. Reflection questions answered in markdown cells

### Grading Criteria
- **Stage 1-2 (15%):** Data loading and feature selection
- **Stage 3-4 (25%):** Cost function and gradient step implementation
- **Stage 5 (20%):** Full gradient descent with visualization
- **Stage 6 (15%):** Learning rate experiments
- **Stage 7 (15%):** Comparison with scikit-learn
- **Stage 8 (10%):** Reflection and understanding

### Tips for Success
- ✅ Work through stages sequentially
- ✅ Test your functions with simple examples first
- ✅ Print intermediate results to debug
- ✅ Pay attention to array shapes and dimensions
- ✅ Use feature scaling for better convergence
- ✅ Document your thought process in markdown cells

## 🐛 Common Issues & Solutions

### Issue: Dataset not found
**Solution:** Ensure `CarPrice_Assignment.csv` is in the same directory as the notebook

### Issue: Cost not decreasing
**Solutions:**
- Check your cost function implementation
- Verify gradient calculation
- Try feature scaling (StandardScaler)
- Reduce learning rate

### Issue: Cost diverging (increasing)
**Solution:** Learning rate is too high - reduce α

### Issue: Slow convergence
**Solution:** Learning rate might be too small - increase α

### Issue: Coefficients don't match sklearn
**Solutions:**
- Ensure you're using the same features
- Apply feature scaling consistently
- Run enough iterations for convergence
- Check for implementation bugs

## 📚 Additional Resources

### Understanding Gradient Descent
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [3Blue1Brown - Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)

### Python Libraries Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## 🏆 Challenge Yourself
After completing the basic assignment:
- Try implementing batch, mini-batch, and stochastic gradient descent
- Experiment with momentum and adaptive learning rates
- Test with additional features
- Implement regularization (Ridge/Lasso)

---

**Good luck on your quest to master Gradient Descent!** 🚀

*Remember: The journey of understanding is more valuable than just getting the right answer.*