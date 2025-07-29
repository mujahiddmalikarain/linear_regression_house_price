from sklearn.tree import DecisionTreeClassifier, plot_tree


import matplotlib.pyplot as plt


# Step 3: Prepare training data
# X = [age, weight]
X = [
    [18, 65],  # Pass
    [20, 85],  # Fail
    [17, 55],  # Pass
    [21, 90],  # Fail
    [19, 60],  # Pass
]
y = ['Pass', 'Fail', 'Pass', 'Fail', 'Pass']  # Target labels


# Step 4: Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(8, 5))
plot_tree(model, feature_names=["Age", "Weight"], class_names=["Fail", "Pass"], filled=True)
plt.title("Decision Tree for Pass/Fail Prediction")
plt.show()

# Step 6: Make a prediction for a new student
new_student = [[19, 60]]  # Age 20, Weight 70
prediction = model.predict(new_student)
print("Prediction for student:", prediction[0])