from sklearn.linear_model import Perceptron
classifier = Perceptron(random_state=0)   #linear classifier
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(accuracy_score(y_test, y_pred))



from sklearn.metrics import classification_report
y_true = y_test
target_names = ["Low", "Medium", "High"]
print(classification_report(y_true, y_pred, target_names=target_names))


cls = ["Low", "Medium", "High"]
cm_model = confusion_matrix(y_test, y_true)
d_model = calculations(cm_model,cls)

cm_df = pd.DataFrame(cm_model,
                     index = cls, 
                     columns = cls)
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix of linear classification')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


import matplotlib.pyplot as plt

# Define class labels
cls = ["Low", "Medium", "High"]

# Create confusion matrix
cm_model = confusion_matrix(y_test, y_true)

# Create a pie chart for the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(cm_model.ravel(), autopct='%1.2f%%',  colors=['#1f77b4', '#ff7f0e', '#2ca02c'])

ax.set_title("Pie chart of Linear Classification")
plt.show()





