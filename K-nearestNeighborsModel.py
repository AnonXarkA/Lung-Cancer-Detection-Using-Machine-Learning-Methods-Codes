from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = cancer_patient.drop(["Level"], axis = 1)
y = cancer_patient["Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
y_pred = knn .predict(X_test)



print(accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
y_true = y_test
target_names = ["Low", "Medium", "High"]
print(classification_report(y_true, y_pred, target_names=target_names))



cls = ["Low", "Medium", "High"]
cm_model = confusion_matrix(y_test, y_pred)
d_model = calculations(cm_model,cls)

cm_df = pd.DataFrame(cm_model,
                     index = cls, 
                     columns = cls)
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix of knn')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


import matplotlib.pyplot as plt

# Define class labels
cls = ["Low", "Medium", "High"]

# Create confusion matrix
cm_model = confusion_matrix(y_test, y_pred)

# Create a pie chart for the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(cm_model.ravel(), autopct='%1.2f%%',  colors=['#1f77b4', '#ff7f0e', '#2ca02c'])

ax.set_title("Pie chart of Knn")
plt.show()


