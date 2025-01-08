import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
import plotly.express as py
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# this library is for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# this library is data processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# this library is for modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv

# # this library is for model evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, add, GlobalAveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model
from skimage import io
import matplotlib.pyplot as plt, numpy as np

from tensorflow.keras.utils import image_dataset_from_directory

from skimage import io, transform, color, exposure, img_as_float
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd



def calculations(cm_test,classes):     #cm = confusion matrix , Each row in "cm_test" corresponds to the predicted labels for a particular class, and each column corresponds to the true labels for a particular class.
# #"classes" is expected to be a list of class labels for which the performance metrics will be calculated.



  cal_d_test = {}
  for i in range(len(classes)):
    cal_d_test[classes[i]] = []      #Initializes an empty list for the class in "cal_d_test" with the class label as the key.
    TP = cm_test[i][i]                 #True Positives (TP)
    FN = 0                            #False Negatives (FN)
    FP = 0                            #False Positives (FP)
    TN = 0                            #True Negatives (TN)

#Calculates  False Negatives (FN), False Positives (FP), and True Negatives (TN) from the confusion matrix.
    for j in cm_test[i]:
      FN+=j
    FN = FN - TP
  
    for j in range(len(classes)):          ##Calculates  False Positives (FP)
      FP += cm_test[j][i]
    FP = FP - TP

    for x in range(len(classes)):            ##Calculates True Negatives (TN)
      for y in range(len(classes)):
        TN += cm_test[x][y]
    TN = TN - TP - FN - FP


    sensitivity = TP/(TP+FN)                  #Calculates sensitivity (True Positive Rate or Recall), specificity
    specificity = TN/(TN+FP)
    ppv = TP/(TP+FP)                           # positive predictive value (PPV), negative predictive value (NPV)
    npv = TN/(TN+FN)
    f1 = 2*(ppv*sensitivity)/(ppv+sensitivity)     #f1 score
    cal_d_test[classes[i]].append(sensitivity)
    cal_d_test[classes[i]].append(specificity)       #Appends the calculated performance metrics (sensitivity, specificity, PPV, NPV, and F1 score) to the list with the current class label in "cal_d_test" dictionary.
    cal_d_test[classes[i]].append(ppv)
    cal_d_test[classes[i]].append(npv)
    cal_d_test[classes[i]].append(f1)

  return cal_d_test



#reading csv file
cancer_patient = pd.read_csv("/content/drive/MyDrive/STudy Brac/422/Project/cancer patient data sets.csv")         
cancer_patient.head()


#import statements for plotting using Matplotlib and Seaborn libraries

#for Matplotlib plots to be displayed directly in the notebook output cells.
%matplotlib inline          

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt       #imports the Matplotlib library's pyplot module
import seaborn as sns                 #imports the Seaborn library, it is a statistical data visualization library
plt.style.use("seaborn-whitegrid")  


cancer_patient.info();
cancer_patient.describe()

#drops patient id column 

cancer_patient.drop(["Patient Id"], axis = 1, inplace= True)


fig = plt.figure(figsize = (15,10))                 
sns.heatmap(cancer_patient.corr(),cmap="crest",annot=True);          #calculates the correlation matrix for the numerical columns using .corr()
 #Higher positive correlations represents brighter colors, and higher negative correlations represents darker colors. 


plot = sns.countplot(data = cancer_patient, x='Gender', hue='Age', palette=['black','blue'])  #creates a countplot of Gender vs age


fig, ax = plt.subplots()    # creates a histogram plot of the "Age" column
hist = ax.hist(x = cancer_patient["Age"]);


fig, ax = plt.subplots()       #creates a histogram plot of the "Gender" column
hist = ax.hist(x = cancer_patient["Gender"]);

plot = sns.countplot(data = cancer_patient, x='Level', hue='Gender', palette=['black','pink'])   #counting the plot from gender based on levels of the cancer 

cancer_patient.columns


cancer_below50 = cancer_patient[ cancer_patient["Age"] < 50]    #calculating the info of patients below 50years old
cancer_below50.head(10)


cancer_over50 = cancer_patient[cancer_patient["Age"] > 50]      #calculating the info of patients over 50years old
cancer_over50.head()






# Making Subplots  for age over 50 years

#The "Genetic Risk" and "Smoking" columns used for the y-axis of the second and third subplots, 
#The "Gender" and "Alcohol use" columns  are used for the x-axis and y-axis of the fourth subplot,

fig, ((ax1, ax2 ,ax5), (ax3, ax4, ax6)) = plt.subplots(nrows = 2, ncols= 3, figsize=(12, 15))

# Adding Data to the plot


# For Plot ax1
scatter = ax1.scatter(x = cancer_over50["Age"], y = cancer_over50["Alcohol use"], cmap = "winter")
ax1.set(title = "Age with respect to Alcohol Use", 
        xlabel = "Age", 
        ylabel = "Alcohol Use")
ax1.axhline(cancer_over50["Alcohol use"].mean(),
           linestyle = "--");
ax1.set_xlim([50, 80])
ax1.set_ylim([0, 8.5])

# For Plot ax2
scatter = ax2.scatter(x = cancer_over50["Age"], y = cancer_over50["Smoking"])
ax2.set(title = "Age with respect to Smoking", xlabel = "Age", ylabel = "Smoking")
ax2.axhline(cancer_over50["Smoking"].mean(),
           linestyle = "--");
ax2.set_xlim([50, 80])
ax2.set_ylim([0, 8.5])

# For Plot ax3
scatter = ax3.scatter(x = cancer_over50["Age"], y = cancer_over50["Genetic Risk"])
ax3.set(title = "Age with respect to Genetic Risk", xlabel = "Age", ylabel = "Genetic Risk")
ax3.axhline(cancer_over50["Genetic Risk"].mean(),
           linestyle = "--");
ax3.set_xlim([50, 80])
ax3.set_ylim([0, 8.5])

# For Plot ax4
scatter = ax4.scatter(x = cancer_over50["Gender"], y = cancer_over50["Alcohol use"])
ax4.set(title = "Gender with respect to Alcohol Use", xlabel = "Gender", ylabel = "Alcohol Use")
ax4.axhline(cancer_over50["Alcohol use"].mean(),
           linestyle = "--");

# For Plot ax5
scatter = ax5.scatter(x = cancer_over50["Gender"], y = cancer_over50["Dust Allergy"])
ax5.set(title = "Gender with respect to Dust Allergy", xlabel = "Gender", ylabel = "Dust Allergy")
ax5.axhline(cancer_over50["Dust Allergy"].mean(),
           linestyle = "--");
         

# For Plot ax6
scatter = ax6.scatter(x = cancer_over50["Gender"], y = cancer_over50["chronic Lung Disease"])
ax6.set(title = "Gender with respect to chronic Lung Disease", xlabel = "Gender", ylabel = "chronic Lung Disease")
ax6.axhline(cancer_over50["chronic Lung Disease"].mean(),
           linestyle = "--");
           





# Making Subplots  for age below 50 years

#The "Genetic Risk" and "Smoking" columns used for the y-axis of the second and third subplots, 
#The "Gender" and "Alcohol use" columns  are used for the x-axis and y-axis of the fourth subplot,

fig, ((ax1, ax2 ,ax5), (ax3, ax4, ax6)) = plt.subplots(nrows = 2, ncols= 3, figsize=(12, 15))

# Adding Data to the plot


# For Plot ax1
scatter = ax1.scatter(x = cancer_below50["Age"], y = cancer_below50["Alcohol use"], cmap = "winter")
ax1.set(title = "Age with respect to Alcohol Use", 
        xlabel = "Age", 
        ylabel = "Alcohol Use")
ax1.axhline(cancer_below50["Alcohol use"].mean(),
           linestyle = "--");
ax1.set_xlim([50, 80])
ax1.set_ylim([0, 8.5])

# For Plot ax2
scatter = ax2.scatter(x = cancer_below50["Age"], y = cancer_below50["Smoking"])
ax2.set(title = "Age with respect to  Smoking", xlabel = "Age", ylabel = "Smoking")
ax2.axhline(cancer_below50["Smoking"].mean(),
           linestyle = "--");
ax2.set_xlim([50, 80])
ax2.set_ylim([0, 7.5])

# For Plot ax3
scatter = ax3.scatter(x = cancer_below50["Age"], y = cancer_below50["Genetic Risk"])
ax3.set(title = "Age with respect to Genetic Risk", xlabel = "Age", ylabel = "Genetic Risk")
ax3.axhline(cancer_below50["Genetic Risk"].mean(),
           linestyle = "--");
ax3.set_xlim([50, 80])
ax3.set_ylim([0, 8.5])

# For Plot ax4
scatter = ax4.scatter(x = cancer_below50["Gender"], y = cancer_below50["Alcohol use"])
ax4.set(title = "Gender with respect to Alcohol Use", xlabel = "Gender", ylabel = "Alcohol Use")
ax4.axhline(cancer_below50["Alcohol use"].mean(),
           linestyle = "--");

# For Plot ax5
scatter = ax5.scatter(x = cancer_below50["Gender"], y = cancer_below50["Dust Allergy"])
ax5.set(title = "Gender with respect to Dust Allergy", xlabel = "Gender", ylabel = "Dust Allergy")
ax5.axhline(cancer_below50["Dust Allergy"].mean(),
           linestyle = "--");

# For Plot ax6
scatter = ax6.scatter(x = cancer_below50["Gender"], y = cancer_below50["chronic Lung Disease"])
ax6.set(title = "Gender with respect to chronic Lung Disease", xlabel = "Gender", ylabel = "chronic Lung Disease")
ax6.axhline(cancer_below50["chronic Lung Disease"].mean(),
           linestyle = "--");


cancer_patient.info()


sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize = (15,10))            #This plot is a histogram with stacked bars for different age groups (hue) based on the "Level" column values.
sns.despine(f)
sns.histplot(
    cancer_patient, x= "Level", hue = "Age",
    multiple="stack",palette="dark:c_r",
    edgecolor = "0.5"
) 


#Feature Encoding
#Labelencodng
#Converting categorical data to numeric data

cancer_patient["Level"].replace(["Low", "Medium", "High"], ["0", "1", "2"], inplace=True)
cancer_patient["Level"] = cancer_patient["Level"].astype(int)
cancer_patient.head()


fig, ax = plt.subplots(figsize = (10, 6));                #for scatter plot

scatter = ax.scatter(x = cancer_patient["Age"],            #"Age" column  x-axis values, , 
                     y = cancer_patient["Genetic Risk"],   #the "Genetic Risk" column as the y-axis 
                     c = cancer_patient["Level"],          #"Level" column as the color values for the scatter points.
                     cmap = "spring")

ax.set(xlabel = "Age", 
       ylabel = "Genetic Risk");

ax.legend(*scatter.legend_elements(), title = "Level");

ax.axhline(cancer_patient["Level"].mean(),
           linestyle = "--");


cancer_patient.plot.kde(figsize = (20,5));   #for plotting kde(kernel density estimate) it process of estimating an unknown probability density function


cancer_patient.hist(color="black")   #plotting histogram of each column


# Creating NumPy array from the list
np.array([cancer_patient["Gender"][:10]])

gender_list = cancer_patient["Gender"].head(10).tolist()


gender_array = np.array(gender_list)

# Print the NumPy array
print(gender_array)


#calculating the number of males and females
male = 0
female = 0
for x in cancer_patient["Gender"]:
    if x == 1:
        male += 1
    elif x == 2:
        female += 1
f"Number of Male: {male}, Number of females: {female}"



cancer_patient_male = cancer_patient[cancer_patient["Gender"] == 1]
cancer_patient_male.head()


cancer_patient_female = cancer_patient[cancer_patient["Gender"] == 2]
cancer_patient_female.head()

#making histogram of male and female cancer patients
cancer_patient_male.hist(),cancer_patient_female.hist()



fig, ax = plt.subplots(figsize = (10, 6))                   #it used to create a scatter plot with "Age" column as the x-axis 
scatter = ax.scatter(x = cancer_patient["Age"], 
                     y = cancer_patient["Alcohol use"],     # "Alcohol use" column as the y-axis,
                     c = cancer_patient["Level"],           #"Level" column as the color.
                     cmap = "winter")

ax.set(xlabel = "Age", 
       ylabel = "Alcohol use");

ax.legend(*scatter.legend_elements(), title = "Level");

ax.axhline(cancer_patient["Level"].mean(),
           linestyle = "--");




fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block
plot=sns.scatterplot(data=cancer_patient, 
                     x='Genetic Risk',
                     y='Smoking', 
                     hue='Level', 
                     palette=['darkblue','darkred','darkgreen'], 
                     s=50, 
                     marker='o')#Count plot    





fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block
plot=sns.scatterplot(data=cancer_patient, 
                     x='Alcohol use',
                     y='Fatigue', 
                     hue='Level', 
                     palette=['darkblue','darkred','darkgreen'], 
                     s=50, 
                     marker='o')#Count plot



fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block
plot=sns.scatterplot(data=cancer_patient, 
                     x='Air Pollution',
                     y='Dust Allergy', 
                     hue='Level', 
                     palette=['black','blue','green'], 
                     s=50, 
                     marker='o')#Count plot



fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block
plot=sns.scatterplot(data=cancer_patient, 
                     x='chronic Lung Disease',
                     y='Weight Loss', 
                     hue='Level', 
                     palette=['black','blue','green'], 
                     s=50, 
                     marker='o')#Count plot



fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block
plot=sns.scatterplot(data=cancer_patient, 
                     x='Frequent Cold',
                     y='Dry Cough', 
                     hue='Level', 
                     palette=['black','blue','green'], 
                     s=50, 
                     marker='o')#Count plot







#This pie chart displays the distribution of values in each of symtopms as a percentage of the whole dataset.

# Extract two columns from the dataset
# column1 = cancer_patient["Age"]
column2 = cancer_patient["Air Pollution"]
column3 = cancer_patient["Dust Allergy"]
column4 = cancer_patient["OccuPational Hazards"]
column5 = cancer_patient["Genetic Risk"]
column6 = cancer_patient["chronic Lung Disease"]
column7 = cancer_patient["Fatigue"]
column8 = cancer_patient["Weight Loss"]
column9 = cancer_patient["Shortness of Breath"]
column10 = cancer_patient["Wheezing"]
column11= cancer_patient["Swallowing Difficulty"]
column12 = cancer_patient["Clubbing of Finger Nails"]
column13 = cancer_patient["Frequent Cold"]
column14 = cancer_patient["Dry Cough"]
column15 = cancer_patient["Snoring"]


# Create a pie chart
fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size
labels = ["Air Pollution",  "Dust Allergy", "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
          "Fatigue", "Weight Loss","Shortness of Breath", "Wheezing", "Swallowing Difficulty", "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"]  # Labels for the pie chart
values = [ column2.sum(), column3.sum(), column4.sum(), column5.sum(), column6.sum(), 
          column7.sum(), column8.sum(), column9.sum(), column10.sum(), column11.sum(), column12.sum(), column13.sum(), column14.sum(), column15.sum()]  # Values for the pie chart
colors = ["#1f77b4", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#ff7f0e"]  # Colors for the pie chart
ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)

# Set title for the pie chart
ax.set_title("Pie Chart: Values of class distribution")

# Show the plot
plt.show()













