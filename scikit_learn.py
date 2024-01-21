# problem statement: predict a student's final exam score based on the number of the hours they study
'''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# step-2
ashmadata = {'Hour_study':[2,3,4,5,6,7,8,9,10], 'Exam_score': [50,55,65,70,75,80,85,92,96]}
# step-3
df = pd.DataFrame(ashmadata)
# step-4
X = df[['Hour_study']]
Y = df[['Exam_score']]
# step-5
x_train, x_test, y_train,y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
# step-6
model = LinearRegression()
# step-7
model.fit(x_train,y_train)
# user Input testing 
user_input = float(input("Enter the number of hours you study:"))
predicted_score = model.predict([[user_input]])
# printing output
print(f"Predicted Exam Score: {predicted_score[0][0]:.2f}")

# In an ecommerce company the management want to predict wheather a customer will purchase a high-value product based on their age, time spent on the website, and wheather they have added items to their cart. The goal is to optimize marketing strategies by targeting potential customers more effectively therby increasing sales and revenue
# logistic regression through
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression
X = np.array([[25,30,0], [30,40,1], [42,20,0], [35,45,1]])
Y = np.array([0,1,0,1])
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(f"Model Accuracy: {accuracy}")
user_age = float(input("Enter customer age:"))
user_time_spend = float(input("Enter time spent on wesite:"))
user_add_to_cart = int(input("Enter 1 if added to cart, else enter 0:"))
user_data = np.array([[user_age,user_time_spend,user_add_to_cart]])
prediction = model.predict(user_data)
if prediction [0]==1:
    print('The customer is likely to purchase')
else:
    print('The customer is unlikely to purchase')

# SVM(support Vector Machine): A telecommunication company wants to reduce customer churn by identifying customers at risk of leaving. They have historical data on customer behaviour and want to build a model to predict which customer are most likely to churn.
import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
data = {'Age':[30,25,37,28,40,55,32,36], 'Monthly_Charge': [50,60,70,80,100,200,55,75], 'churn':[0,1,0,1,1,0,1,0]}
df = pd.DataFrame(data)
X = df[['Age','Monthly_Charge']]
Y = df['churn']
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(x_train,y_train)
y_pred = svc_model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
report = classification_report(y_test, y_pred)
print(report)
user_age = float(input("Enter customer age:"))
user_charge = float(input("Enter customer monthly_charge:"))
user_data=np.array([[user_age,user_charge]])
prediction = svc_model.predict(user_data)
if prediction [0] == 1:
    print('The customer is likely to chur:')
else:
    print('The customer will go')

# KNN
# A retail company wants to predict customer purchasing based on their age, salary, and past purchase history. The company aims to use KNN algorith to classify customer into potential buying groups to personalize marketing strategies. This predictive model will help the company understand and target specific customer segments more effectively, thereby increasing sales and customer satisfaction.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
data = np.array([[25,35000,2],[30,45000, 1],[41,75000,3],[22,1500,2], [34,90000,3],[29,35000,5]])
labels = np.array([1,2,0,2,1,0]) # 0:low, 1: middle, 2: high
X_train,X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=42)
scalar= StandardScaler()
X_train= scalar.fit_transform(X_train)
X_test= scalar.transform(X_test)
knn = KNeighborsClassifier (n_neighbors=1)
knn.fit(X_train,y_train)
accuracy = knn.score(X_test,y_test)
print(f'Model accuracy: {accuracy}')
# user input
user_input= np.array([[32,15000,1]])
user_input_scalar = scalar.transform(user_input)
ashma = knn.predict(user_input_scalar)
print(ashma)

# Naive Bays
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
reviews = ['The product is excellent and works perfectly', 'The product is not good or very disappointing', 'Terrible product totally waste of money', 'I love this product , it is amazing']
sentiments = np.array([1, 0, 0, 1])
vectorizer =CountVectorizer()
x = vectorizer.fit_transform(reviews) # training_testing
classifier = MultinomialNB()
classifier.fit(x, sentiments)
def classify_new_reviews(review): # data input liyeko ho
    review_vectorized = vectorizer.transform([review])
    prediction = classifier.predict(review_vectorized)
    if prediction[0] == 1:
        return "Positive Sentiments"
    else:
        return "Negative Sentiments"

user_review = input("Enter your review: ")
result = classify_new_reviews(user_review)
print(f"The review '{user_review}' is classified '{result}'")

# Decision Tree Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x = np.array([[35, 60000, 3], [24, 35000,1],[31,45000,2],[27,54000,2],[20,1500,1],[26,36000,2],[30,70000,3],[41,9000,3],[31,47000,2]])
y = np.array([1,0,1,1,0,1,1,0,1])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
ypred = model.predict(x_test)
accuracy = accuracy_score(y_test,ypred)
print(accuracy)
age = float(input("Enter your age"))
income = float(input("Enter your income"))
education = int(input("Enter your education"))
user_input = np.array([[age,income,education]])
prediction = model.predict(user_input)
if prediction[0] ==1:
    print("person made purchase")
else:
    print("person not purchase")
import pandas as pd
# Random Forest
# Use a random forest classifier to predict wheather a person is likely to purchase a product based on certain features like age, gender, and estimated salary.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
data = {'Age':[34,23,42,22,31,21,53,56,60,36], 'Gender': ['M','F','M','M','F','M','F','M','F','M'],'Estimated_Salary':[23000,34000,45000,80000,36000,70000,27000,84000,35000,90000],'Purchase':[1,0,0,1,1,0,1,0,1,1]}
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
x = df.drop('Purchase', axis=1)
y = df['Purchase']
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier .fit(x_train,y_train)
y_pred = rf_classifier .predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy is: {accuracy}")
user_age = float(input("Enter your age: "))
user_gender = input("Enter your gender M/F: ")
user_salary = int(input("Enter your salary: "))
user_gender_encoded = label_encoder.transform([user_gender])[0]
user_data = [[user_age,user_gender_encoded,user_salary]]
prediction = rf_classifier.predict(user_data)
if prediction[0] ==1:
    print('The user likely to purchase')
else:
    print('The user unlikely to purchase')

# Gradiant Boosting
# create a predictive model using Gradient Boosting to forecast housing prices based on various features such as sq.ffotage, number of bed room, number of bathrooms, and locations.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
data = {'sq_footage':[1922,1844,1250,1560,1789,1234, 3214,8965],
        'no_of_bedroom':[5,2,3,4,2,1,4,2],
        'no_of_bathroom':[1.5,2,1,2.5,1,1,1,2],
        'location':['suburb','city','rural','city','rural','suburb','city','rural'],
        'price':[300000,400000,134000,457000,457000,234000,124500,987000]}
df = pd.DataFrame(data)
df = pd.get_dummies(df,columns=['location'])
x = df.drop(['price'], axis=1)
y = df['price']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
model = GradientBoostingRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = mean_squared_error(y_test,y_pred)
print(f'accuracy is: {accuracy}')
house_size = float(input("enter the house square footage:"))
house_bed_room = int(input("enter no of bedroom: "))
house_bathroom = float(input("enter the no of bathroom: "))
house_location = input("enter the location(suburb,rural,city: ")
input_location = [0, 0, 0]
if house_location == 'suburb':
    input_location[0] = 1
elif house_location == 'rural':
    input_location[1] = 1
elif house_location == 'city':
    input_location[2] = 1
user_input = pd.DataFrame({'sq_footage': [house_size],
                            'no_of_bedroom': [house_bed_room],
                            'no_of_bathroom': [house_bathroom],
                            'location_suburb': [input_location[0]],
                            'location_rural': [input_location[1]],
                            'location_city': [input_location[2]]})
# Reorder columns to match the order during training
user_input = user_input.reindex(columns=x_train.columns, fill_value=0)
predicted_price = model.predict(user_input)
print(f"predicted price for the house is: {predicted_price[0]}")

# Neural Network
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
hour_studied = [2.5,4.5,5.5,6.5,7.5,1.8,3.3,5.4,1.2,1.7]
previous_exam_score = [55,68,70,80,40,60,64,45,50,51]
exam_outcome = ['Pass', 'Fail','Pass','Pass','Fail','Fail','Pass','Fail','Pass','Fail']
label_encoder = LabelEncoder()
encoded_exam_outcome = label_encoder.fit_transform(exam_outcome)
x = np.column_stack([hour_studied,previous_exam_score])
y = encoded_exam_outcome
clf =MLPClassifier(hidden_layer_sizes =(4,),activation='logistic', max_iter=1000, random_state=42)
clf.fit(x,y)
new_student_data = np.array([[2.0,78]]) # hours studied and previous exam score
predicted_outcome = clf.predict(new_student_data)
predicted_outcome_decode = label_encoder.inverse_transform(predicted_outcome)
print(f"Predicted exam outcome for the new student is: {predicted_outcome_decode[0]}")

# K-means Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
np.random.seed(42)
data = {'Annual_Income': np.random.randint(30000,100000,100),'Spending_Score': np.random.randint(1,100,100)}
df = pd.DataFrame(data)
x = df.values
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(x)
plt.scatter(df['Annual_Income'],df['Spending_Score'], c=df['Cluster'], cmap='rainbow')
plt.title('KMeans Clustering: Customer Data - Annual Income Vs Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
user_input = {'Annual_Income': [50000],'Spending_Score':[55]}
user_df = pd.DataFrame(user_input)
user_cluster = kmeans.predict(user_df)
print(f"The prediction is: {user_cluster[0]}")

# Hierarchical Agglomerative Clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

data = pd.DataFrame({'Age': np.random.randint(18, 65, 100),
                     'income': np.random.randint(30000, 100000, 100),
                     'spending_score': np.random.randint(1, 100, 100)})

scalar = StandardScaler()  # for standardization
scaled_data = scalar.fit_transform(data[['income', 'spending_score']])

clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
data['Cluster'] = clustering.fit_predict(scaled_data)

plt.scatter(data['income'], data['spending_score'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Hierarchical Agglomerative')
new_customer = pd.DataFrame({'Age': [30, 31, 32], 'income': [50000, 51000, 52000], 'spending_score': [70, 71, 72]})
scaled_new_customer = scalar.transform(new_customer[['income', 'spending_score']])
predicted_cluster = clustering.fit_predict(scaled_new_customer)
print(f"The new customer predicted clusters are: {predicted_cluster}")
plt.show()

# DBSCAN(Density based spatial clustering application with noise) Clustering
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
np.random.seed(42)
data = np.random.rand(100,2)
scalar = StandardScaler()
data_scale = scalar.fit_transform(data)
dbscan = DBSCAN(eps=0.3, min_samples=5)
cluster = dbscan.fit_predict(data_scale)
plt.scatter(data[:,0],data[:,1],c=cluster, cmap='viridis', marker='o', s=50)
plt.title('DBSCAN Clustering', c='pink')
plt.xlabel('Feature-1', c='red')
plt.ylabel('Feature-2',c='green')
plt.show()
result_df = pd.DataFrame({'Feature-1':data[:,0],'Feature-2':data[:,1], 'Cluster':cluster})
print('Number of cluster:',len(np.unique(cluster)))
print("size of each cluster:")
print(result_df['Cluster'].value_counts())
# model testing
user_input = np.array([[0.6,0.7]])
user_input_scaled= scalar.transform(user_input)
user_cluster = dbscan.fit_predict(user_input_scaled)
print(f'User input belongs to cluster:', user_cluster[0])'''

# Gaussian Mixture Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
segment1 = np.random.normal(loc=30,scale=5,size=100)
segment2 = np.random.normal(loc=60,scale=9,size=150)
segment3 = np.random.normal(loc=90,scale=7,size=120)
data = np.concatenate([segment1,segment2,segment3]).reshape(-1,1)
scalar = StandardScaler()
data_scaled = scalar.fit_transform(data)
n_component = 3
gmm = GaussianMixture(n_components=n_component, random_state=42)
gmm.fit(data_scaled)
cluster_label = gmm.predict(data_scaled)
plt.scatter(data, np.zeros_like(data), c=cluster_label, cmap='viridis')
plt.title('Cluster Segmentation')
plt.xlabel('Purchase Amount')
plt.show()

user_input = float(input("Enter a purchase amount to predict the customer segment:"))
user_input_scaled = scalar.transform(np.array([[user_input]]))
user_predicted_label = gmm.predict(user_input_scaled.reshape(-1,1))[0]
print(f"The predicted customer segment for a purchadse amount of {user_input} is:,{user_predicted_label + 1}")























































