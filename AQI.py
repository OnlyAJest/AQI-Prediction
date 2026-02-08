import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("city_day.csv")

data.dropna(axis = 0, inplace = True)
features_to_drop = ['Benzene', 'Toluene', 'Xylene']
data = data.drop(columns=features_to_drop)
#Features selected based on domain knowledge

data['Date'] = pd.to_datetime(data["Date"])


#cutoff = (data["Date"].max() - pd.Timedelta(days = 300))

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)


#--- Line Plot over Time
for city, data_city in data.head().groupby("City"):
	#data_city = data_city[data_city["Date"] > cutoff]
	ax[0].plot(data_city["Date"], data_city["AQI"], label = city)
ax[0].set_xlabel("City")
ax[0].set_ylabel("AQI")
ax[0].legend()
ax[0].set_title("AQI over time by city")

#--- Line Plot over City
labels = []
aqi_data= []

for city, data_city in data.groupby("City"):
	#data_city = data_city[data_city["Date"] > cutoff]
	aqi_data.append(data_city["AQI"])
	labels.append(city)

ax[1].boxplot(aqi_data)
ax[1].set_xlabel("City")
ax[1].set_ylabel("AQI")
ax[1].set_xticklabels(labels)

#--- Scatter plot over features
atr = ['PM2.5', 'NO2', 'CO', 'O3', 'AQI']
pd.plotting.scatter_matrix(data[atr], figsize = (20, 8))

plt.show()

feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
# Splitting the dataset into features (X) and targets (y)
X = data[feature_columns]
y = data['AQI']


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model on the data
history = model.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_split=0.2)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

loss = model.evaluate(X_test_scaled, y_test)
print("Mean Squared Error on Test Data:", loss)

model.save('AQImodel.h5')
joblib.dump(scaler, 'AQIscaler.pkl')