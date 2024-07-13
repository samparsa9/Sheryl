from sklearn.preprocessing import StandardScaler

# Uses StandardScaler from sci-kit learn library to scale the features of each symbol so it can be passed
# into the K means clustering algorithm
def scale_data(df, features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].dropna())
    return scaled_data