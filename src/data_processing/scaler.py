from sklearn.preprocessing import StandardScaler

def scale_data(df, features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].dropna())
    return scaled_data