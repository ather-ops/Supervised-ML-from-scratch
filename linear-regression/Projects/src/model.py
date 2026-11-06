import numpy as np

# Prediction function
def predict(x,w,b):
    pred = x@w +b
    return pred


# Install parameters function
def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0.0
    return w,b

# Installin MSE function

def mean_squared_error(y_actual, y_pred):
    errors = y_actual - y_pred
    mse = np.mean(errors**2)
    return mse

# Training loop
def train(x,y):
    lr = 0.1
    epochs = 1000
    n = len(x)
    n_features = x.shape[1]
    w , b = initialize_parameters(n_features)
    loss_history = []
    for  epoch in range(epochs):
        y_pred = predict(x,w,b)
        error = y_pred - y
        loss = mean_squared_error(y,y_pred)

        dw = (2/n)*(x.T@error)
        db = (2/n)*np.sum(error)

        w = w-lr*dw
        b = b-lr*db

        loss_history.append(loss)

    return w,b,loss_history

# Final evalution

def evaluate_model(y_actual, y_pred):

    mae = np.mean(np.abs(y_actual - y_pred))
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }
