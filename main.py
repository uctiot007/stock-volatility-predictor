from src.data_loader import get_data
from src.features import prepare_dataset
from src.model import gradient_descent, predict

if __name__ == "__main__":
    df = get_data()

    X, y, df_final = prepare_dataset(df)

    # Normalize features — important for gradient descent stability
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    # Train model
    w, b, loss_history = gradient_descent(X_norm, y, lr=0.1, epochs=500)

    # Predictions
    y_pred = predict(X_norm, w, b)

    print("\nFinal weights:", w)
    print("Final bias:", b)

    print("\nFirst 5 predictions vs actual:")
    for i in range(5):
        print(f"Pred: {y_pred[i]:.4f}, Actual: {y[i]:.4f}")