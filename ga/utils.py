import pandas as pd

#Load target Data
def load_target_values(filepath):
    # Skip lines starting with '#' (comments)
    data = pd.read_csv(filepath, comment='#')
    required_columns = {"x", "y"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}. Found: {data.columns}")
    if data.empty:
        raise ValueError("The CSV file is empty.")
    x = data["x"].values
    y = data["y"].values
    return x, y

#Evaluate function
def evaluate_function(coeffs, x_vals):
    # Ensure coeffs has exactly 8 values
    coeffs = coeffs[:8] if len(coeffs) > 8 else coeffs + [0] * (8 - len(coeffs))
    a, b, c, d, e, g, h, i = coeffs
    y_pred = [a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + g*x**2 + h*x + i for x in x_vals]
    return y_pred

#Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return sum((y_true - y_pred)**2) / len(y_true)

#Save results
def save_results(filepath, results):
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

#Load results
def load_results(filepath):
    return pd.read_csv(filepath).to_dict(orient="records")

#Plot convergence
def plot_convergence(results):
    import matplotlib.pyplot as plt
    plt.plot(results["generation"], results["best_fitness"])
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Convergence Plot")
    plt.show()