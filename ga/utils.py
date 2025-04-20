import pandas as pd
import numpy as np

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

import numpy as np

def evaluate_function(genes, x_vals):
    """
    f(x) = exp(a) 
         + b*x 
         + c*x^2 
         + d*x^3 
         + e_coef*x^4 
         + g*x^5 
         + h*x^6 
         + i*x^7

    genes = [a, b, c, d, e_coef, g, h, i]
    """
    genes = np.array(genes, dtype=float)
    # precompute exp(a)
    const_term = np.exp(genes[0])
    
    # build the polynomial part: b*x + c*x^2 + ... + i*x^7
    # note: genes[1] multiplies x^1, genes[2] multiplies x^2, etc.
    # we can vectorize via a dot over powers 1..7
    powers = np.vstack([x_vals**j for j in range(1, len(genes))]).T  # shape (n,7)
    poly   = powers.dot(genes[1:])  # shape (n,)

    y_pred = const_term + poly
    
    # clamp any NaN or inf just in case
    return np.where(np.isfinite(y_pred), y_pred, 1e10)
    
#Mean Squared Error
def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error with safety checks
    """
    try:
        if len(y_true) != len(y_pred):
            return 1e10
            
        # Replace any NaN or inf values with large numbers
        y_pred = np.array([y if np.isfinite(y) else 1e10 for y in y_pred])
        
        mse = np.mean((np.array(y_true) - y_pred) ** 2)
        
        # If MSE is infinite or NaN, return a large value
        if not np.isfinite(mse):
            return 1e10
        
        return mse
    except Exception as e:
        print(f"Error in MSE calculation: {e}")
        return 1e10

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