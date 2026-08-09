import numpy as np

class FuzzyPremiseLayer:
    def __init__(self, means, sigmas):
        self.means = means
        self.sigmas = sigmas

    def forward(self, X):
        # Reshape input X to a column vector (2, 1) for broadcasting
        X = np.array(X).reshape(2, 1) 
        
        # Calculate Gaussian membership degrees for all inputs
        # mu shape will be (2, 3)
        mu = np.exp(-0.5 * ((X - self.means) / self.sigmas)**2)
        
        # Separate the membership degrees for input X1 and input X2
        mu_x1 = mu[0] # Shape: (3,)
        mu_x2 = mu[1] # Shape: (3,)
        
        rules = np.outer(mu_x1, mu_x2).flatten()
        
        return rules

# --- Test Script ---
if __name__ == "__main__":
    # Test means matrix: row 1 for X1, row 2 for X2
    test_means = np.array([[1.0, 5.0, 9.0], 
                           [2.0, 4.0, 6.0]])
    
    # Test standard deviations matrix
    test_sigmas = np.array([[2.0, 2.0, 2.0], 
                            [1.5, 1.5, 1.5]])
    
    # Sample input vector [X1, X2]
    test_X = [3.0, 3.5] 
    
    # Instantiate the layer
    layer = FuzzyPremiseLayer(test_means, test_sigmas)
    
    # Compute forward pass
    output_rules = layer.forward(test_X)
    
    # Display results
    print("Input X: ", test_X)
    print("-" * 40)
    print("Output (Rule Firing Strengths):")
    print(np.round(output_rules, 4))
    print("-" * 40)
    print(f"Output Shape: {output_rules.shape} -> (Expected 9 rules)")
