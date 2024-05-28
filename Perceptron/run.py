import numpy as np
import pandas as pd

Xtrain_path = "./Xtrain.csv"
Ytrain_path = "./Ytrain.csv"

# PERCEPTRON

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    Xtrain_input = np.loadtxt(Xtrain_file, delimiter=',')
    Ytrain_label = np.loadtxt(Ytrain_file)
    
    
    training_shape = Xtrain_input.shape
    num_samples = training_shape[0]
    num_features = training_shape[1]
    # Bias term
    Xtrain_input = np.hstack((Xtrain_input, np.ones((num_samples, 1))))
    
    print(num_samples)
    
    t = 0
    max_epochs = 100
    k = 0
    
    w = np.array([np.zeros(num_features + 1)])
    c = [0]
    
    print(f"SHAPES: {Xtrain_input.shape}, {w.shape}")
    
    while t < max_epochs:
        for i in range(num_samples):
            xi = Xtrain_input[i]
            yi = Ytrain_label[i]
            
            product_result = np.dot(w[k], xi)
            product_result = np.sign(product_result)
            
            if product_result == -1:
                product_result = 0
            
            if product_result == yi:
                c[k] += 1
            else:                
                new_weight_vector = np.array(w[k]) + np.dot(yi, xi)
                np.append(w, new_weight_vector)
                # w.append(new_weight_vector)
                c.append(1)
                k = k + 1
        t = t + 1 
        
    # prediction part
    test_data = np.loadtxt(test_data_file, delimiter=",")
    predictions = []
    
    # for every test_data, predict its label
    for i in range(len(test_data)):
        y_result = predict(test_data[i], weights=w, c=c, max_k=k)
        predictions.append(y_result)
        # print(f"{i}: {y_result} == {Ytrain_label[i]}")
    
    np.savetxt(pred_file, predictions, "%d")


def predict(x_vector, weights, c, max_k):
    s = 0
    for i in range(max_k):
        s += c[i]*np.sign(np.dot(weights[i], x_vector))
        
    y_result = np.sign(s)
    if y_result == -1:
        y_result = 0
    
    return y_result
    

# ===== MAIN ===== 

run(Xtrain_path, Ytrain_path, Xtrain_path, "output.csv")