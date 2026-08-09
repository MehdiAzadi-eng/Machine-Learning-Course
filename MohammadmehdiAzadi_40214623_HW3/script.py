import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# just for loading dataset:
import os
from sklearn.datasets import load_breast_cancer

if not os.path.exists("breast_cancer.csv"):
    data = load_breast_cancer()
    df = pd.DataFrame(data.data , columns = data.feature_names)
    df['target'] = data.target
    df.to_csv("breast_cancer.csv" , index = False)
#-------------------------------------------------------------------------------------#
### Constants:
Epsilon = 1e-8
Population = 50
Generations = 100
Crossover_rate = 0.80
Mutation_rate = 0.05
Tournament_size = 3
Pairs = Population // 2
#-------------------------------------------------------------------------------------#
# He initialization:
chs = np.zeros((Population , 161))
# W1:
var1 = 2 / 30
std1 = np.sqrt(var1)
chs[: , :150] = std1 * np.random.randn(Population , 150)
# W2:
var2 = 2 / 5
std2 = np.sqrt(var2)
chs[: , 150:155] = std2 * np.random.randn(Population , 5)
# Saving:
np.savez("chromosomes.npz" , chromosomes = chs)

FILE = np.load("chromosomes.npz")
CHR = FILE['chromosomes']
#-------------------------------------------------------------------------------------#
### Definitions of Functions:

#|=============================================|#
#| 1. Calculating Fitnesses
#|=============================================|#
def Fitness(Y_true , Y_pred):
    Accuracy = np.mean(Y_true == Y_pred, axis=1) * 100
    return Accuracy

#|=============================================|#
#| 2. ReLU Activation Function
#|=============================================|#
def ReLU(x):
    return np.maximum(0, x)

#|=============================================|#
#| 3. Sigmoid Activation Function
#|=============================================|#
def Sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

#|=============================================|#
#| 4. Making Weights from Chromosomes
#|=============================================|#
def Chromosomes_to_Weights(Chs):
    w1 = Chs[: , :150].reshape(-1,5,30)
    w2 = Chs[: , 150:155].reshape(-1,1,5)
    b1 = Chs[: , 155:160].reshape(-1,5,1)
    b2 = Chs[: , 160].reshape(-1,1,1)
    return w1 , w2 , b1 , b2

#|=============================================|#
#| 5. Forward Path
#|=============================================|#
def feed_forward(Chs , X):
    w1 , w2 , b1 , b2 = Chromosomes_to_Weights(Chs)
    z1 = (w1 @ X.T) + b1
    a1 = ReLU(z1)
    z2 = (w2 @ a1) + b2
    a2 = Sigmoid(z2)
    y = np.where(a2 >= 0.5 , 1 , 0)
    return np.squeeze(y , axis=1)

#|=============================================|#
#| 6. Conducting the Tournaments
#|=============================================|#
def Tournament(Chs, fitness, k=Tournament_size):
    pop = fitness.shape[0]
    # Parent 1:
    arr = []
    idx = np.random.randint(0 , pop-1 , size=k).tolist()
    for index in idx:
        arr.append( (index , fitness[index]) )
    arr_sorted = sorted(arr , key=lambda x : x[1] , reverse=True)
    p1 = arr_sorted[0][0]
    # Parent 2:
    arr = []
    idx = np.random.randint(0 , pop-1 , size=k).tolist()
    for index in idx:
        arr.append( (index , fitness[index]) )
    arr_sorted = sorted(arr , key=lambda x : x[1] , reverse=True)
    p2 = arr_sorted[0][0]
    P1 = Chs[p1 , :]
    P2 = Chs[p2 , :]
    # Crossover Rate:
    cr = np.random.rand()
    if cr < Crossover_rate:
        # Making Children:
        split_point = int(np.random.randint(0 , 160))
        C1 = np.zeros_like(P1)
        C1[:split_point] = P1[:split_point]
        C1[split_point:] = P2[split_point:]
        C2 = np.zeros_like(P2)
        C2[:split_point] = P2[:split_point]
        C2[split_point:] = P1[split_point:]
    else:
        C1 = P1
        C2 = P2
    return C1 , C2

#|=============================================|#
#| 7. Performing Crossover on the Population
#|=============================================|#
def Crossover(Chs, fitness):
    New_Chs = np.zeros_like(Chs)
    C = 0
    for _ in range(Pairs):
        child1 , child2 = Tournament(Chs, fitness)
        New_Chs[C , :] = child1
        C += 1
        New_Chs[C , :] = child2
        C += 1
    return New_Chs

#|=============================================|#
#| 8. Performing Mutation on the Population
#|=============================================|#
def Mutation(Chs , Rate=Mutation_rate):
    Mask = np.random.random(Chs.shape)
    Mask = np.where(Mask < Rate , 1 , 0)
    Noise = np.random.randn(Chs.shape[0] , Chs.shape[1])
    Chs += Noise * Mask
    return Chs
#-------------------------------------------------------------------------------------#
### 1. Data Preprocessing and Preparation:
df = pd.read_csv("breast_cancer.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffling dataset
N_samples , N_columns = df.shape

N_Test = (30 * N_samples) // 100
N_Train = N_samples - N_Test
N_features = N_columns - 1

X_TRAIN = df.iloc[:N_Train , :-1].to_numpy()
X_TEST = df.iloc[N_Train: , :-1].to_numpy()

mean = np.mean(X_TRAIN, axis=0)
std = np.std(X_TRAIN, axis=0)

X_TRAIN_SCALED = (X_TRAIN - mean) / (std + Epsilon)
X_TEST_SCALED = (X_TEST - mean) / (std + Epsilon)
Y_TRAIN = df.iloc[:N_Train , -1].to_numpy()
Y_TEST = df.iloc[N_Train: , -1].to_numpy()
#-------------------------------------------------------------------------------------#
### 2. Encoding and Fitness Function:
neurons = [30 , 5 , 1]

C_W1 = neurons[1] * neurons[0]   # (5 , 30)
C_W2 = neurons[2] * neurons[1]   # (1 , 5)
C_b1 = neurons[1] * 1            # (5 , 1)
C_b2 = neurons[2] * 1            # (1 , 1)

C_tot = C_W1 + C_W2 + C_b1 + C_b2
print(f"The lengths of all the chromosomes: {C_tot}\n") # 161
#-------------------------------------------------------------------------------------#
### 3. Genetic Algorithm Settings/Parameters:
Max_Fitnesses = []
Mean_Fitnesses = []
best_index = None

for gen in range(Generations):
    Y_pred = feed_forward(CHR , X_TRAIN_SCALED)
    Gen_fitness = Fitness(Y_TRAIN , Y_pred)
    best_index = int(np.argmax(Gen_fitness))
    best_chromosome = CHR[best_index , :].copy()

    CHR = Crossover(CHR , Gen_fitness)
    CHR = Mutation(CHR)
    Max_Fitnesses.append( float(np.max(Gen_fitness)) )
    Mean_Fitnesses.append( float(np.mean(Gen_fitness)) )

np.savez("chromosomes.npz" , chromosomes = CHR)
#-------------------------------------------------------------------------------------#
### 4. Visualization and Reporting:
plt.figure(figsize=(10,5))
plt.plot(Max_Fitnesses, label='Max Fitness', color='blue')
plt.plot(Mean_Fitnesses, label='Mean Fitness', color='orange')
plt.title(f'Genetic Algorithm Learning Curve')
plt.xlabel('Generations')
plt.ylabel('Accuracy (%)')
plt.ylim([0 , 105])
plt.legend()
plt.grid(True)
plt.show()

Y_pred_test = feed_forward(best_chromosome.reshape(1,-1) , X_TEST_SCALED)
accuracy = Fitness(Y_TEST , Y_pred_test)
print(f"Test Accuracy: {accuracy[0]:.2f}%\n")

Y_pred_test_flat = Y_pred_test.flatten()

TP = np.sum((Y_TEST == 1) & (Y_pred_test_flat == 1))
TN = np.sum((Y_TEST == 0) & (Y_pred_test_flat == 0))
FP = np.sum((Y_TEST == 0) & (Y_pred_test_flat == 1))
FN = np.sum((Y_TEST == 1) & (Y_pred_test_flat == 0))

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

Confusion_Matrix = np.array([
    [TN , FP],
    [FN , TP]
])
Precision = (100 * TP) / (TP + FP)
Recall = (100 * TP) / (TP + FN)

print(f"\nConfusion Matrix:\n{Confusion_Matrix}")
print(f"Precision: {Precision:.2f}%")
print(f"Recall: {Recall:.2f}%\n")