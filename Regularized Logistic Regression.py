# This is the Logistic Regression Code


# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import numpy as np


def sigmoid_or_logistic_function(x):    
    return 1 / (1 + math.exp(-x))

def generating_features_array_upto_certain_power(df,degree):
    column_names = len(df.columns) 
    for i in range(2,degree+1):
        for j in range(0,i+1):
            temp = (df[0]**(i-j)) * (df[1]**j)
            df[column_names] = temp
            column_names = column_names + 1 
    return df

    




# Opening the file
file = pd.read_csv('ex2data2.txt',header = None)

# Separating the Independent and the Dependent Variables
IndependentVariable = file.iloc[:, 0:2].values
IndependentVariable = pd.DataFrame(IndependentVariable)
DependentVariable = file.iloc[:, 2].values


# Visualizing the Graphs
unique = list(set(DependentVariable))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [IndependentVariable[0][j] for j  in range(len(IndependentVariable)) if DependentVariable[j] == u]
    yi = [IndependentVariable[1][j] for j  in range(len(IndependentVariable)) if DependentVariable[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()
plt.show()



# Mapping feature to higher powers
Independent_Variables_with_additional_features = deepcopy(IndependentVariable)

Independent_Variables_with_additional_features = generating_features_array_upto_certain_power(Independent_Variables_with_additional_features,6)




# Variables and Lists Required during the whole program
list_containing_cost_at_each_thetas  = []
no_of_training_examples = len(Independent_Variables_with_additional_features)
No_of_thetas = len(Independent_Variables_with_additional_features.columns)
No_of_thetas = No_of_thetas + 1
predicted_values = []
mean_of_columns = []
standard_deviation_of_columns = []
array_containing_corresponding_thetas = []
Normalized_Independent_Variables = deepcopy(Independent_Variables_with_additional_features)
       



# Checking if no of features are greater than 2 then there is a need for 
# Feature Normalization 
# In this case we perform the mean Normalization of each feature

if No_of_thetas > 2:
    for loop_control_variable in range(0,No_of_thetas-1):
        mean = Independent_Variables_with_additional_features[[loop_control_variable]].mean()
        std = Independent_Variables_with_additional_features[[loop_control_variable]].std()     
        mean_of_columns.append(mean)
        standard_deviation_of_columns.append(std)
        for loop_control_variable_1 in range(0,len(Independent_Variables_with_additional_features)):
            Normalized_Independent_Variables[loop_control_variable][loop_control_variable_1] = ((Normalized_Independent_Variables[loop_control_variable][loop_control_variable_1] - mean_of_columns[loop_control_variable])/standard_deviation_of_columns[loop_control_variable])   
            
            
    




# Variables and Lists required during the Training Time after which we get the optimal values of Thetas

Values_of_thetas = [0] * No_of_thetas
h_of_x = 0
x_note = 1
array_containing_difference_of_actual_and_predicted_values = []
actual_values = deepcopy(DependentVariable)
array_storing_h_of_x_minus_y_into_x_values = []
updated_values_of_thetas = []       
alpha_value = 0.5
regularization_parameter = 100
array_containing_iteration_no = []


# Variables Used Specifically to logistic regression


# Changing will be made in Hypothesis Function and the Cost Function and the gradient descent 
# formula would remain the same except the changing of Hypothesis Function.
   

predicted_by_logistic_function = 0
predictions_by_logistic_functions = []
h_of_xs = []


for biggest_loop_control_variable in range(0,1500):

    predicted_values[:] = []
    # Hypothesis Equation 
    
    array_containing_iteration_no.append(biggest_loop_control_variable)
    for i in range(0,len(IndependentVariable)):
        h_of_x = 0
        for j in range(0,len(Values_of_thetas)):  
            if j == 0:
                h_of_x = h_of_x + Values_of_thetas[j] * x_note
            else: 
                h_of_x = h_of_x + Values_of_thetas[j] * Normalized_Independent_Variables[j-1][i]
        predicted_by_logistic_function = sigmoid_or_logistic_function(h_of_x)
        predicted_values.append(predicted_by_logistic_function)
            
    
    # Now calculating (h(x) - y)^2 to be used in calculating Theta and updating theta
    array_containing_difference_of_actual_and_predicted_values[:] = []
    temp = 0
    for k in range(0,len(actual_values)):
        temp = (-actual_values[k]*math.log(predicted_values[k])) - ((1-actual_values[k]) * math.log(1-predicted_values[k]))
        array_containing_difference_of_actual_and_predicted_values.append(temp) 
        
    
     
           
    # Now calculating Summation(h(x) - y)^2    
    sum_term = 0
    for l in range(0,len(array_containing_difference_of_actual_and_predicted_values)):
        sum_term = sum_term + array_containing_difference_of_actual_and_predicted_values[l]
        
        
    sum_term = (1 / no_of_training_examples) * sum_term
    
    # Adding the Regularization Parameter term into the cost
    array_containing_square_of_thetas = []
    for thetas in range(0,len(Values_of_thetas)):
        array_containing_square_of_thetas.append(Values_of_thetas[thetas]**2)
    sum_temp = 0    
    for thetas1 in range(0,len(Values_of_thetas)):
        sum_temp = sum_temp +  array_containing_square_of_thetas[thetas1]
    sum_temp = sum_temp * (regularization_parameter / (2 * no_of_training_examples))
    sum_term = sum_term + sum_temp
    ########################################################
    
    
    list_containing_cost_at_each_thetas.append(sum_term)
    array_containing_corresponding_thetas.append(Values_of_thetas[:])
     
    
    #latest_cost = list_containing_cost_at_each_thetas[-1] 
    
    # Now here i ll apply the gradient descent Method
    # to calculate the new values of thetas
    # Here i 'll add the Regularization Term for the Gradient Descent Formula
    
    
    array_storing_h_of_x_minus_y_into_x_values[:] = []
    updated_values_of_thetas[:] = []
    
    for m in range(0,No_of_thetas):
        for n in range(0,len(IndependentVariable)):
            if m == 0:
                temp = (predicted_values[n] - float(actual_values[n])) * x_note
                array_storing_h_of_x_minus_y_into_x_values.append(temp) 
            else:
                temp = ((predicted_values[n] - float(actual_values[n])) * Normalized_Independent_Variables[m-1][n])
                array_storing_h_of_x_minus_y_into_x_values.append(temp) 
        sum_of_h_of_x_minus_y_into_x_array = 0 
        for o in range(0,len(array_storing_h_of_x_minus_y_into_x_values)):
            sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array + array_storing_h_of_x_minus_y_into_x_values[o] 
        
        sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array / no_of_training_examples
        
        # Here the Regularization Term is added for the Gradient Descent Formula
        
        regularization_term = (regularization_parameter / no_of_training_examples) * Values_of_thetas[m]
        sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array + regularization_term
        
        ########################################################################
        
        sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array * alpha_value
        updated_values_of_thetas.append(Values_of_thetas[m] - sum_of_h_of_x_minus_y_into_x_array)



    #Values_of_thetas = updated_values_of_thetas        
    for loop in range(0,len(Values_of_thetas)):
        Values_of_thetas[loop] = deepcopy(updated_values_of_thetas[loop])



# Visualising the Graph between the number of iterations and the relevant costs deviation
plt.scatter(array_containing_iteration_no, list_containing_cost_at_each_thetas, color = 'red')
plt.title('Cost Deviation')
plt.xlabel('No of iterations')
plt.ylabel('Costs')
plt.show()




min_cost = min(list_containing_cost_at_each_thetas)
thetas_corresponding_to_min_cost = array_containing_corresponding_thetas[list_containing_cost_at_each_thetas.index(min(list_containing_cost_at_each_thetas))]



# Adding a column of 1's at the beginning of a dataframe
location_of_one_column = 0
new_col = [1] * len(Independent_Variables_with_additional_features) # can be a list, a Series, an array or a scalar   
Independent_Variables_with_additional_features.insert(location_of_one_column, 'Ones Column', new_col)


 
two_d_array_of_thetas = np.array(thetas_corresponding_to_min_cost)[np.newaxis]
two_d_array_of_thetas = two_d_array_of_thetas.T
dot_product = Independent_Variables_with_additional_features.dot(np.array(two_d_array_of_thetas))
 




# Visualizing the Graphs
unique = list(set(DependentVariable))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [IndependentVariable[0][j] for j  in range(len(IndependentVariable)) if DependentVariable[j] == u]
    yi = [IndependentVariable[1][j] for j  in range(len(IndependentVariable)) if DependentVariable[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))


plt.legend()



















