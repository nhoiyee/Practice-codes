#!/usr/bin/env python
# coding: utf-8

# Import Numpy in Program

# In[1]:


import numpy as np


# Create Vector

# In[3]:


#Create a row vector
row_vector = np.array([10,15,20])
print(row_vector)


# In[4]:


#Create a column vector
column_vector = np.array([[10],[20],[20]])
print(column_vector)


# Create Matrix

# In[6]:


#Create a Matrix
matrix = np.array([[10,20,30],[40,50,60],[1,2,3]])
print(matrix)


# Accessing Elements

# In[7]:


#Create a vector as a Row
row_vector = np.array([10,20,30,40,50,60])
print("row_vector ",row_vector)


# In[8]:


#Create a Matrix
matrix= np.array([[10,20,30],[40,50,60],[70,80,90]])
print("Matrix",matrix)


# In[9]:


print("row_vector[1]: ",row_vector[1])


# In[10]:


#Select 3rd row 3rd column
print("matrix[2,2]: ",matrix[2,2])


# In[11]:


#Select all elements of a vector
print("row_vector[:]: ",row_vector[:])


# In[12]:


#Select the last element
print("row_vector[-1]: ",row_vector[-1])


# In[13]:


#Select the first 2 rows and all the columns of the matrix
print("matrix[:2,:]: ",matrix[:2 ,:])


# Matrix

# In[14]:


#Create a Matrix
matrix= np.array([[10,20,30],[40,50,60],[70,80,90]])


# In[15]:


#View the number of rows and columns
print("Rows and columns: ",matrix.shape)


# In[16]:


#View the number of elements (rows*columns)
print("Total Elements",matrix.size)


# In[17]:


#View the number of Demensions (2 in this case)
print("Demension",matrix.ndim)


# Min and Max

# In[18]:


#Create a Matrix
matrix= np.array([[10,20,30],[40,50,60],[70,80,90]])
print(matrix)


# In[19]:


#Return the max element
print(np.max(matrix))


# In[20]:


#Return the min element
print(np.min(matrix))


# In[21]:


#To find the max element in each column
print(np.max(matrix,axis=0))


# In[23]:


#To find the max element in each row
print(np.max(matrix,axis=1))


# Calculate Average

# In[24]:


#Create a Matrix
matrix= np.array([[10,20,30],[40,50,60],[70,80,90]])
print(matrix)


# In[25]:


#mean
print(np.mean(matrix))


# Reshape

# In[26]:


#Create a Matrix
matrix= np.array([[10,20,30],[40,50,60],[70,80,90]])
print(matrix)


# In[28]:


print(matrix.reshape(9,-1))


# In[29]:


#Here -1 says as many columns as needed and 1 row
print(matrix.reshape(1,-1))


# In[30]:


#If we provide only 1 value Reshape would return a 1-d array of that length
print(matrix.reshape(9))


# In[31]:


#We can also use the flatten method to convert a matrix to 1-d array
print(matrix.flatten())


# Transpose

# In[32]:


#Create a Matrix
matrix= np.array([[10,20,30],[40,50,60],[70,80,90]])
print(matrix)


# In[33]:


#Transpose the matrix
print(matrix.T)


# Diagonal

# In[34]:


#Print the Principal diagonal
print(matrix.diagonal())


# Dot Product

# In[35]:


#Create vector_1
vector_1 = np.array([10,20,30])


# In[36]:


#Create vector_2
vector_2 = np.array([40,50,60])


# In[37]:


#Calculate Dot Product
print(np.dot(vector_1, vector_2))


# Adding, Subtracting and Multplying Matrices

# In[38]:


#Create a Matrix_1
matrix_1= np.array([[1,2,3],[4,5,6],[7,8,9]])

#Create a Matrix_2
matrix_2= np.array([[7,8,9],[4,5,6],[1,2,3]])

#Add the 2 Matrices
print(np.add(matrix_1,matrix_2))

#Subtraction
print(np.subtract(matrix_1,matrix_2))


# In[40]:


#Multiplcation (Element wise)
print(matrix_1*matrix_2)


# In[42]:


#Multiplcation (Row column wise)
print(np.matmul(matrix_1,matrix_2))


# Zeros and ones

# In[43]:


zeros = np.zeros([4,4])
print(zeros)


# In[44]:


ones = np.ones([3,3])
print(ones)


# Generate Random Values

# In[45]:


#Generate 3 random integers b/w 1 and 10
print(np.random.randint(0,11,3))


# In[46]:


#Generate 3 random from a normal distribution with mean 1.0 and std 2.0
print(np.random.normal(1.0,2.0,6))


# In[ ]:




