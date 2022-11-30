import numpy as np
from numpy import *
import time
import matplotlib.pyplot as plt
start_time = time.time()
# This is the only scipy method you are allowed to use
# Use of scipy is not allowed otherwise
#from scipy.linalg import khatri_rao
import random as rnd
import time as tm

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL
# THE ONLY EXCEPTION TO THIS IS THE USE OF THE KHATRI-RAO PRODUCT METHOD FROM THE SCIPY LIBRARY
# HOWEVER, NOTE THAT NO OTHER SCIPY METHOD MAY BE USED IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHODS solver, get_features, get_renamed_labels BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def get_renamed_labels( y ):
	y_new = y
	for i in range(0,len(y_new)):
		if(y_new[i] == 0):
			y_new[i] = 1
		else :
			y_new[i]= -1

	y_new= np.array(y_new)

	return y_new.reshape( ( y_new.size, ) )					# Reshape y_new as a vector




################################
#  Non Editable Region Ending  #
################################

	# Since the dataset contain 0/1 labels and SVMs prefer -1/+1 labels,
	# Decide here how you want to rename the labels
	# For example, you may map 1 -> 1 and 0 -> -1 or else you may want to go with 1 -> -1 and 0 -> 1
	# Use whatever convention you seem fit but use the same mapping throughout your code
	# If you use one mapping for train and another for test, you will get poor accuracy



################################
# Non Editable Region Starting #
################################
def get_features( X ):
    X=np.array(X)

################################
#  Non Editable Region Ending  #
################################
    # X.push(1)  # one is pushed to make X of dimension = 9 to incorporate the contant term
    n = len(X)
    y=np.full(n,1)
    # print(X)
    X=np.append(X,y.reshape(len(y),1),axis=1)
    # print(X.shape)
    # X_new = []
    X_new = np.empty((n,729))
    # print(X_new.shape)
    for z in range(n):
        # Y_new = []
        # X[z].append(1)
        X[z]=get_renamed_labels(X[z])
        count=0
        for i in range(0,9):
            for j in range(0,9):
                for k in range(0,9):
                    X_new[z][count] = (X[z][i]*X[z][j]*X[z][k])
                    count+=1
    # print(X_new)
    return X_new

def my_features(X):
  #X.push(1)  # one is pushed to make X of dimension = 9 to incorporate the contant term
	n = len(X)
	X_new = []
	for i in range(0,n):
		for j in range(0,n):
			for k in range(0,n):
				X_new.append(X[i]*X[j]*X[k])
	return X_new


	# Use this function to transform your input features (that are 0/1 valued)
	# into new features that can be fed into a linear model to solve the problem
	# Your new features may have a different dimensionality than the input features
	# For example, in this application, X will be 8 dimensional but your new
	# features can be 2 dimensional, 10 dimensional, 1000 dimensional, 123456 dimensional etc
	# Keep in mind that the more dimensions you use, the slower will be your solver too
	# so use only as many dimensions as are absolutely required to solve the problem




################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
    (n, d) = X.shape
    t = 0
    totTime = 0
    # W is the model vector and will get returned once timeout happens
    # B is the bias term that will get returned once timeout happens
    # The bias term is optional. If you feel you do not need a bias term at all, just keep it set to 0
    # However, if you do end up using a bias term, you are allowed to internally use a model vector
    # that hides the bias inside the model vector e.g. by defining a new variable such as
    # W_extended = np.concatenate( ( W, [B] ) )
    # However, you must maintain W and B variables separately as well so that they can get
    # returned when timeout happens. Take care to update W, B whenever you update your W_extended
    # variable otherwise you will get wrong results.
    # Also note that the dimensionality of W may be larger or smaller than 9

    W = []
    B = 0
    tic = tm.perf_counter()
    ################################
    #  Non Editable Region Ending  #
    ################################

    # You may reinitialize W, B to your liking here e.g. set W to its correct dimensionality
    # You may also define new variables here e.g. step_length, mini-batch size etc
    C = 1
    neta = 1
    for i in range(729):
        W.append(1/4)
    W = np.array(W)




    data = np.append(X,y.reshape(len(y),1),axis=1)
    # data = loadtxt('train.dat')
    lis = []
    for i in range(0,256):
        lis.append(-1)
    lis = np.array(lis)
    p = 128
    out = 0
    n = len(data)
    for i in range(n):
        out = 0
        p=128
        for j in range(0,8):
            out+=int(p*data[i][j])
            p/=2
        lis[out] = int(data[i][8]) 

    count = 0
    for i in range(0,256):
        if(lis[i]==0 or lis[i]==1):
            count+=1

    if(True):
        cnt=np.empty([count,729],dtype=int)
        res=np.empty([count],dtype=int)

        number=0
        for i in range(256):
            if(lis[i]!=-1):
                arr=[]
                for j in reversed(range(8)):
                    arr.append(1&(i>>j))
                arr.append(1)
                arr=get_renamed_labels(arr)
                brr=np.empty([9],dtype=int)
                pro = 1
                for kk in range(8,-1,-1):
                    pro=pro*arr[kk]
                    brr[kk]=pro
                track = my_features(brr)
                for k in range(729):
                    cnt[number][k]=track[k]
                res[number]=lis[i]
                number+=1
        

        res=get_renamed_labels(res)
        c=0
        f=0
        tim =[]
        acc =[]
        while(c<1000):
            arr=np.zeros(729,dtype=float)
            for i in range(count):
                r=W.dot(cnt[i])
                r=r*res[i]
                if(r<1):
                    arr=arr-res[i]*cnt[i]
            if(f==0):
                arr=C*arr+W
        
            c+=1
            if(np.dot(arr,arr)<=0.1 and count>0):
                break
            W=W-(neta/(c))*arr/sqrt(np.dot(arr,arr))

            x = 0.0
            c1 = 0
            c2=0
            for i in range(count):
                r=W.dot(cnt[i])
                r=r*res[i]
                if(r>0):
                    c2+=1
                if(r<1):
                    x=x+(1-r)
                else:
                    c1 += 1

            x =C*x + W.dot(W)/2

            ac = c1/256
            if(c1>200):
                tim.append(time.time() - start_time)
                acc.append(ac)
            if(x<50):
                f=1
            if(x<100 and c1==count):
                break
        
        
        
    ################################
    # Non Editable Region Starting #
    ################################
    while True:
        t = t + 1
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                return ( W.reshape( ( W.size, ) ), B, totTime )			# Reshape W as a vector
            else:
                tic = tm.perf_counter()
    ################################
    #  Non Editable Region Ending  #
    ################################

        # Write all code to perform your method updates here within the infinite while loop
        # The infinite loop will terminate once timeout is reached
        # Do not try to bypass the timer check e.g. by using continue
        # It is very easy for us to detect such bypasses which will be strictly penalized
        
        # Note that most likely, you should be using get_features( X ) and get_renamed_labels( y )
        # in this part of the code instead of X and y -- please take care
        
        # Please note that once timeout is reached, the code will simply return W, B
        # Thus, if you wish to return the average model (as is sometimes done for GD),
        # you need to make sure that W, B store the averages at all times
        # One way to do so is to define a "running" variable w_run, b_run
        # Make all GD updates to W_run e.g. W_run = W_run - step * delW (similarly for B_run)
        # Then use a running average formula to update W (similarly for B)
        # W = (W * (t-1) + W_run)/t
        # This way, W, B will always store the averages and can be returned at any time
        # In this scheme, W, B play the role of the "cumulative" variables in the course module optLib (see the cs771 library)
        # W_run, B_run on the other hand, play the role of the "theta" variable in the course module optLib (see the cs771 library)

    return ( W.reshape( ( W.size, ) ), B, totTime )			# This return statement will never be reached






# C = 1
# neta = 1
# W = []
# for i in range(729):
#     W.append(random.random()/4-1/4)
# W = np.array(W)




# # data = np.append(X,y,axis=0)
# data = loadtxt('train.dat')
# A=get_features(data)
# print(A)
# lis = []
# for i in range(0,256):
#     lis.append(-1)
# lis = np.array(lis)
# p = 128
# out = 0
# n = len(data)
# for i in range(n):
#     out = 0
#     p=128
#     for j in range(0,8):
#         out+=int(p*data[i][j])
#         p/=2
#     lis[out] = int(data[i][8]) 

# count = 0
# for i in range(0,256):
#     if(lis[i]==0 or lis[i]==1):
#         count+=1

#     if(count==256):
#         cnt=np.empty([count,729],dtype=int)
#         res=np.empty([count],dtype=int)

#         number=0
#         for i in range(256):
#             if(lis[i]!=-1):
#                 arr=[]
#                 for j in reversed(range(8)):
#                     arr.append(1&(i>>j))
#                 arr.append(1)
#                 arr=get_renamed_labels(arr)
#                 brr=np.empty([9],dtype=int)
#                 pro = 1
#                 for i in range(8,-1,-1):
#                     pro=pro*arr[i]
#                     brr[i]=pro
#                 track = get_features(brr)
#                 for k in range(729):
#                     cnt[number][k]=track[k]
#                 res[number]=lis[i]
#                 number+=1

#         res=get_renamed_labels(res)
#         c=0
#         f=0
#         tim =[]
#         acc =[]
#         while(c<1250):
#             arr=np.zeros(729,dtype=float)
#             for i in range(count):
#                 r=W.dot(cnt[i])
#                 r=r*res[i]
#                 if(r<1):
#                     arr=arr-res[i]*cnt[i]
#             if(f==0):
#                 arr=C*arr+W
           
#             c+=1
#             if(np.dot(arr,arr)<=0.1):
#                 break
#             W=W-(neta/(c))*arr/sqrt(np.dot(arr,arr))

#             x = 0.0
#             c1 = 0
#             c2=0
#             for i in range(count):
#                 r=W.dot(cnt[i])
#                 r=r*res[i]
#                 if(r>0):
#                     c2+=1
#                 if(r<1):
#                     x=x+(1-r)
#                 else:
#                     c1 += 1

#             x =C*x + W.dot(W)/2
#             B=x

#             print(c1,c2,x,sep='\n',end='\n\n')
#             ac = c1/256
#             if(c1>200):
#                 tim.append(time.time() - start_time)
#                 acc.append(ac)
#             if(x<50):
#                 f=1
#             if(x<100 and c1==count):
#                 break
#         # plt.xlabel("Time")
#         # plt.ylabel("Accuracy")
#         # plt.plot(tim,acc)
#         # plt.show()
#         print(c)
#         # print(W)
#         print("--- %s seconds ---" % (time.time() - start_time))