import numpy as np
import pandas as pd
v_w,v_b=0,0

def train_test_split(x,y,test_size,shuffle=False):
    stop=len(x)-round(len(x)*test_size)
    if shuffle==True:
        x=x.sample(frac=1)
        #y=y.sample(frac=1)
    x_train=x.iloc[:stop]
    y_train=y.iloc[:stop]
    x_test=x.iloc[stop:]
    y_test=y.iloc[stop:]
    x_train = np.array(x_train.astype(np.float64).values.tolist())
    y_train = np.array([y_train.astype(np.float64).values.tolist()])
    y_train=y_train.T
    x_test = np.array(x_test.astype(np.float64).values.tolist())
    y_test = np.array(y_test.astype(np.float64).values.tolist())
    y_test=y_test.T
    return x_train,y_train,x_test,y_test

def initilization(inputsize,outputsize,hiddenlayers,hiddenlayer_neurons):
    weights=[]
    bias=[]
    for i in range((hiddenlayers+outputsize)):
        if i==0:
            wh=np.random.uniform(size=(inputsize,hiddenlayer_neurons[i]),low=-0.05,high=0.05)
            bh=np.random.uniform(size=(1,hiddenlayer_neurons[i]),low=-0.05,high=0.05)
            weights.append(wh)
            bias.append(bh)
        else:
            wh=np.random.uniform(size=(hiddenlayer_neurons[i-1],hiddenlayer_neurons[i]),low=-0.05,high=0.05)
            bh=np.random.uniform(size=(1,hiddenlayer_neurons[i]),low=-0.05,high=0.05)
            weights.append(wh)
            bias.append(bh)
    return weights,bias
def next_batch(X, y, batchSize):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    for i in np.arange(0, X.shape[0], batchSize):
        
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batchSize], y[i:i + batchSize])

def normalization(x):
    df=pd.DataFrame()
    for i in range(len(x.columns)):
        df[i]=(x.iloc[:,i])/(np.max(x.iloc[:,i]))
    return df

def MSE(actual,predicted,m,derivative=False):
    mse=[]
    if derivative==True:
        mse=-(2/m)*(actual-predicted)
        return mse
    mse=np.sum(np.square(actual-predicted))   
    return mse/m


def linear(x,derivative=False):
    if derivative==True:
        return np.full((len(x), 1), 1,dtype=np.float64)
    return x
def sigmoid(x, derivative=False):
    if (derivative == True):
        return x* (1 - x)
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return np.tanh(x)
def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass
            else:
                x[i][k] = 0
    return x

def error(act,pred):
    err=0.0
    for x,y in zip(pred,act):
        err+=(x-y)**2
        err=np.mean(err)
    return err
def grad_b(w,b,x,y,h_out):
    g_b=[]
    out_grad=-(2/len(y_train))*np.sum((y-h_out[-1]))*activation[-1](h_out[-1],derivative=True)
    #g_b.append(out_grad)
    for i in range(hiddenlayer,-1,-1):
        db=np.sum(out_grad,axis=0,keepdims=True)
        g_b.append(db)
        dh=np.dot(out_grad,w[i].T)
        out_grad=dh*activation[i-1](h_out[i-1],derivative=True)

    return g_b
def grad_w(w,b,x,y,h_out):
    grad=[]
    out_grad=-(2/len(y_train))*np.sum((y-h_out[-1]))*activation[-1](h_out[-1],derivative=True)
    #grad.append(out_grad)
    for i in range(hiddenlayer,-1,-1):
        if i!=0:
            dw=np.dot(h_out[i-1].T,out_grad)
        else:
            dw=np.dot(x.T,out_grad)
        grad.append(dw)
        dh=np.dot(out_grad,w[i].T)
        out_grad=dh*activation[i-1](h_out[i-1],derivative=True)
    return grad
def forward(x,w,b):
    output=[]
    for i in range(hiddenlayer):
        a=np.dot(x,w[i])+b[i]
        output.append(activation[i](a))
        x=output[i]
    a=np.dot(x,w[-1])+b[-1]
    output.append(activation[-1](a))
    e=error(Y,output[-1])
    return output,e
def do_gradient_descent(x,y,w,b,out):
   lr=0.0002
   #dw,db=0,0
   dw =grad_w(w,b,x,y,out)
   db =grad_b(w,b,x,y,out)
   for i in range(hiddenlayer+1):
       w[i]=w[i]-lr*(dw[hiddenlayer-i])
       b[i]=b[i]-lr*(db[hiddenlayer-i])
   return w,b

def momentum_gradient_descent(x,y,w,b,out):
   lr,gamma=0.0002,0.9
   #dw,db=0,0
   dw =grad_w(w,b,x,y,out)
   db =grad_b(w,b,x,y,out)
   for i in range(hiddenlayer+1):
       v_w=gamma*prev_w[i]+lr*dw[hiddenlayer-i]
       v_b=gamma*prev_b[i]+lr*db[hiddenlayer-i]
       w[i]=w[i]-v_w
       b[i]=b[i]-v_b
       prev_w[i]=v_w
       prev_b[i]=v_b
   return w,b
def nesterov_gradient_descent(x,y,w,b,out):
   lr,gamma=0.002,0.89
   n_w,n_b=[],[]
   for i in range(hiddenlayer+1):
       v_w.append(gamma*prev_w[i])
       v_b.append(gamma*prev_b[i])
       n_w.append(w[i]-v_w[i])
       n_b.append(b[i]-v_b[i])
   dw =grad_w(n_w,n_b,x,y,out)
   db =grad_b(n_w,n_b,x,y,out)
   for i in range(hiddenlayer+1):
       v_w[i]=gamma*prev_w[i]+lr*dw[hiddenlayer-i]
       v_b[i]=gamma*prev_b[i]+lr*db[hiddenlayer-i]
       w[i]=w[i]-v_w[i]
       b[i]=b[i]-v_b[i]
       prev_w[i]=v_w[i]
       prev_b[i]=v_b[i]
   return w,b


def adagard(x,y,w,b,out):
   lr,eps=0.02,1e-8
   #dw,db=0,0
   dw =grad_w(w,b,x,y,out)
   db =grad_b(w,b,x,y,out)
   for i in range(hiddenlayer+1):
       prev_w[i]=prev_w[i]+np.square(dw[hiddenlayer-i])
       prev_b[i]=prev_b[i]+np.square(db[hiddenlayer-i])
       w[i]=w[i]-(lr/np.sqrt(prev_w[i]+eps))*dw[hiddenlayer-i]
       b[i]=b[i]-(lr/np.sqrt(prev_b[i]+eps))*db[hiddenlayer-i]
   return w,b
#x=pd.read_csv("TCS.csv")
cols=["a", "b", "c", "d","e","f","g","h"]
df = pd.read_csv("AXISBANK.csv",names=cols)
for i in cols:
    df[i] = df[i].astype(str).str.replace(",","").astype(float)
x=df
y_max=max(x.iloc[:,-1])
x=normalization(x)
print(y_max)
Y=x.iloc[:,-1]
print(x)

#x=x.head(65)
x=x.drop(x.columns[-1],axis=1)

#print(y_max,"yahi print hua")

#y=y.shift(-1)
X=x #.drop(x.columns[1],axis=1)
inputsize=len(X.columns)
print(x)
outputsize=1
hiddenlayer=int(input("Enter no. of hiddenlayers"))
hiddenlayer_neurons=[]
activation=[]
for i in range(hiddenlayer):
    j=int(input("Enter hiddenlayer neurons"))
    hiddenlayer_neurons.append(j)
hiddenlayer_neurons.append(outputsize)

for i in (range(len(hiddenlayer_neurons))):
    if i==(len(hiddenlayer_neurons)-1):
        k=input("Enter Output Layer Activation Function")
    else:
        k=input("Enter hiddenlayer Activation Function")
    activation.append(locals()[k])

loss='MSE'
xtrain,ytrain,xtest,ytest=[],[],[],[]
test_s=0.2
x_train,y_train,x_test,y_test=train_test_split(X,Y,test_s,shuffle=False)
wts,bs=initilization(inputsize,outputsize,hiddenlayer,hiddenlayer_neurons)

prev_w=[np.array(np.zeros(i.shape))for i in wts]
prev_b=[np.zeros(i.shape)for i in bs]
m_w=[np.array(np.zeros(i.shape))for i in wts]
m_b=[np.zeros(i.shape)for i in bs]

for i in range(1000):
    e=0
    for (batchX, batchY) in next_batch(x_train, y_train,len(x_train)):
        dw,db=0,0
        out,err=forward(batchX,wts,bs)
        e +=e+err
        wts,bs=do_gradient_descent(batchX,batchY,wts,bs,out)
        #wts,bs=momentum_gradient_descent(batchX,batchY,wts,bs,out)
        #wts,bs=nesterov_gradient_descent(batchX,batchY,wts,bs,out)
        #wts,bs=RMSProp(batchX,batchY,wts,bs,out)
        #wts,bs=adagard(batchX,batchY,wts,bs,out)
        #wts,bs=adam(batchX,batchY,wts,bs,out,i)
    #print(e)
out1,err2=forward(x_test,wts,bs)
print(out1[-1]*y_max)
print(y_test*y_max)
