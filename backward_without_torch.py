#注：本文件是本人自己写的,为了自己试试手动反向传播并更加模块化，难免会和正规写法的参数名有区别，恳请谅解。
import torch
class Linear:
    def __init__(self,nin,nout,):
        self.weight=torch.randn((nin,nout),requires_grad=False)/nin**0.5
        self.bias=torch.zeros((nout,),requires_grad=False)
        self.weightgrad=torch.zeros((nin,nout),requires_grad=False)
        self.biasgrad=torch.zeros((nout,),requires_grad=False)
    def forward(self,input):
        out=input@self.weight+self.bias
        self.input=input
        self.output=out
        return out
    def backward(self,outputgrad):
        self.weightgrad=self.input.T@outputgrad
        self.inputgrad=outputgrad@self.weight.T
        self.biasgrad=outputgrad.sum(0)
        return self.inputgrad
    def requiregrad_parameters(self):
        x=[self.weightgrad,self.biasgrad]
        return x
    def parameters(self):
        return [self.weight,self.bias]
#调用的时候大概是一个括号套括号的套娃
class Tanh:
    def forward(self,input):
        out=torch.tanh(input)
        self.output=out
        return out
    def backward(self,outputgrad):#再说一下套路就是根据传入数据创建算符然后再在forward里让input传进来
        ones=torch.ones_like(self.output)
        inputgrad=outputgrad*(ones-self.output*self.output)
        return inputgrad
    def requiregrad_parameters(self):
        return []
    def parameters(self):
        return []
#那干脆保持套娃的调用

class Batchnorm:
    def __init__(self,fan_in,epsilon=1e-5,momentum=0.1):
        self.fan_in=fan_in
        self.epsilon=epsilon
        self.bngain=torch.ones((fan_in,),requires_grad=False)
        self.bnbias=torch.zeros((fan_in,),requires_grad=False)
        self.running_mean=torch.zeros((fan_in,),requires_grad=False)
        self.running_var=torch.ones((fan_in,),requires_grad=False)
        self.training=True
        self.momentum=momentum
    def forward(self,input):
        if self.training:
            mean=input.mean(0,keepdim=True)
            var=input.var(0,keepdim=True,unbiased=False)
        else:
            mean=self.running_mean
            var=self.running_var
        self.mean=mean
        self.input=input
        self.x=torch.sqrt(var+self.epsilon)
        fakeout=(input-self.mean)/self.x
        self.fakeout=fakeout
        self.rlout=fakeout*self.bngain+self.bnbias
        if self.training:
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*var
        return self.rlout
    def backward(self,outputgrad):
        part=(self.input*outputgrad).sum(0,keepdim=True)*(self.input-self.mean)+self.input.mean(0,keepdim=True)**2*outputgrad.sum(0,keepdim=True)-self.input*outputgrad.sum(0,keepdim=True)*self.mean
        fakeinputgrad=outputgrad/self.x-(1/outputgrad.shape[0])*(1/self.x)*outputgrad.sum(0,keepdim=True)-(1/outputgrad.shape[0])*(1/self.x**3)*part
        self.inputgrad=self.bngain*fakeinputgrad
        self.bnbiasgrad=outputgrad.sum(0)
        self.bngaingrad=(outputgrad*self.fakeout).sum(0)
        return self.inputgrad
#这个大长串我实在不想再写一遍
    def requiregrad_parameters(self):#类似在存应该梯度下降的对象
        self.requiregard_parameters=[self.bngaingrad,self.bnbiasgrad]
        return self.requiregard_parameters
    def parameters(self):
        return [self.bngain,self.bnbias]
#我默认这个embedding直接对样本数*没onehot的stocklength个字符的输入做处理，输出合并好的.view（）过的东西
class embedding:
    def __init__(self,num_char,dim_of_emb):#依次传入的是：self不说，numchar表示你总共的有多少元素（就是比如名字是27个）
#是用来做onehot的，然后dim_of_emb是传入你想把每个字母编码成多少维的随机vector。
        self.num_char=num_char
        self.dim_of_emb=dim_of_emb
        self.embedmat=torch.randn((self.num_char,self.dim_of_emb))
    def forward(self,input):
        self.embedmat_grad=torch.zeros_like(self.embedmat)
        triDmat=self.embedmat[input]#这是view前的已经把0-26换成embedding vector的那个三维张量
        self.embedded=triDmat.view(input.shape[0],-1)
        self.input=input
        return self.embedded
    def backward(self,outputgrad):
        outputgrad3D=outputgrad.view(outputgrad.shape[0],-1,self.dim_of_emb)#这个并不是我的设计，确实巧妙。我本来想循环+排序+重组之类的
        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                index=self.input[i,j]
                self.embedmat_grad[index]+=outputgrad3D[i,j,:]
        return self.embedmat_grad
    def parameters(self):
        return[self.embedmat]
    def requiregrad_parameters(self):
        return[self.embedmat_grad]

class CalLoss:
    def forward(self,input,answer):#这个input就已经是那个列数=numchar的那个最终计算loss前一步的矩阵了
#answer是预测结果的列表然后这个东西是按照0-26那个排法排的
        self.answer=answer
        self.FKcounts=input.exp()
        self.prob=self.FKcounts/self.FKcounts.sum(1,keepdim=True)
        self.x=self.FKcounts.sum(1,keepdim=True)
        self.loss=-self.prob[torch.arange(len(self.answer)),self.answer].log().mean()

        return self.loss
    def backward(self,outputgrad=None):#那这里是反向传播的头了不会再接受新东西了
        grad_after_softmax=torch.zeros_like(self.FKcounts)#这个东西是softmax后的prob对应位置对loss的导数
        for i,j in zip(range(self.FKcounts.shape[0]),self.answer):
            grad_after_softmax[i,j]-=(1/self.FKcounts.shape[0])*(1/self.prob[i,j])
        self.inputgrad=grad_after_softmax*self.FKcounts/self.x-(self.FKcounts*grad_after_softmax).sum(1,keepdim=True)*self.FKcounts/(self.x**2)

        return self.inputgrad
    def requiregrad_parameters(self):
        return []
    def parameters(self):
        return []


class operation:
    def __init__(self,layers):#layers should be a list also with softmax
        self.layers=layers
    def FORWARD(self,input,answer):
        for layer in self.layers[:-1]:
            input=layer.forward(input)
        out=self.layers[-1].forward(input,answer)
        return out
    def BACKWARD(self,output):
        for layer in reversed(self.layers):
            output=layer.backward(output)
        self.all_grad=[]
        self.all_parameters=[]
        for layer in self.layers:
            self.all_grad.extend(layer.requiregrad_parameters())
            self.all_parameters.extend(layer.parameters())#这种属性都像函数一样要让他返回只能加括号调用。

        
def one_training_loop(input,answer,ope,lr=0.1):#第二个位置传input的答案，第三个位置接受operation类,第四个传入参数更新率（默认0.1）
    output=ope.FORWARD(input,answer)
    backwardprocess=ope.BACKWARD(None)
    for i,j in zip(ope.all_parameters,ope.all_grad):
        i-=lr*j
    return output
#这个指代有些精妙，i=i-lr*j直接是错的
#详细说的话，比如a=torch.tenor((10,))
#然后i=a的话会创建一个新的名字i，但i和a指向的硬件上的东西是一个。但如果这时候你写i=i-2，那么实际上
#pytorch会先做i-2创造一个新的值，再让i指向这个新值。但注意a的指向始终没有变化，所以说你原先的
#更新式是无法更新的。//但是-=的逻辑是不一样的，i-=2做的是把原来i指向的那个tensor里的东西直接减2，而此时
#i和a都还是指向原来的结果。



        
#input和output的grad只是用来让所有层连起来，而真正梯度下降的参数会再存到一个名字里。
#下面大部分是粘贴batchnorm_version里的代码只是不使用torch自带backward函数

words=open('names.txt','r').read().splitlines()

chars=sorted(set(''.join(words)))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i+1:s for i,s in enumerate(chars)}
itos[0]='.'

stock_length=4


def makeegs(words):
    X=[]
    Y=[]
    for word in words:    
        chs=list(word)+['.']
        x=[0]*stock_length#initial
        for i in range(len(chs)):
            y=chs[i]
            numy=stoi[y]
            Y.append(numy)
            X.append(x)
            x=x[1:]+[numy]
    Xtensor=torch.tensor(X)
    Ytensor=torch.tensor(Y)
    return Xtensor,Ytensor

n_emb=15
batch_size=128
char_num=len(chars)+1
hidden1=200
hidden2=150
hidden3=100
hidden4=50
layers=[embedding(char_num,n_emb),Linear(stock_length*n_emb,hidden1),Batchnorm(hidden1),Tanh(),Linear(hidden1,hidden2),
        Batchnorm(hidden2),Tanh(),Linear(hidden2,hidden3),Batchnorm(hidden3),Tanh(),
        Linear(hidden3,hidden4),Batchnorm(hidden4),Tanh(),Linear(hidden4,char_num),CalLoss()]
op=operation(layers)

n1=int(0.8*len(words))
n2=int(0.9*len(words))
Xtr,Ytr=makeegs(words[:n1])
Xdev,Ydev=makeegs(words[n1:n2])
Xtst,Ytst=makeegs(words[n2:])

for _ in range(30):
    ix=torch.randint(0,len(Xtr),(batch_size,))
    input1=Xtr[ix]
    answer1=Ytr[ix]
    loss=one_training_loop(input1,answer1,op,lr=0.001)
    print(loss.item())

