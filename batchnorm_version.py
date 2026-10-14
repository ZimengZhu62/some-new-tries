import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
words=open('names.txt','r').read().splitlines()

chars=sorted(set(''.join(words)))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i+1:s for i,s in enumerate(chars)}
itos[0]='.'

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

class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        #self.weight=torch.randn((fan_in,fan_out),requires_grad=True)
        #self.realweight=self.weight*(5/3)/(fan_in**0.5)
        #这个是规避下面不是leaf的那个情况，不过就是还有更好的做法那就是先等你初始化完再给这个整体加requiregrad
        self.weight=torch.randn((fan_in,fan_out))/(fan_in**0.5)
        self.bias=torch.zeros((fan_out,)) if bias else None
    def __call__(self,input):
        out=input@self.weight
        if self.bias==None:
            None
        else:
            out+=self.bias
        return out
    def parameters(self):
        out=[self.weight]+([] if self.bias==None else [self.bias]) 
        return out

class Batchnorm:
    def __init__(self,fan_in,epsilon=1e-5,momentum=0.1):
        self.bngain=torch.ones((fan_in,))
        self.bnbias=torch.zeros((fan_in,))#bngain和bnbias的更新过程在主程序里
        self.epsilon=epsilon
        self.training=True
        self.momentum=momentum
        self.running_mean=torch.zeros((fan_in,))#初始化不准确的影响会随着迭代次数升高而被削弱
        self.running_var=torch.ones(fan_in)

    def __call__(self,input):
        self.mean=input.mean(0,keepdim=True)
        self.var=input.var(0,keepdim=True)
        if self.training:
            mean=self.mean
            var=self.var
        else:
            mean=self.running_mean
            var=self.running_var
        #所以可以看出这个batchnorm必须是在循环外定义好再进循环更新，不然running根本更新不了
        out=(input-mean)/torch.sqrt(var+self.epsilon)
        out=out*self.bngain+self.bnbias
        if self.training:
            with torch.no_grad():
                #self.running_mean=input.mean(0,keppdim=True)
                #self.running_var=input.var(0,)
                #为何不能有上面那两行？如果有那么在训练时这个模块就不会迭代，因为每次都给新的输入先赋个值
                #那上一轮留下的结果就没用。
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*var
        return out
    def parameters(self):
        out=[self.bngain,self.bnbias]
        return out
class Tanh:
    def __call__(self,x):
        out=torch.tanh(x)
        return out
    def parameters(self):
        return[]


stock_length=4
#W1=torch.randn((stock_length*15,300),requires_grad=True)
#B1=torch.randn((300),requires_grad=True)
#W2=torch.randn((300,27),requires_grad=True)*0.01
##你注意你一开始这个写法会让W2直接不是leaf！就只有直接定义，不能做任何乘法这样才可以是leaf.(好坑。。)
#W2=torch.randn((300,27),requires_grad=True)
#B2=torch.zeros(27,requires_grad=True)
##这也是你一开始用randn*0也是一个问题一旦*了那randn本身就不是Leaf，所以就直接用zeros就是正确的。
#embedding=torch.randn((27,15),requires_grad=True)#用15维向量表示每个字母
#xenc=F.one_hot(torch.tensor(range(27)),num_classes=27).float()
#bngain=torch.ones((300,),requires_grad=True)
#bnbias=torch.zeros((300,),requires_grad=True)
#parameters=[W1,W2,B2,embedding,bngain,bnbias]

stock_length=4
n_emb=15
batch_size=128
char_num=len(chars)+1
hidden1=200
hidden2=150
hidden3=100
hidden4=50
layers=[Linear(stock_length*n_emb,hidden1),Batchnorm(hidden1),Tanh(),Linear(hidden1,hidden2),
        Batchnorm(hidden2),Tanh(),Linear(hidden2,hidden3),Batchnorm(hidden3),Tanh(),
        Linear(hidden3,hidden4),Batchnorm(hidden4),Tanh(),Linear(hidden4,char_num)]
xenc=F.one_hot(torch.tensor(range(27)),num_classes=27).float()
embedding=torch.randn((27,n_emb))#用n_emb维向量表示每个字母,require放在后面统一写

parameters=[embedding]
for layer in layers:
    parameters.extend(layer.parameters())

layers[-1].weight*=0.1#初始化让对下一个字符的预测尽量均匀

#初始化完再让他们有导数并作为leaf。leaf是精修后的
for parameter in parameters:
    parameter.requires_grad=True
    

n1=int(0.8*len(words))
n2=int(0.9*len(words))
Xtr,Ytr=makeegs(words[:n1])
Xdev,Ydev=makeegs(words[n1:n2])
Xtst,Ytst=makeegs(words[n2:])

for i in range(100000):
    #make batchs
    ix=torch.randint(1,len(Xtr),(batch_size,))
    Xemb=xenc@embedding#这个就是最终用来index的前面的那个矩阵
    Xenc=Xemb[Xtr[ix]]
    Xenc=Xenc.view(-1,stock_length*n_emb)
    for layer in layers:
        Xenc=layer(Xenc)
    logits=Xenc
    loss=F.cross_entropy(logits,Ytr[ix])
    loss.backward()
    for parameter in parameters:
        parameter.data+= -0.1*parameter.grad
    for parameter in parameters:
        parameter.grad=None
    if i%1000==0:
        print(f"第{i}轮")

print(loss.item())

#产生新名字
for _ in range(20):
    generated_names=[]
    context=[0]*stock_length

    #调节batchnorm为生成模式
    for layer in layers:
        if isinstance(layer,Batchnorm):
            layer.training=False
    while len(generated_names)<15:
        Xenc_generate=embedding[context]
        Xenc_generate=Xenc_generate.view(1,-1)
        for layer in layers:
            Xenc_generate=layer(Xenc_generate)
        prob_generate=F.softmax(Xenc_generate)
        ixnext=torch.multinomial(prob_generate,num_samples=1).item()
        charnext=itos[ixnext]
        generated_names.append(charnext)
        context=context[1:]+[ixnext]
        if ixnext==0:
            break
    print(''.join(generated_names))

