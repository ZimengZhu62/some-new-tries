import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
words=open('names.txt','r').read().splitlines()
#words=["emma","bean"]#for test

chars=sorted(set(''.join(words)))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i+1:s for i,s in enumerate(chars)}
itos[0]='.'
#原生字典没有按位置索引

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

stock_length=3
W1=torch.randn((stock_length*15,300),requires_grad=True)
B1=torch.randn((300),requires_grad=True)
W2=torch.randn((300,27),requires_grad=True)
B2=torch.randn(27,requires_grad=True)
embedding=torch.randn((27,15),requires_grad=True)#用15维向量表示每个字母
xenc=F.one_hot(torch.tensor(range(27)),num_classes=27).float()
parameters=[W1,B1,W2,B2,embedding]


#lre=torch.linspace(-3,0,100)
#lrs=10**lre
#lri=[]
#lossi=[]

#产生三个数据集
n1=int(0.8*len(words))
n2=int(0.9*len(words))
Xtr,Ytr=makeegs(words[:n1])
Xdev,Ydev=makeegs(words[n1:n2])
Xtst,Ytst=makeegs(words[n2:])

for i in range(200000):
    #make batchs
    ix=torch.randint(1,len(Xtr),(64,))
    Xemb=xenc@embedding#这个就是最终用来index的前面的那个矩阵
    Xenc=Xemb[Xtr[ix]]
    Xenc=Xenc.view(-1,stock_length*15)
    middle1=Xenc@W1+B1
    middle1=torch.tanh(middle1)
    logits=middle1@W2+B2
    #FKcounts=logits.exp()
    #prob=FKcounts/FKcounts.sum(1,keepdim=True)
    #loss=-prob[torch.arange(len(Y)),Y].log().mean()
    loss=F.cross_entropy(logits,Ytr[ix])
    #print(loss.item())
    for parameter in parameters:
        parameter.grad=None
    loss.backward()
    for parameter in parameters:
        parameter.data+= -0.1*parameter.grad
    #lossi.append(loss.item())
    #lri.append(lre[i])
    if i%1000==0:
        print(f"第{i}轮")
#plt.plot(lri,lossi)
#plt.show()
print(loss.item())


#暂时略去测试

#产生新名字
for _ in range(20):
    generated_names=[]
    context=[0]*stock_length
    while len(generated_names)<10:
        middle_generate=torch.tanh(Xemb[context].view(-1,45)@W1+B1)
        logits_generate=middle_generate@W2+B2
        prob_generate=F.softmax(logits_generate,dim=1)
        ixnext=torch.multinomial(prob_generate,num_samples=1).item()
        charnext=itos[ixnext]
        generated_names.append(charnext)
        context=context[1:]+[ixnext]
        if ixnext==0:
            break
    print(''.join(generated_names))


        


