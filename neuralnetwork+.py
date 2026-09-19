import math
import random
class value:
    def __init__(i,data,children=(),_op=' ',grad=0):
        i.data=data
        i.prev=set(children)
        i._op=_op
        i.grad=grad
        def pbackward():
            None
        i.pbackward=pbackward
    def __repr__(self):
        #你看,写一个a等价于调用a.repr(),由于repr只要一个参数所以括号里啥都不写。
        return(str(self.data))
        
    def __add__(i,j):
        if isinstance(j,value):
            out=value(i.data+j.data,(i,j),'+',0)
        else:
            j=value(j)
            out=value(i.data+j.data,(i,j),'+',0)
        #你看这里的out就是init里的第一位，也就是i。那么存入的结果就是out.一堆。然后value里第一个写的其实是init里的第二个。
        def pbackward():
            i.grad+=out.grad
            j.grad+=out.grad
        out.pbackward=pbackward
        return out
    def __radd__(i,j):
        return i+j
    
    def __mul__(i,j):
        if isinstance(j,value):
            out=value(i.data*j.data,(i,j),'*',0)
        else:
            j=value(j)
            out=value(i.data*j.data,(i,j),'*',0)
        def pbackward():
            i.grad+=j.data*out.grad
            j.grad+=i.data*out.grad
        out.pbackward=pbackward
        return out
    def __rmul__(i,j):
        return i*j
    def sq(i):
        out=value(i.data*i.data,(i,),'**2',0)
        def pbackkward():
            i.grad+=2*i.data*out.grad
        out.pbackward=pbackkward
        return out
    def __sub__(i,j):
        out=value(i.data-j.data,(i,j),'-',0)
        def pbackkward():
            i.grad+=out.grad
            j.grad+=(-1)*out.grad
        out.pbackward=pbackkward
        return out
    def tanh(i):
        x = i.data
        t =(math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = value(t, (i, ), 'tanh')

        def pbackward():
            i.grad += (1 - t**2) * out.grad
        out.pbackward = pbackward
        return out
    def wbackward(i):
        i.grad=1.0
        visited=set()
        topofalse=[]
        def topo(i):
            if i not in visited:
                visited.add(i)
                for child in i.prev:
                    topo(child)
                topofalse.append(i)
        topo(i)
        topofalse.reverse()    
        for j in topofalse:
            j.pbackward()

#下面这些都是init下在定义算符(算符里面有啥，存了啥），call下在定义调用（调用的时候怎么去用算符存好的属性计算）
class neuron:#（neuron里该存被value的权重向量与偏置。调用时要点乘，加偏置并激活）
    def __init__(self,nin):
        self.weight=[value(random.uniform(-1,1) ) for _ in range(nin)]
        self.bias=value(random.uniform(-1,1))
    def __call__(self,other):
        act=sum((i*j for i,j in zip(self.weight,other)),self.bias)
        out=act.tanh()
        return out
    def parameters(self):
        return self.weight+[self.bias]
    
class layer:#（layer里应该存着和输出数一样的不同neuron方法数，调用时要直接进行neuron（）的操作并存一个表或者说向量）
    def __init__(self,nin,nout):
        self.neurons=[neuron(nin) for _ in range (nout)]
        #self.weight=[x.weight for x in self.neurons]
        #self.bias=[x.bias for x in self.neurons]
    def __call__(self,inputvector):
        out=[x(inputvector) for x in self.neurons]
        return out
    def parameters(self):
        out=[]
        for neuron in self.neurons:
            s1= neuron.parameters()
            out.extend(s1)
        return out 
class MLP:#（MLP里是通过几次layer来实现，存就应该存着维度和输入对应的layer方法，调用就是挨个调用）
    def __init__(self,nin,nouts):#nouts是一个列表表示每一步分别压缩成多少维
        shapeadjust=[nin]+nouts
        self.layers=[layer(shapeadjust[i],shapeadjust[i+1]) for i in range(len(nouts))]
    def __call__(self,inputvector):
        for layer in self.layers:
            inputvector=layer(inputvector)
        return inputvector
    def parameters(self):
        out=[]
        for layer in self.layers:
            s1=layer.parameters()
            out.extend(s1)
        return out
#注意逻辑，那就是先定义“算符”，即用n=mlp（里面填mlp的参数），然后把处理对象扔进定义好的算符n里（n此时已经是mlp的实例了）

#对训练sinx的尝试：

xs = [value(i * 0.1 - 3) for i in range(61)]
ys = [value( math.sin(x.data)) for x in xs]
n=MLP(1,[3,12,2,1])
i=1
while i<=30000:
    studylength=0.002*(1-i/30000)
    losses=[]
    for x,y in zip(xs,ys):
        ypred=n([x])[0]
        partialloss=(y-ypred).sq()
        losses.append(partialloss)
    loss=sum(losses)
    for j in n.parameters():
        j.grad=0
    loss.wbackward()
    for j in n.parameters():
        j.data-=j.grad*studylength
    print("第"+str(i)+"轮， "+str(loss.data))
    i+=1

x=[value(1.5)]
print(str(n(x)[0].data))

    


