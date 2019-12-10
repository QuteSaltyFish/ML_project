import torch as t 

if __name__ == "__main__":
    a = t.tensor([1,2,3])
    b = t.tensor([4,5,6])

    t.save([a,b],'test.pth')
    [c, d] = t.load('test.pth')
    print(c, d)
