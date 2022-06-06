import sys
sys.path.append('..\\')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input('./mnist_example.jpg', width=4, height=4),
    to_ConvConvRelu(name='conv1', s_filer=784, offset='(0,0,0)', to='(0,0,0)', width=(2, 2), height=16, depth=16, caption='Conv1+ReLU'),
    to_ConvConvRelu(name='conv2', offset='(1,0,0)', to='(1,0,0)', width=(2, 2), height=32, depth=32, caption='Conv2+ReLU'),
    to_Pool('pool', offset='(1,0,0)', to='(conv2-east)', width=1, height=16, depth=16, caption='Max Pool'),
    to_Conv('linear1', 25088, 1, offset='(2,0,0)', to='(2,0,0)', width=1, height=40, depth=40, caption='Linear1'),
    to_SoftMax('relu', s_filer=500, offset='(3,0,0)', to='(3,0,0)', width=3, height=3, depth=20, caption="ReLU"),
    to_Conv('linear2', 1024, 1, offset='(4,0,0)', to='(4,0,0)', width=1, height=20, depth=20, caption='Linear2'),
    to_connection('conv1', 'conv2'),
    to_connection('conv2', 'pool'),
    to_connection('pool', 'lineaer1'),
    to_connection('linear1', 'relu'),
    to_connection('relu', 'linear2'),
    to_end()
    ]

def main():
    # namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, 'my_cnn.tex')

if __name__ == '__main__':
    main()