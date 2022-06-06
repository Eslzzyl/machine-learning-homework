import sys
sys.path.append('..\\')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input('./mnist_example.jpg', width=4, height=4),
    to_Conv('forward1', s_filer=784, n_filer=1, offset='(0,0,0)', to='(0,0,0)', height=32, depth=32, width=2, caption='Input'),
    to_SoftMax('relu1', s_filer=500, offset='(1,0,0)', to='(1,0,0)', width=3, height=3, depth=20, caption="ReLU"),
    to_Conv('forward2', s_filer=500, n_filer=1, offset='(2,0,0)', to='(2,0,0)', width=2, height=28, depth=28, caption='Output'),
    to_connection('forward1', 'relu1'),
    to_connection('relu1', 'forward2'),
    to_end()
    ]

def main():
    # namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, 'my_fnn.tex')

if __name__ == '__main__':
    main()