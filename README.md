Nueral Network Workshop – Lab 1 macOS
Install Anaconda, Graphviz, and Python Packages
Click here for the Windows Instructions

Install GraphViz
Install https://www.macports.org/install.php
Open a Terminal Window
sudo port install graphviz (NOTE: If you do not have xcode installed this will install it for you, and the command will fail afterwards. Execute the command a second time to install)
Close Terminal Window
[collapse]
Install Anaconda and Python
Install macOS 64 bit Python 3.x https://www.anaconda.com/download/#macos
[collapse]
Install Packages
Open Terminal
conda install theano
conda install pydot
pip install tensorflow
pip install msgpack
pip install keras
conda update –all
Close Terminal
[collapse]
Test Install
Test Install
Launch Anaconda Navigator
Open Spyder
Create a New File
Add the following code and run it:
from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')# that-tensorflow-lab
