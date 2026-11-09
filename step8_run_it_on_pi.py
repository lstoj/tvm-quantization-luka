
#Script imported into the emulated Raspberry PI 
import numpy as np
import tvm
from tvm.contrib import graph_executor

dev = tvm.cpu(0)
for name in ["resnet_float_pi.so", "resnet50_int8_pi.so"]:
    lib = tvm.runtime.load_module(name) # loads the compiled model file
    m = graph_executor.GraphModule(lib["default"](dev)) #opens it on PI's CPU
    x = np.random.randn(1,3,224,224).astype("float32") # creates a random image
    m.set_input("input0", x) #enters the image into the model
    m.run() 
    print(m.benchmark(dev, repeat = 3, number = 1))