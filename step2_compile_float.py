import torch
import torchvision
import tvm
from tvm import relay
from tvm.contrib import graph_executor


# creates resnet50 with random weights and is set for "use" mode rather than training
model = torchvision.models.resnet50(weights = None).eval()

x = torch.randn(1,3,224,224)
# random image object

#coverts the image that is python into mathematical operations that our model can understand
scripted = torch.jit.trace(model, x).eval()


input_name = "input0"


# slightly confusing but it assigns the model's structure (layers and operations) to mod and model's numbers to params
#TVm translator into its own format (relay)
#mod contains the all the operations
#params-models's numbers
mod, params = relay.frontend.from_pytorch(scripted, [(input_name, (1,3,224,224))])


# tells the tvm copiler to compile for my laptop's CPU 
target = tvm.target.Target("llvm")

#Points to my CPU that my laptop has
dev = tvm.cpu(0)

#compile step
with tvm.transform.PassContext(opt_level = 3):
    lib = relay.build(mod, target = target, params = params)
# first line of the block assigns the effort the TVM compiler should try to optimize
# second line does the compiling, taking the model of the item and its weight and turns them into fast code for my CPU
# turns the modle into CPU code


#loads the compiled model onto my CPU
m = graph_executor.GraphModule(lib["default"](dev))

#it gives the model the image before it runs it
m.set_input(input_name, tvm.nd.array(x.numpy()))

# runs the model on computer
m.run()

# test part to see if the code above works
out = m.get_output(0).numpy()
#gets the model's output which shoudl be 1000

# gets the time how long the model runs for
# used for step 5
print(m.benchmark(dev,repeat = 10, number = 10))

# it should work if it prints [1,1000]
print(out.shape)


# errors that i experienced in this step
# My GPU was too new for TVM v0.15
# target = tvm.target.Target("cuda") had to change it to target = tvm.target.Target("cuda -arch=sm_90")
#switched from my gpu to cpu do to issues with compiling the model
open("relay_float.txt", "w").write(str(mod))
#turns the quantized model into text and saves it to file called relay_int8.txt
print(str(mod).count('dtype="int8"'))