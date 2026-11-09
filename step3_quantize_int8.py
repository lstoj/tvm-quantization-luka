import torch
import torchvision
import tvm
from tvm import relay
from tvm.contrib import graph_executor


model = torchvision.models.resnet50(weights = None).eval()

x = torch.randn(1,3,224,224)

scripted = torch.jit.trace(model, x).eval()


input_name = "input0"


mod, params = relay.frontend.from_pytorch(scripted, [(input_name, (1,3,224,224))])

#Step 3 addition
with relay.quantize.qconfig(calibrate_mode = "global_scale", global_scale = 8):
    mod = relay.quantize.quantize(mod,params)
# first line sets the quantization settings.
# instead of the weigths having a huge range, we assume they are between -8 and 8 and then map that range onto int8 (-128 to 127)
#second line converts the model's decimal number (float 32) into 8 bit integers (int8)
# should save compiling time and integer math is faster.

target = tvm.target.Target("llvm")

dev = tvm.cpu(0)

#compile step
with tvm.transform.PassContext(opt_level = 3):
    lib = relay.build(mod, target = target, params = params)

m = graph_executor.GraphModule(lib["default"](dev))


m.set_input(input_name, tvm.nd.array(x.numpy()))


m.run()


out = m.get_output(0).numpy()

#Step 5
print(m.benchmark(dev,repeat = 10, number = 10))


print(out.shape)

#Step 4 addition
open("relay_int8.txt", "w").write(str(mod))
#turns the quantized model into text and saves it to file called relay_int8.txt
print(str(mod).count('dtype="int8"'))
# shows how many time int8 appears