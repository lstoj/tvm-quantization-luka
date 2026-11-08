import torch
import torchvision
import tvm
from tvm import relay
from tvm.contrib import cc


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
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")

#Points to my CPU that my laptop has

#compile step
with tvm.transform.PassContext(opt_level = 3):
    lib_float = relay.build(mod, target = target, params = params)

lib_float.export_library("resnet_float_pi.so", cc.cross_compiler("aarch64-linux-gnu-g++"))

with relay.quantize.qconfig(calibrate_mode = "global_scale", global_scale = 8):
    qmod = relay.quantize.quantize(mod,params)
with tvm.transform.PassContext(opt_level = 3):
    lib_int8 = relay.build(qmod, target = target)

lib_int8.export_library("resnet50_int8_pi.so", cc.cross_compiler("aarch64-linux-gnu-g++"))