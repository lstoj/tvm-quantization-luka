
Step 1:
I loaded ResNet50 from tourchvision and assigned it to variable model. I did weights = none which means the model starts with random values. I made a fake random image assigning it to variable x (1,3,224,224). I ran the image through the model to see if the output of the shape was correct [1,1000]; It was because when I run the file step1 I get those values.

Step 2:
I took the ResNet50 model from step 1 and gave it to TVM to compile which coverted the image that is python into mathematical operations that our model can understand. Used the function torch.jit.trace. After that I imported it into TVM which split it into mod (the layers of operations of the model) and params (model's weights /the model's numerical values). Using the function relay.frontend.from_pytorch function. Then I set the target to "llvm" which means compilling for my CPU. I set dev to point to my cpu where it will actually run. Then I compilled it with relay.build. (Compiled the model on my cpu). Using graph_executor.GraphModule I opened the compiled model on my cpu and then I inputted my fake image into it. With .run() I ran it image through the model. I was able to find its average time it takes to run to be 68 milliseconds and the output shape which remained (1,1000) menaing that it worked.

For this section I struggled with the syntax and idea in general but watching few vidoes cleared it up as well as online syntax material. Prior to this I really had no experience with python used at such level.

Step 3:
I did the same thing as in step 2 but I additionally before compiling the model quantized the model to int8 using the relay.quantize function. Instead of the model's numbers being numbers from 32-bit deciamls(float32), I changed it to 8-bit whole numbers(int8) with range spanning from -128 to 127. Using global scale I assumed that the numbers would be between -8 and 8 (the usual default value). Then I converted the model to int8. By running it I still got that the shape was (1,1000) and the time it took to run was 208 ms which was around 3 times slower than float version.

For this problem I struggled with how to properly quantize the data with the available hint helped with along with instructional material available.

Step 4:
In this step I determined that the model was truly quantized. To see the model's code, I turned it into text with str(mod). In the procees I creted the file relay_float.txt and relay_int8.text(quantized) where the mod values where represented. Using the .count function I counted how many time the int8 data type  showed up in each one. The results were following:
Float Model = 0
Quantixed model = 137
SO the quantization really worked.
In this portion I had issues in properly determining if the model do possess the int8 values so I had to specify that it had to be dtype of int8.

Step 5:
I used the TVM's benchmark function .benchmark to time both models. It runs each model 100 times and gives the average time. The results were
Float32 : 67 ms
Int8 : 208 ms

Int8 model was 3 times slower even though it was quantized which would suggest it should be faster.

My personal guess why this might be the case is that the quantized model keeps converting between float and int8 and keeps rescalling each one by layers, which adds time the float model doesn't.

Another possibility is that TVM uses generic code instead of the code optimized for my CPU. The generic code probably works fine for float values but isnt optimized for int8 values resulting longer processing time.


Step 6:
TVm used int8 automatically as when I looked at TVM's source code in tvm/python/tvm/relay/quantize/quantize.py QConfig class had a list of default settings and the defaults were values like nbit_input,nbit_weight, dtype_input, and dtype_weigth and they were all assigned to int8 by default.

int 8: works perfectly

int 4: 