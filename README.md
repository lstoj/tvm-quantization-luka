
Step 1:
I loaded ResNet50 from tourchvision and assigned it to variable model. I chose weights = none which means the model starts with random values. I made a random image assigning it to variable x (1,3,224,224). I ran the image through the model to see if the output of the shape was correct [1,1000]; It was because when I ran the file I gave me those values.

Step 2:
I took the ResNet50 model from step 1 and prepared it for TVM to compile. Using torch.jit.trace, I ran the image through the model once, which recorded its mathematical operations in a form TVM can understand. After that I imported it into TVM using the function relay.frontend.from_pytorch which split it into mod (the layers of operations of the model) and params (model's weights /the model's numerical values). Then I set the target to "llvm" which means compilling for my CPU. I set dev to point to my cpu where it will actually run. Then I compilled it with relay.build. (Compiled the model on my cpu since my GPU NVidia 5060 could not be used on older versions of tools used here). Using graph_executor.GraphModule function, I opened the compiled model on my cpu and then I inputted my fake image into it. With .run() I ran it image through the model and I was able to find the average time it takes to run to be around 68 milliseconds. 

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
TVM used int8 automatically as when I looked at TVM's source code in tvm/python/tvm/relay/quantize/quantize.py QConfig class had a list of default settings and the defaults were values like nbit_input,nbit_weight, dtype_input, and dtype_weigth and they were all assigned to int8 by default.

int 8: works perfectly

int 4: Quantizing worked, but it failed compiling. The error measssages I was getting were that the TVM's CPU code generator recuires every value to be at least 8 bits.

int 2: Quantizing failed and it never reached compiling. I am currently unaware why this happens to only int2 compare to int4 and int1.

int 1: Quantizing worked, but it failed compiling. TVM treats int1 like true/false or 0 and 1 but the values can go into negative so it can't properly compile.


Step 7:
For this step my laptop could not compile the code for Raspberry Pi, so I cross- compiled it aka built the model on my laptop that only runs for Pi's chip.

I changed the target from my CPU so TVM would compile for the Pi's ARM chip instead of my laptop's CPU. I installed an ARM compiler in Ubuntu with the command sudo apt install -y g++-aarch64-linux-gnu and used it to save each model as a file that Pi can load. I save both the files as .so as Pi can compile it 

From previous steps I learned how to quantizes the model and run it but I struggled with cross-compiling it and saving the model as a file. 


Step 8:
For this step I had to run my two compiled models from step 7 on an emulated Raspberry Pi. I used QEMU to emulate one. In order to accomplish this I had to downland Raspberry Pi OS Lite, extract the kernel and device tree files QEMU needs to boot it,set up a login user with SSH, and resize the disk image due to insufficient room for the TVM and my model files.

My first try used the newest Raspberry Pi OS, but it was too new for my QEMU version. I kept running into problems with the Pi rebooting itself and the login prompt never appearing. I fixed this issue by switching to an older version (Bullseye, 2023) which worked and allowed me to run the emulated version. On this version I also had to resize the disk due to it being full.

Since my .so files from step 7 needed TVM's runtime to run, I fed TVM source code to the emulated Pi. Once completed, using commands on the Pi I built just TVM's runtime (part that runs compiled models). Then I copied my to .so files to the Pi and ran a script found in step 8 file that loads each model, assigns a random image, and times it.


Results on the emulated Pi:

Float32: 15,699 ms
Int8: 5,903 ms

Based on these results the Int8 model ran approximately 2.7 times faster than the Float32 model on the emulated Raspberry Pi. My best guess on why the int8 ran faster on the Raspberry Pi compared to my laptop's CPU is that TVM has better default int8 code for type of processeing unit found in the Raspberry Pi compared to the CPU found in my laptop.

File size for int8 was around 6 times smaller when compared to float32 (31 MB vs 192 MB).

By far out of all the steps this one was the hardest to complete. I spent majority of my time trying to get the emulated Raspberry Pi to work, having to troubleshoot multiple times throughout the process. 