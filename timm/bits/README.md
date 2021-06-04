# Timm Bits

## Intro
A collection of reusable components and lightweight abstractions for training and evaluating NN with PyTorch.

This is an early WIP (consider it pre-alpha) with the primary goal to get up and running on TPUs w/ PyTorch XLA as the first priority. Expect significant changes, rewrites, additions, and of course bugs.

The current train.py and validate.py scripts are evolving to use the timm.bits components, they will also change significantly.

## Bits Design Brief

`bits` is designed to be a lightweight and modular set of training abstractions. It certainly shares concepts with other libraries (fastai, ignite, lightning, keras, etc, etc) but is not modeled after any specific one. It is supposed to be a 'bit different', hackable, and not everything to everyone.

`timm` models will always be useable in pure PyTorch w/o `bits` or anything besides the utils / helpers for pretrained models, feature extraction, default data config. I may breakout bits into a diff project if there is any interest besides my own use for timm image and video model training.

The layers:
* DeviceEnv - DeviceEnv dataclass abstraction deals with PyTorch CPU, GPU and XLA device differences, incl distributed helpers, wrappers, etc. There is more than a passing similarity to HuggingFace Accelerate, but developed in parallel and with some difference in the detail.
* Updater - Dataclass that combines the backward pass, optimizer step, grad scaling, grad accumulation is a possibly device specific abstraction.
  * Currently basic single optimizer, single forward/backward Updaters are included for GPU, XLA.
  * Deepseed will need its own Updater(s) since its Engine is a monolith of epic proportions that breaks all separations of concern in PyTorch (UGH!). NOTE Deepspeed not working yet nor is it a priority.
* Monitor - pull together all console logging, csv summaries, tensorboard, and WandB summaries into one module for monitoring your training.
* Checkpoint Manager - keeps track of your checkpoints
* Metrics - yet another set of metrics, although this may be replaced w/ an external set of classes. Uses same update / reset / compute interface as Ignite and Lightning (in theory interchangeable w/ an adapter). Metrics keep state on GPU / TPU to avoid device -> cpu transfers (esp for XLA).
* Task (not implemented yet) - combine your model(s) w/ losses in a task specific module, will also allow task factory for easy build of related metrics
* Train State - dataclasses to hold your tasks (models), updater state, etc
* Train Loop Functions (still in train.py script, not refined) - set of functions for train step, 'after step', evaluate using all of the components mentioned

How is this different than other options? 
* I'm very much trying to avoid a monolithic trainer / learner / model wrapping type class with billions of hooks (avoiding granular inversion of control!). 
* The goal is to provide reusable modules that can (hopefully) be mixed and matched w/ other code.
* Many of the components are based on Python dataclasses to reduce boilerplate.
* The train loop components are (will be) functional with easy to follow flow control, and are intended to be replaced when something different is needed, not augmented with hooks via callbacks or inheritence at every conceivable touch point.


## Quick Start

Most initial users will likely be interested in training timm models w/ PyTorch XLA on TPU-VM instances, this quick start will get you moving.

If you haven't noticed, this code is on a branch, make sure you checkout the `bits_and_tpu` branch on `timm` before doing this. You can test locally on your GPU too, in either XLA + GPU in a container or the usual PyTorch w/ GPU.

## Setup Python environment

This setup assumes you've SSH'd into your TPU-VM after setting it up (https://cloud.google.com/tpu/docs/users-guide-tpu-vm). Don't forget to do this in a TMUX session or you'll be sad if you lose your connection!

The TPU-VM instances I've been using have a usable version of PyTorch XLA 1.8.1 installed in the python3 environment, we will be using that.

I've found that leveraging TFDS w/ datasets in TFRecord format, streamed from Google Storage buckets is the most practical / cost-effective solution. I've written a PyTorch IterabeDataset wrapper around TFDS so we will install Tensorflow datasets and use that. Note that traditionaly PyTorch datasets on local disks do work both on TPU-VM, GPU cloud instances, or you local machine. Setting up persistent disks wasn't the easiest thing to do on TPU-VM for awhile so TFDS was my default.

One thing to watch, be very careful that you don't use a GS based dataset in a different continent from you TPU-VM instances. I burned through a few thousand USD leaving some wires crossed for 1 day. Otherwise the cost of training w/ buckets in same region are quite low.

### Install TFDS (if using GS buckets)

```
    pip3 install tensorflow-datasets
```

In some earlier tpu-vm instances the installed tensorflow version had issues with the GS bucket reading support and I often ended up installing a diff version. This could conflict with other use cases so only do it if needed.

```
    pip3 install --upgrade tensorflow-cpu
```

You may run into some numpy / pytorch version dependency issues here, try capping the version of tensorflow at 2.4.1 in above command.


### Get your dataset into buckets

You will need to host your dataset in buckets. I have tried creating custom datasets for this setup, but have used a number of TFDS datasets such as ImageNet, Flowers, caltech Birds, Oxford Pets that are available in TFDS.

The TFDS dataset pages (https://www.tensorflow.org/datasets/catalog/imagenet2012) have directions for the various datasets, I recommend building them in a different VM or local machine and then uploading to your training bucket. Many of them will auto-download and build the tfrecord shards for you. ImageNet needs to be downloaded manually.

### Use a custom allocator

With PyTorch XLA on a TPU-VM and TFDS you'll end up with a lot of processes and buffering. The instance memory will be used up quickly. I highly recommend using a custom allocator via `LD_PRELOAD`. tcmalloc may now be a default in the tpu-vm instanecs (check first). jemalloc also worked well for me. If LD_PRELOAD is not set in your env, do the following

```
    sudo apt update
    sudo apt install google-perftools
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
``` 

# Train, train, train

With all the above done, you should be ready to train... below is one particular train command I've just recently been using for some trials on vision MLP models...

Make sure the TPU config for PyTorch XLA on TPU-VM is set:
```
    export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

Then, launch fighters!

```
    python3 launch_xla.py --num-devices 8 train.py gs://my-imagenet-bucket --dataset tfds/imagenet2012:5.0.0 --model resmlp_24_224  --opt adamw --opt-eps 1e-6 --clip-grad 1.0 --drop-path 0.1 --mixup 0.5 --cutmix 1.0 --aa rand-m6-n4-mstd1.0-inc1 --weight-decay .08 --model-ema --model-ema-decay 0.99993 --sched cosine -j 4 --warmup-lr 1e-6 --warmup-epochs 20 --epochs 500 --lr 8.8e-4  -b 256
```

NOTE: build my TFDS dataset at ver 5.0.0 and it defaults to a newer version now. Change accordingly.

# Gotchas and Known Issues
* When PyTorch XLA crashes, you hit a TPU OOM etc, lots of processes get orphaned. Get in the habit of killing all python processes before starting a new train run.
  * `alias fml='pkill -f python3'`
* For TFDS use, due to the way PyTorch IterableDatasets work at the loader level, each worker process builds batches independently -- they are not dequeued and collated across workers. For validation especially, getting all the samples evenly divided across BOTH the distributed processes AND the dataset workers is a bit annoying. For now keeping the num_workers arg (j) low is advisable, especially for very small validation sets. This can limit your throughput though.
* Random erasing for on-device XLA tensors doesn't work. XLA isn't compatible with the array slicing approach to my RE impl, currently it's done by default after moving tensors to device. I need to fix.
* There are a number of models using ops that aren't lowered to XLA, this will REALLY slow things down to the point of being unusable. There are flags you can set to debug this, see PyTorch XLA troubleshooting page (https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md)
  * For NFNet models, force the ScaledStdConv `use_layernorm` arg to True, it is lowered, `std_mean` op is not
* This code doesn't currently work when float16 is forced via `XLA_USE_BF16=1` env arg, it will mess up metrics tensors that overflow in bfloat16. Better controlling model activation vs weight precision vs other tensors is a TODO.
* I haven't tested this code with pre TPU-VM (2-VM) setups, but it should work w/ correct config. I intend to make it work with Colab and Kaggle TPU notebooks soon.
* Your first batch, and generally first epoch will be slow with Pytorch XLA, after that things pick up and move along quickly. Be patient.

# Bugs and Discussion

If you find bugs (there are likely many), feel free to file an issue with `[BITS]` as the title prefix. Open a discussion if you have design ideas, again use `[BITS]` in the title.

# Acknowledgements

The TPU-VMs I've used for creating and testing this code, and that I hope to use for many future `timm` models were made available by the TPU Research Cloud (https://sites.research.google/trc/).
