# FedAvg-with-Reproducing-Program
## Build an environment
```
$ python3 -m venv fl-test
$ cd fl-test/
$ source bin/activate
(fl-test) $ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
(fl-test) $ pip3 install pyyaml tensorflow tqdm
```

## Clone a Repository and Modify a bit
```
(fl-test) $ git clone https://github.com/vaseline555/Federated-Averaging-PyTorch.git
(fl-test) $ cd Federated-Averaging-PyTorch/
(fl-test) $ vim config.yaml
(fl-test) $ vim src/utils.py
(fl-test) $ git diff
diff --git a/config.yaml b/config.yaml
index 4124434..d442b19 100644
--- a/config.yaml
+++ b/config.yaml
@@ -1,6 +1,7 @@
 global_config:
   seed: 5959
-  device: "cuda"
+  #device: "cuda"
+  device: "cpu"
   is_mp: True
 ---
 data_config:
diff --git a/src/utils.py b/src/utils.py
index 85bfe0a..bbbcd70 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -5,6 +5,7 @@ import numpy as np
 import torch
 import torch.nn as nn
 import torch.nn.init as init
+import torchvision

 from torch.utils.data import Dataset, TensorDataset, ConcatDataset
 from torchvision import datasets, transforms
@@ -71,8 +72,9 @@ def init_net(model, init_type, init_gain, gpu_ids):
     Returns:
         An initialized torch.nn.Module instance.
     """
-    if len(gpu_ids) > 0:
-        assert(torch.cuda.is_available())
+    #if len(gpu_ids) > 0:
+    if len(gpu_ids) > 0 and torch.cuda.is_available():
+        #assert(torch.cuda.is_available())
         model.to(gpu_ids[0])
         model = nn.DataParallel(model, gpu_ids)
     init_weights(model, init_type, init_gain)
```

## Train
```
(fl-test) $ python main.py
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

TensorBoard 2.10.0 at http://0.0.0.0:5252/ (Press CTRL+C to quit)

[WELCOME] Unfolding configurations...!
{'global_config': {'seed': 5959, 'device': 'cpu', 'is_mp': True}}
{'data_config': {'data_path': './data/', 'dataset_name': 'MNIST', 'num_shards': 200, 'iid': False}}
{'fed_config': {'C': 0.1, 'K': 100, 'R': 500, 'E': 10, 'B': 10, 'criterion': 'torch.nn.CrossEntropyLoss', 'optimizer': 'torch.optim.SGD'}}
{'optim_config': {'lr': 0.01, 'momentum': 0.9}}
{'init_config': {'init_type': 'xavier', 'init_gain': 1.0, 'gpu_ids': [0, 1, 2]}}
{'model_config': {'name': 'CNN', 'in_channels': 1, 'hidden_channels': 32, 'num_hiddens': 512, 'num_classes': 10}}
{'log_config': {'log_path': './log/2022-09-10_00:18:53', 'log_name': 'FL.log', 'tb_port': 5252, 'tb_host': '0.0.0.0'}}

[Round: 0000] ...successfully initialized model (# parameters: 1662752)!
[Round: 0000] ...successfully created all 100 clients!
[Round: 0000] ...successfully finished setup of all 100 clients!
[Round: 0000] ...successfully transmitted models to all 100 clients!
[Round: 0001] Select clients...!
[Round: 0001] ...successfully transmitted models to 10 selected clients!
[Round: 0001] Start updating selected client 0001...!
[Round: 0001] Start updating selected client 0014...!
[Round: 0001] Start updating selected client 0016...!
[Round: 0001] Start updating selected client 0019...!
[Round: 0001] Start updating selected client 0071...!
[Round: 0001] Start updating selected client 0089...!
[Round: 0001] Start updating selected client 0099...!
[Round: 0001] Start updating selected client 0077...!
[Round: 0001] Start updating selected client 0036...!
[Round: 0001] Start updating selected client 0097...!
[Round: 0001] ...client 0077 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0099 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0097 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0036 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0019 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0071 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0016 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0001 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0089 is selected and updated (with total sample size: 600)!
[Round: 0001] ...client 0014 is selected and updated (with total sample size: 600)!
[Round: 0001] Evaluate selected 10 clients' models...!
        [Client 0071] ...finished evaluation!
        => Test loss: 0.0008
        => Test accuracy: 100.00%

        [Client 0077] ...finished evaluation!
        => Test loss: 0.0003
        => Test accuracy: 100.00%

        [Client 0001] ...finished evaluation!
        => Test loss: 0.0010
        => Test accuracy: 100.00%

        [Client 0089] ...finished evaluation!
        => Test loss: 0.0005
        => Test accuracy: 100.00%

        [Client 0019] ...finished evaluation!
        => Test loss: 0.0004
        => Test accuracy: 100.00%

        [Client 0097] ...finished evaluation!
        => Test loss: 0.0004
        => Test accuracy: 100.00%

        [Client 0016] ...finished evaluation!
        => Test loss: 0.0006
        => Test accuracy: 100.00%

        [Client 0014] ...finished evaluation!
        => Test loss: 0.0005
        => Test accuracy: 100.00%

        [Client 0036] ...finished evaluation!
        => Test loss: 0.0005
        => Test accuracy: 100.00%

        [Client 0099] ...finished evaluation!
        => Test loss: 0.0006
        => Test accuracy: 100.00%

[Round: 0001] Aggregate updated weights of 10 clients...!
[Round: 0001] ...updated weights of 10 clients are successfully averaged!
[Round: 0001] Evaluate global model's performance...!
        [Server] ...finished evaluation!
        => Loss: 2.3815
        => Accuracy: 29.68%

[Round: 0002] Select clients...!
[Round: 0002] ...successfully transmitted models to 10 selected clients!
:
```
