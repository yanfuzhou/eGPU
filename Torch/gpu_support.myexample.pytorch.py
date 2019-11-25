import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("Num: %d" % torch.cuda.device_count())
