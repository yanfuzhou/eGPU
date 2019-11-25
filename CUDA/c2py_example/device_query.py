import pycuda.autoinit
import pycuda.driver as cuda

(free, total) = cuda.mem_get_info()
print("Global memory occupancy:%f%% free" % (free*100/total))

for devicenum in list(range(cuda.Device.count())):
    device = cuda.Device(devicenum)
    attrs = device.get_attributes()
    print(attrs)
    # Beyond this point is just pretty printing
    print("\n===Attributes for device %d" % devicenum)
    for key, value in attrs.items():
        print("%s:%s" % (str(key), str(value)))
