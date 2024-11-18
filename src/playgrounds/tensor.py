import torch
import ctypes


def print_internal(t: torch.Tensor):
    current_memory_location = t.data_ptr()

    ending_memory_location = current_memory_location + t.storage().nbytes()

    while current_memory_location < ending_memory_location:
        string = ctypes.string_at(current_memory_location, t.element_size())
        tensor = torch.frombuffer(string, dtype=t.dtype)
        print(f"{current_memory_location} / {ending_memory_location} -> {tensor}")

        current_memory_location += t.element_size()
 
t = torch.arange(0, 24).reshape(1, 2, 3, 4)
print(t.storage_offset())
print(t[:, :, :, 2].storage_offset())
print(t[:, :, :, 2].is_contiguous())
print_internal(t[:, :, :, 2])

t_reshaped = t[:, :, :, 2].reshape(3, 2)
print(t_reshaped)
print(t_reshaped.is_contiguous())
print(t_reshaped.stride())

t_reshaped_contiguous = t_reshaped.contiguous()
print(t_reshaped_contiguous)
print(t_reshaped_contiguous.is_contiguous())
print(t_reshaped_contiguous.stride())
print_internal(t_reshaped_contiguous)