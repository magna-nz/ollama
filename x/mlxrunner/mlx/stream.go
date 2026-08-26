package mlx

// #include "generated.h"
import "C"

import "log/slog"

type Device struct {
	ctx C.mlx_device
}

func (d Device) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer freeString(str)
	mlxCheck(C.mlx_device_tostring(&str, d.ctx))
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

var (
	defaultDevice    Device
	defaultDeviceSet bool
	defaultStream    Stream
	defaultStreamSet bool
)

func resetDefaultStreamCache() {
	defaultDeviceSet = false
	defaultStreamSet = false
}

func DefaultDevice() Device {
	if !defaultDeviceSet {
		d := C.mlx_device_new()
		mlxCheck(C.mlx_get_default_device(&d))
		defaultDevice = Device{d}
		defaultDeviceSet = true
	}

	return defaultDevice
}

// GPUIsAvailable returns true if a GPU device is available.
func GPUIsAvailable() bool {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	if dev.ctx == nil {
		panic(lastError())
	}
	defer freeDevice(dev)
	var avail C.bool
	mlxCheck(C.mlx_device_is_available(&avail, dev))
	return bool(avail)
}

// SetDefaultDeviceGPU sets the default MLX device to GPU.
func SetDefaultDeviceGPU() {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	if dev.ctx == nil {
		panic(lastError())
	}
	mlxCheck(C.mlx_set_default_device(dev))
	mlxCheck(C.mlx_device_free(dev))
	resetDefaultStreamCache()
}

type Stream struct {
	ctx C.mlx_stream
}

func (s Stream) LogValue() slog.Value {
	str := C.mlx_string_new()
	defer freeString(str)
	mlxCheck(C.mlx_stream_tostring(&str, s.ctx))
	return slog.StringValue(C.GoString(C.mlx_string_data(str)))
}

func DefaultStream() Stream {
	if !defaultStreamSet {
		s := C.mlx_stream_new()
		mlxCheck(C.mlx_get_default_stream(&s, DefaultDevice().ctx))
		defaultStream = Stream{s}
		defaultStreamSet = true
	}

	return defaultStream
}
