// Package mlx wraps the MLX C API.
//
// MLX keeps stream and backend state in thread-locals, so all calls into this
// package must come from a single goroutine locked to its OS thread (see
// internal/mlxthread).
package mlx

//go:generate go run generator/main.go -output=. ./include/mlx/c/*.h

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/include
// #cgo LDFLAGS: -lstdc++
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
// #include "generated.h"
// #include <string.h>
//
// static char _mlx_last_error_msg[1024] = {0};
// static int  _mlx_last_error_flag = 0;
//
// static void _mlx_capture_error_handler(const char* msg, void* data) {
//     (void)data;
//     strncpy(_mlx_last_error_msg, msg, sizeof(_mlx_last_error_msg) - 1);
//     _mlx_last_error_msg[sizeof(_mlx_last_error_msg) - 1] = '\0';
//     _mlx_last_error_flag = 1;
// }
//
// static void mlx_install_capture_handler(void) {
//     if (mlx_set_error_handler_) {
//         mlx_set_error_handler_(_mlx_capture_error_handler, NULL, NULL);
//     }
// }
//
// static void mlx_clear_last_error(void) {
//     _mlx_last_error_flag = 0;
//     _mlx_last_error_msg[0] = '\0';
// }
//
// static const char* mlx_get_last_error(void) {
//     return _mlx_last_error_flag ? _mlx_last_error_msg : "";
// }
import "C"

import (
	"fmt"
)

func init() {
	// Replace the default exit(-1) error handler with one that captures
	// the error message so we can surface it in Go.
	C.mlx_install_capture_handler()
}

// lastError consumes the captured MLX error, or returns nil when none is
// pending.
func lastError() error {
	msg := C.GoString(C.mlx_get_last_error())
	if msg == "" {
		return nil
	}
	C.mlx_clear_last_error()
	return fmt.Errorf("mlx: %s", msg)
}

// mlxCheck panics with the captured MLX error if a call failed. Most array
// operations cannot recover from a failed graph construction or evaluation.
func mlxCheck(ret C.int) {
	if ret != 0 {
		panic(lastError())
	}
}

// Deferred frees go through these helpers: defer evaluates a call's
// arguments immediately, so defer mlxCheck(C.mlx_..._free(v)) would
// free v on the spot and defer only the check.
func freeString(s C.mlx_string)            { mlxCheck(C.mlx_string_free(s)) }
func freeVectorArray(v C.mlx_vector_array) { mlxCheck(C.mlx_vector_array_free(v)) }
func freeClosure(c C.mlx_closure)          { mlxCheck(C.mlx_closure_free(c)) }
func freeStream(s C.mlx_stream)            { mlxCheck(C.mlx_stream_free(s)) }
func freeDevice(d C.mlx_device)            { mlxCheck(C.mlx_device_free(d)) }

// Version returns the MLX core library version string.
func Version() string {
	str := C.mlx_string_new()
	defer freeString(str)
	mlxCheck(C.mlx_version(&str))
	return C.GoString(C.mlx_string_data(str))
}

func doEval(outputs []*Array, async bool) {
	if len(outputs) == 0 {
		return
	}

	vector := C.mlx_vector_array_new()
	defer freeVectorArray(vector)

	for _, output := range outputs {
		if output != nil && output.Valid() {
			mlxCheck(C.mlx_vector_array_append_value(vector, output.ctx))
		}
	}

	if async {
		mlxCheck(C.mlx_async_eval(vector))
	} else {
		mlxCheck(C.mlx_eval(vector))
	}
}

func AsyncEval(outputs ...*Array) {
	doEval(outputs, true)
}

func Eval(outputs ...*Array) {
	doEval(outputs, false)
}

// MetalIsAvailable returns true if a Metal GPU is available.
func MetalIsAvailable() bool {
	var available C._Bool
	mlxCheck(C.mlx_metal_is_available(&available))
	return bool(available)
}

// CUDAIsAvailable returns true if a CUDA GPU is available.
func CUDAIsAvailable() bool {
	var available C._Bool
	mlxCheck(C.mlx_cuda_is_available(&available))
	return bool(available)
}
