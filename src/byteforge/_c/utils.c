#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <stdint.h>

static int64_t twoscomp(uint64_t val, uint8_t bits) {
    uint64_t sign_bit = (uint64_t)1 << (bits - 1);
    if (val >= sign_bit) {
        return (int64_t)(val - ((uint64_t)1 << bits));
    }
    return (int64_t)val;
}
