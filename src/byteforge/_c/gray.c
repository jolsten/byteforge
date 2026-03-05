/*
 * gray.c — GrayCode decode ufunc
 */

static void uint64_gray_decode(char **args, const npy_intp *dimensions,
                                const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0];
    npy_intp in2_step = steps[1];
    npy_intp out1_step = steps[2];

    for (i = 0; i < n; i++) {
        uint64_t val = *(uint64_t *)in1;
        uint8_t bit_width = *(uint8_t *)in2;
        uint64_t shift = 1;

        while (shift < bit_width) {
            val ^= val >> shift;
            shift <<= 1;
        }

        *((uint64_t *)out1) = val;

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

PyUFuncGenericFunction gray_decode_funcs[1] = {&uint64_gray_decode};

static char gray_decode_types[3] = {NPY_UINT64, NPY_UINT8, NPY_UINT64};
