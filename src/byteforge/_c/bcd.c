/*
 * bcd.c — BCD encode and decode ufuncs
 */

static void uint64_bcd_decode(char **args, const npy_intp *dimensions,
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
        uint64_t input = *(uint64_t *)in1;
        uint8_t max_digits = *(uint8_t *)in2;
        uint64_t result = 0;
        uint64_t multiplier = 1;
        int invalid = 0;
        int d;

        for (d = 0; d < max_digits; d++) {
            uint8_t nibble = (input >> (d * 4)) & 0xF;
            if (nibble >= 10) {
                invalid = 1;
                break;
            }
            result += (uint64_t)nibble * multiplier;
            multiplier *= 10;
        }

        *((uint64_t *)out1) = invalid ? UINT64_MAX : result;

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

static void uint64_bcd_encode(char **args, const npy_intp *dimensions,
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
        uint8_t max_digits = *(uint8_t *)in2;
        uint64_t result = 0;
        int d;

        for (d = 0; d < max_digits; d++) {
            result |= (val % 10) << (d * 4);
            val /= 10;
        }

        *((uint64_t *)out1) = result;

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

PyUFuncGenericFunction bcd_decode_funcs[1] = {&uint64_bcd_decode};
PyUFuncGenericFunction bcd_encode_funcs[1] = {&uint64_bcd_encode};

static char bcd_decode_types[3] = {NPY_UINT64, NPY_UINT8, NPY_UINT64};
static char bcd_encode_types[3] = {NPY_UINT64, NPY_UINT8, NPY_UINT64};
