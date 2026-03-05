/*
 * ibm.c — IBM hexadecimal floating point 32-bit and 64-bit encode/decode ufuncs
 *
 * Layout: [sign 1 | exponent 7 | mantissa (24 or 56)]
 * Decode: value = (-1)^s * (m / 2^mbits) * 16^(e - 64)
 */

/* --- Decode --- */

static void uint32_ibm32_decode(char **args, const npy_intp *dimensions,
                                 const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    for (i = 0; i < n; i++) {
        uint32_t dn = *(uint32_t *)in1;

        uint32_t s = (dn >> 31) & 1;
        uint32_t e = (dn >> 24) & 0x7F;
        uint32_t m = dn & 0xFFFFFF;

        if (m == 0) {
            *((double *)out1) = 0.0;
        } else {
            double sign = (s == 0) ? 1.0 : -1.0;
            double M = (double)m / (double)(1 << 24);
            int32_t E = (int32_t)e - 64;
            *((double *)out1) = sign * M * pow(2.0, 4.0 * (double)E);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint64_ibm64_decode(char **args, const npy_intp *dimensions,
                                 const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    for (i = 0; i < n; i++) {
        uint64_t dn = *(uint64_t *)in1;

        uint64_t s = (dn >> 63) & 1;
        uint64_t e = (dn >> 56) & 0x7F;
        uint64_t m = dn & 0x00FFFFFFFFFFFFFFULL;

        if (m == 0) {
            *((double *)out1) = 0.0;
        } else {
            double sign = (s == 0) ? 1.0 : -1.0;
            /* 56-bit mantissa: split to avoid precision loss */
            double M = (double)m / (double)((uint64_t)1 << 52) / 16.0;
            int32_t E = (int32_t)e - 64;
            *((double *)out1) = sign * M * pow(2.0, 4.0 * (double)E);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* --- Encode --- */

static void float64_ibm32_encode(char **args, const npy_intp *dimensions,
                                  const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    for (i = 0; i < n; i++) {
        double val = *(double *)in1;

        if (val == 0.0) {
            *((uint32_t *)out1) = 0;
        } else {
            uint32_t sign = (val < 0) ? 1 : 0;
            double abs_val = fabs(val);

            /* hex_exp = ceil(log2(abs_val) / 4) */
            double log2_val = log2(abs_val);
            int32_t hex_exp = (int32_t)ceil(log2_val / 4.0);

            /* mant_frac = abs_val / 16^hex_exp, should be in [1/16, 1) */
            double mant_frac = abs_val / pow(2.0, 4.0 * (double)hex_exp);

            /* Normalize: if rounding pushes >= 1, increment exponent */
            if (mant_frac >= 1.0) {
                hex_exp++;
                mant_frac = abs_val / pow(2.0, 4.0 * (double)hex_exp);
            }

            uint32_t m_val = (uint32_t)round(mant_frac * (double)(1 << 24));
            if (m_val > 0xFFFFFF) m_val = 0xFFFFFF;

            int32_t e_biased = hex_exp + 64;
            if (e_biased < 0) e_biased = 0;
            if (e_biased > 127) e_biased = 127;

            *((uint32_t *)out1) = (sign << 31) | ((uint32_t)e_biased << 24) | m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void float64_ibm64_encode(char **args, const npy_intp *dimensions,
                                  const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    for (i = 0; i < n; i++) {
        double val = *(double *)in1;

        if (val == 0.0) {
            *((uint64_t *)out1) = 0;
        } else {
            uint64_t sign = (val < 0) ? 1 : 0;
            double abs_val = fabs(val);

            double log2_val = log2(abs_val);
            int32_t hex_exp = (int32_t)ceil(log2_val / 4.0);

            double mant_frac = abs_val / pow(2.0, 4.0 * (double)hex_exp);

            if (mant_frac >= 1.0) {
                hex_exp++;
                mant_frac = abs_val / pow(2.0, 4.0 * (double)hex_exp);
            }

            /* 56-bit mantissa */
            uint64_t m_val = (uint64_t)round(mant_frac * pow(2.0, 56.0));
            if (m_val > 0x00FFFFFFFFFFFFFFULL) m_val = 0x00FFFFFFFFFFFFFFULL;

            int32_t e_biased = hex_exp + 64;
            if (e_biased < 0) e_biased = 0;
            if (e_biased > 127) e_biased = 127;

            *((uint64_t *)out1) = (sign << 63) | ((uint64_t)e_biased << 56) | m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* Function arrays and type signatures */

PyUFuncGenericFunction ibm32_decode_funcs[1] = {&uint32_ibm32_decode};
PyUFuncGenericFunction ibm64_decode_funcs[1] = {&uint64_ibm64_decode};
PyUFuncGenericFunction ibm32_encode_funcs[1] = {&float64_ibm32_encode};
PyUFuncGenericFunction ibm64_encode_funcs[1] = {&float64_ibm64_encode};

static char ibm32_decode_types[2] = {NPY_UINT32, NPY_FLOAT64};
static char ibm64_decode_types[2] = {NPY_UINT64, NPY_FLOAT64};
static char ibm32_encode_types[2] = {NPY_FLOAT64, NPY_UINT32};
static char ibm64_encode_types[2] = {NPY_FLOAT64, NPY_UINT64};
