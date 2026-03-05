/*
 * ti.c — Texas Instruments floating point 32-bit and 40-bit encode/decode ufuncs
 *
 * Layout: [exponent 8 | sign 1 | mantissa (23 or 31)]
 * Decode: value = ((-2)^s + m / 2^mbits) * 2^e
 *   s=0: value = (1 + m/2^mbits) * 2^e
 *   s=1: value = (-2 + m/2^mbits) * 2^e
 *   e == -128: value = 0
 */

/* --- Decode --- */

static void uint32_ti32_decode(char **args, const npy_intp *dimensions,
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

        uint32_t e_u = (dn >> 24) & 0xFF;
        uint32_t s = (dn >> 23) & 1;
        uint32_t m = dn & 0x7FFFFF;

        int32_t e = (int32_t)twoscomp((uint64_t)e_u, 8);

        if (e == -128) {
            *((double *)out1) = 0.0;
        } else {
            double S = (s == 0) ? 1.0 : -2.0;
            double M = (double)m / (double)(1 << 23);
            *((double *)out1) = (S + M) * pow(2.0, (double)e);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint64_ti40_decode(char **args, const npy_intp *dimensions,
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

        uint64_t e_u = (dn >> 32) & 0xFF;
        uint64_t s = (dn >> 31) & 1;
        uint64_t m = dn & 0x7FFFFFFFULL;

        int32_t e = (int32_t)twoscomp(e_u, 8);

        if (e == -128) {
            *((double *)out1) = 0.0;
        } else {
            double S = (s == 0) ? 1.0 : -2.0;
            double M = (double)m / (double)((uint64_t)1 << 31);
            *((double *)out1) = (S + M) * pow(2.0, (double)e);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* --- Encode --- */

static void float64_ti32_encode(char **args, const npy_intp *dimensions,
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
            /* e_field = 0x80 (-128 two's complement) */
            *((uint32_t *)out1) = (uint32_t)0x80 << 24;
        } else {
            double abs_val = fabs(val);
            uint32_t sign = (val < 0) ? 1 : 0;
            int32_t e = (int32_t)floor(log2(abs_val));

            double scaled = val / pow(2.0, (double)e);
            double m_frac = (sign == 0) ? (scaled - 1.0) : (scaled + 2.0);
            int32_t m_val = (int32_t)round(m_frac * (double)(1 << 23));

            if (m_val < 0) m_val = 0;
            if (m_val > 0x7FFFFF) m_val = 0x7FFFFF;
            if (e < -127) e = -127;
            if (e > 127) e = 127;

            uint32_t e_u = (uint32_t)e & 0xFF;
            *((uint32_t *)out1) = (e_u << 24) | (sign << 23) | (uint32_t)m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void float64_ti40_encode(char **args, const npy_intp *dimensions,
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
            *((uint64_t *)out1) = (uint64_t)0x80 << 32;
        } else {
            double abs_val = fabs(val);
            uint64_t sign = (val < 0) ? 1 : 0;
            int32_t e = (int32_t)floor(log2(abs_val));

            double scaled = val / pow(2.0, (double)e);
            double m_frac = (sign == 0) ? (scaled - 1.0) : (scaled + 2.0);
            int64_t m_val = (int64_t)round(m_frac * (double)((uint64_t)1 << 31));

            if (m_val < 0) m_val = 0;
            if (m_val > 0x7FFFFFFFLL) m_val = 0x7FFFFFFFLL;
            if (e < -127) e = -127;
            if (e > 127) e = 127;

            uint64_t e_u = (uint64_t)e & 0xFF;
            *((uint64_t *)out1) = (e_u << 32) | (sign << 31) | (uint64_t)m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* Function arrays and type signatures */

PyUFuncGenericFunction ti32_decode_funcs[1] = {&uint32_ti32_decode};
PyUFuncGenericFunction ti40_decode_funcs[1] = {&uint64_ti40_decode};
PyUFuncGenericFunction ti32_encode_funcs[1] = {&float64_ti32_encode};
PyUFuncGenericFunction ti40_encode_funcs[1] = {&float64_ti40_encode};

static char ti32_decode_types[2] = {NPY_UINT32, NPY_FLOAT64};
static char ti40_decode_types[2] = {NPY_UINT64, NPY_FLOAT64};
static char ti32_encode_types[2] = {NPY_FLOAT64, NPY_UINT32};
static char ti40_encode_types[2] = {NPY_FLOAT64, NPY_UINT64};
