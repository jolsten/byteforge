/*
 * dec.c — DEC VAX floating point encode/decode ufuncs
 *
 * DECFloat (F4/D4): [sign 1 | exponent 8 | mantissa (23 or 55)], bias=128
 * DECFloatG (G4):   [sign 1 | exponent 11 | mantissa 52], bias=1024
 *
 * Decode: value = (-1)^s * (m / 2^(mant_bits+1) + 0.5) * 2^(e - bias)
 *   e == 0: value = 0 (reserved)
 */

/* --- Decode --- */

static void uint32_dec32_decode(char **args, const npy_intp *dimensions,
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
        uint32_t e = (dn >> 23) & 0xFF;
        uint32_t m = dn & 0x7FFFFF;

        if (e == 0) {
            *((double *)out1) = 0.0;
        } else {
            double sign = (s == 0) ? 1.0 : -1.0;
            double M = (double)m / (double)(1 << 24) + 0.5;
            int32_t E = (int32_t)e - 128;
            *((double *)out1) = sign * M * pow(2.0, (double)E);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint64_dec64_decode(char **args, const npy_intp *dimensions,
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
        uint64_t e = (dn >> 55) & 0xFF;
        uint64_t m = dn & 0x007FFFFFFFFFFFFFULL;

        if (e == 0) {
            *((double *)out1) = 0.0;
        } else {
            double sign = (s == 0) ? 1.0 : -1.0;
            /* 55-bit mantissa: m / 2^56 + 0.5 */
            double M = (double)m / pow(2.0, 56.0) + 0.5;
            int32_t E = (int32_t)e - 128;
            *((double *)out1) = sign * M * pow(2.0, (double)E);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint64_dec64g_decode(char **args, const npy_intp *dimensions,
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
        uint64_t e = (dn >> 52) & 0x7FF;
        uint64_t m = dn & 0x000FFFFFFFFFFFFFULL;

        if (e == 0) {
            *((double *)out1) = 0.0;
        } else {
            double sign = (s == 0) ? 1.0 : -1.0;
            /* 52-bit mantissa: m / 2^53 + 0.5 */
            double M = (double)m / (double)((uint64_t)1 << 53) + 0.5;
            int32_t E = (int32_t)e - 1024;
            *((double *)out1) = sign * M * pow(2.0, (double)E);
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* --- Encode --- */

static void float64_dec32_encode(char **args, const npy_intp *dimensions,
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

            int32_t e_unbiased = (int32_t)floor(log2(abs_val)) + 1;
            double significand = abs_val / pow(2.0, (double)e_unbiased);

            double m_frac = significand - 0.5;
            uint32_t m_val = (uint32_t)round(m_frac * (double)(1 << 24));
            if (m_val > 0x7FFFFF) m_val = 0x7FFFFF;

            int32_t e_biased = e_unbiased + 128;
            if (e_biased < 1) e_biased = 1;
            if (e_biased > 255) e_biased = 255;

            *((uint32_t *)out1) = (sign << 31) | ((uint32_t)e_biased << 23) | m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void float64_dec64_encode(char **args, const npy_intp *dimensions,
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

            int32_t e_unbiased = (int32_t)floor(log2(abs_val)) + 1;
            double significand = abs_val / pow(2.0, (double)e_unbiased);

            double m_frac = significand - 0.5;
            uint64_t m_val = (uint64_t)round(m_frac * pow(2.0, 56.0));
            if (m_val > 0x007FFFFFFFFFFFFFULL) m_val = 0x007FFFFFFFFFFFFFULL;

            int32_t e_biased = e_unbiased + 128;
            if (e_biased < 1) e_biased = 1;
            if (e_biased > 255) e_biased = 255;

            *((uint64_t *)out1) = (sign << 63) | ((uint64_t)e_biased << 55) | m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void float64_dec64g_encode(char **args, const npy_intp *dimensions,
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

            int32_t e_unbiased = (int32_t)floor(log2(abs_val)) + 1;
            double significand = abs_val / pow(2.0, (double)e_unbiased);

            double m_frac = significand - 0.5;
            uint64_t m_val = (uint64_t)round(m_frac * (double)((uint64_t)1 << 53));
            if (m_val > 0x000FFFFFFFFFFFFFULL) m_val = 0x000FFFFFFFFFFFFFULL;

            int32_t e_biased = e_unbiased + 1024;
            if (e_biased < 1) e_biased = 1;
            if (e_biased > 2047) e_biased = 2047;

            *((uint64_t *)out1) = (sign << 63) | ((uint64_t)e_biased << 52) | m_val;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* Function arrays and type signatures */

PyUFuncGenericFunction dec32_decode_funcs[1] = {&uint32_dec32_decode};
PyUFuncGenericFunction dec64_decode_funcs[1] = {&uint64_dec64_decode};
PyUFuncGenericFunction dec64g_decode_funcs[1] = {&uint64_dec64g_decode};
PyUFuncGenericFunction dec32_encode_funcs[1] = {&float64_dec32_encode};
PyUFuncGenericFunction dec64_encode_funcs[1] = {&float64_dec64_encode};
PyUFuncGenericFunction dec64g_encode_funcs[1] = {&float64_dec64g_encode};

static char dec32_decode_types[2] = {NPY_UINT32, NPY_FLOAT64};
static char dec64_decode_types[2] = {NPY_UINT64, NPY_FLOAT64};
static char dec64g_decode_types[2] = {NPY_UINT64, NPY_FLOAT64};
static char dec32_encode_types[2] = {NPY_FLOAT64, NPY_UINT32};
static char dec64_encode_types[2] = {NPY_FLOAT64, NPY_UINT64};
static char dec64g_encode_types[2] = {NPY_FLOAT64, NPY_UINT64};
