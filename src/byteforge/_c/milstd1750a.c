/*
 * milstd1750a.c — MIL-STD-1750A 32-bit and 48-bit encode/decode ufuncs
 */

/* --- Decode --- */

static void uint32_milstd1750a32_decode(char **args, const npy_intp *dimensions,
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

        uint32_t m_raw = (dn >> 8) & 0xFFFFFF;
        uint32_t e_raw = dn & 0xFF;

        int32_t M = (int32_t)twoscomp((uint64_t)m_raw, 24);
        int32_t E = (int32_t)twoscomp((uint64_t)e_raw, 8);

        *((double *)out1) = ((double)M / (double)(1 << 23)) * pow(2.0, (double)E);

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint64_milstd1750a48_decode(char **args, const npy_intp *dimensions,
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

        /* 48-bit layout: [M_hi24 | E8 | M_lo16] */
        uint64_t m_hi = (dn >> 24) & 0xFFFFFF;
        uint64_t e_raw = (dn >> 16) & 0xFF;
        uint64_t m_lo = dn & 0xFFFF;
        uint64_t m_raw = (m_hi << 16) | m_lo;

        int64_t M = twoscomp(m_raw, 40);
        int32_t E = (int32_t)twoscomp(e_raw, 8);

        *((double *)out1) = ((double)M / (double)((int64_t)1 << 39)) * pow(2.0, (double)E);

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* --- Encode --- */

static void float64_milstd1750a32_encode(char **args, const npy_intp *dimensions,
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
            double abs_val = fabs(val);
            int32_t E = (int32_t)floor(log2(abs_val)) + 1;
            int32_t M = (int32_t)round(val * pow(2.0, (double)(23 - E)));

            /* Clamp */
            if (M < -(1 << 23)) M = -(1 << 23);
            if (M > (1 << 23) - 1) M = (1 << 23) - 1;
            if (E < -128) E = -128;
            if (E > 127) E = 127;

            uint32_t m_u = (uint32_t)M & 0xFFFFFF;
            uint32_t e_u = (uint32_t)E & 0xFF;
            *((uint32_t *)out1) = (m_u << 8) | e_u;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void float64_milstd1750a48_encode(char **args, const npy_intp *dimensions,
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
            double abs_val = fabs(val);
            int32_t E = (int32_t)floor(log2(abs_val)) + 1;
            int64_t M = (int64_t)round(val * pow(2.0, (double)(39 - E)));

            /* Clamp */
            int64_t m_min = -((int64_t)1 << 39);
            int64_t m_max = ((int64_t)1 << 39) - 1;
            if (M < m_min) M = m_min;
            if (M > m_max) M = m_max;
            if (E < -128) E = -128;
            if (E > 127) E = 127;

            uint64_t m_u = (uint64_t)M & 0xFFFFFFFFFFULL;
            uint64_t e_u = (uint64_t)E & 0xFF;

            /* 48-bit: [M_hi24 | E8 | M_lo16] */
            uint64_t m_hi = (m_u >> 16) & 0xFFFFFF;
            uint64_t m_lo = m_u & 0xFFFF;
            *((uint64_t *)out1) = (m_hi << 24) | (e_u << 16) | m_lo;
        }

        in1 += in1_step;
        out1 += out1_step;
    }
}

/* Function arrays and type signatures */

PyUFuncGenericFunction milstd1750a32_decode_funcs[1] = {&uint32_milstd1750a32_decode};
PyUFuncGenericFunction milstd1750a48_decode_funcs[1] = {&uint64_milstd1750a48_decode};
PyUFuncGenericFunction milstd1750a32_encode_funcs[1] = {&float64_milstd1750a32_encode};
PyUFuncGenericFunction milstd1750a48_encode_funcs[1] = {&float64_milstd1750a48_encode};

static char milstd1750a32_decode_types[2] = {NPY_UINT32, NPY_FLOAT64};
static char milstd1750a48_decode_types[2] = {NPY_UINT64, NPY_FLOAT64};
static char milstd1750a32_encode_types[2] = {NPY_FLOAT64, NPY_UINT32};
static char milstd1750a48_encode_types[2] = {NPY_FLOAT64, NPY_UINT64};
